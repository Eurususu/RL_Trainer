import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import cv2
from torch.distributions import Categorical
import ale_py


# ---1. 环境预处理 Wapper ----
# 将Atari环境的图片转化为84x84的灰度图，并进行归一化处理
class AtariPreprocess(gym.Wrapper):
    def __init__(self, env):
        super(AtariPreprocess, self).__init__(env)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (1, 84, 84), dtype=np.float32)
    
    def step(self, action):
        obs, reward, terminated, truncatedm, info = self.env.step(action)
        return self._process_obs(obs), reward, terminated, truncatedm, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_obs(obs), info

    def _process_obs(self, obs):
        # 转换为灰度图
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # 调整大小为84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        # 归一化到[0, 1]
        normalized = resized / 255.0
        # 添加通道维度
        return np.expand_dims(normalized, axis=0)
    

# ---2. 定义PPO网络结构 ----
class PPOAgent(nn.Module):
    def __init__(self, action_dim):
        super(PPOAgent, self).__init__()
        self.conv = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=8, stride=4), nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        nn.Flatten()
        )
        # 计算卷积后的扁平化维度。输入84x84经过上述卷积约为 3136
        with torch.no_grad():
            dummy = torch.zeros(1,1,84,84)
            self.fc_input_dim = self.conv(dummy).shape[1]
        
        self.fc = nn.Linear(self.fc_input_dim, 512)
        # Actor head
        self.actor = nn.Linear(512, action_dim)
        # Critic head
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.fc(x))
        return x
    
    def get_action_and_value(self, x, action=None):
        feature =self.forward(x)
        logits = self.actor(feature)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        state_value = self.critic(feature)
        
        return action, probs.log_prob(action), probs.entropy(), state_value
    

# ---3. 超参数配置 ---
HYPERPARAMS = {
    "lr": 2.5e-4,
    "gamma": 0.99,          # 折扣因子
    "gae_lambda": 0.95,     # GAE 参数
    "clip_coef": 0.1,       # PPO 剪切范围
    "ent_coef": 0.01,       # 熵系数
    "vf_coef": 0.5,         # 价值损失系数
    "batch_size": 32 * 128, # 这里为了简化，假设单环境跑多步，实际通常用并行环境
    "update_epochs": 4,     # 每次更新循环几次
    "total_timesteps": 2000000,
    "steps_per_rollout": 128, # 每次收集多少步数据
}

# --- 4. 训练主循环 ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建环境
    env = gym.make("PongNoFrameskip-v4")
    env = AtariPreprocess(env)

    model = PPOAgent(action_dim=env.action_space.n).to(device)
    optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMS["lr"], eps=1e-5)

    # 存储数据的 buffer
    obs = torch.zeros((HYPERPARAMS["steps_per_rollout"], 1, 84, 84)).to(device)
    actions = torch.zeros((HYPERPARAMS["steps_per_rollout"])).to(device)
    logprobs = torch.zeros((HYPERPARAMS["steps_per_rollout"])).to(device)
    rewards = torch.zeros((HYPERPARAMS["steps_per_rollout"])).to(device)
    dones = torch.zeros((HYPERPARAMS["steps_per_rollout"])).to(device)
    values = torch.zeros((HYPERPARAMS["steps_per_rollout"])).to(device)

    # 初始化环境
    next_obs, _ = env.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(1).to(device)

    global_step = 0
    num_updates = HYPERPARAMS["total_timesteps"] // HYPERPARAMS["steps_per_rollout"]

    for update in range(1, num_updates + 1):
        # === 阶段 1: 收集数据 (Rollout) ===
        for step in range(HYPERPARAMS["steps_per_rollout"]):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done
            # 用此时的策略（旧策略）跑了一遍游戏，记录下了动作 actions 和当时的概率 logprobs
            with torch.no_grad():
                # 获取动作和价值，不需要梯度
                action, logprob, _, value = model.get_action_and_value(next_obs.unsqueeze(0))
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob

            # 执行动作
            real_next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(real_next_obs).to(device)
            next_done = torch.Tensor([done]).to(device)

            if done:
                real_next_obs, _ = env.reset()
                next_obs = torch.Tensor(real_next_obs).to(device)

        # === 阶段 2: 计算优势函数 (GAE) - PPO的核心 ===
        with torch.no_grad():
            next_value = model.get_action_and_value(next_obs.unsqueeze(0))[3].reshape(1,-1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(HYPERPARAMS["steps_per_rollout"])):
                if t == HYPERPARAMS["steps_per_rollout"] - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                # 现实发生了什么（reward + 下一刻的局面好坏）减去 你的直觉。如果 delta > 0，说明现实比你想象的更好（意外之喜）。
                # TD Error: 实际奖励 + 折扣后的未来价值 - 当前预测价值  values[t]->你的直觉（Critic）觉得这一步当时能拿几分。
                delta = rewards[t] + HYPERPARAMS["gamma"] * nextvalues * nextnonterminal - values[t]
                # GAE 公式
                # 优势函数计算，把这些“意外之喜”累加起来
                advantages[t] = lastgaelam = delta + HYPERPARAMS["gamma"] * HYPERPARAMS["gae_lambda"] * nextnonterminal * lastgaelam
            returns = advantages + values

        # === 阶段 3: 更新网络 (Optimization) ===
        # 将 buffer 展平
        b_obs = obs.reshape((-1,) + env.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        for epoch in range(HYPERPARAMS["update_epochs"]):
            # 获取当前的新概率和新价值
            _, new_logprob, entropy, new_value = model.get_action_and_value(b_obs, b_actions)
            # 1. 计算比率 (Ratio): pi_new / pi_old
            # log(a/b) = log(a) - log(b) -> a/b = exp(log(a) - log(b))
            logratio = new_logprob - b_logprobs
            ratio = logratio.exp()

            # 2. 归一化 Advantage (重要技巧，加速收敛)
            mb_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            # 3. 计算 PPO 损失 (Clipped Surrogate Objective)
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - HYPERPARAMS["clip_coef"], 1 + HYPERPARAMS["clip_coef"])
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # 4. 计算 Value 损失 (MSE)
            new_value = new_value.view(-1)
            v_loss = 0.5 * ((new_value - b_returns) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - HYPERPARAMS["ent_coef"] * entropy_loss + HYPERPARAMS["vf_coef"] * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5) # 梯度裁剪，防止爆炸
            optimizer.step()

        if update % 10 == 0:
            print(f"Update: {update}/{num_updates}, Loss: {loss.item():.4f}, Value Mean: {b_values.mean().item():.4f}")
    
    # 保存
    torch.save(model.state_dict(), "manual_ppo_pong.pth")
    env.close()


if __name__ == "__main__":
    train()
