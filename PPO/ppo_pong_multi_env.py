import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import cv2
import time
from collections import deque
import ale_py

# === 1. 环境配置与预处理 ===
class AtariPreprocess(gym.Wrapper):
    """
    手动实现的预处理：
    1. 灰度化 + 缩放 (84x84)
    2. 归一化 (0-1)
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (84, 84), dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._process_obs(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_obs(obs), info

    def _process_obs(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized / 255.0

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """
        返回每 skip 帧后的观测结果
        同时对最后两帧做最大化处理（解决Atari画面闪烁问题）
        """
        super().__init__(env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """重复执行动作 skip 次，累加奖励"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # 记录最后两帧用于处理闪烁
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            
            total_reward += reward
            if done: break
            
        # Max pooling: Atari游戏有时会隔帧闪烁，取两帧最大值可以补全画面
        max_frame = self._obs_buffer.max(axis=0)
        
        return max_frame, total_reward, terminated, truncated, info

def make_env(env_id, idx, run_name):
    """工厂函数：用于创建每一个独立的子环境"""
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        # RecordEpisodeStatistics 自动帮我们统计真实分数(Reward Sum)和长度
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # 这会让 1 个 step 对应 4 帧游戏画面，和 Pong-v4 效果一样，但由你控制
        env = MaxAndSkipEnv(env, skip=4)
        env = AtariPreprocess(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        return env
    return thunk

# === 2. 神经网络 (Actor-Critic) ===
# 保持单网络结构，但能够处理 Batch 数据
class ActorCritic(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 84, 84)
            self.fc_input_dim = self.conv(dummy).shape[1]

        self.fc = nn.Linear(self.fc_input_dim, 512)
        self.actor = nn.Linear(512, num_actions)
        self.critic = nn.Linear(512, 1)

    def get_value(self, x):
        x = self.conv(x)
        x = F.relu(self.fc(x))
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = self.conv(x)
        x = F.relu(self.fc(x))
        logits = self.actor(x)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

# === 3. 超参数设置 ===
ARGS = {
    "env_id": "PongNoFrameskip-v4",
    "num_envs": 8,           # 并行环境数量 (根据CPU核数调整，8-16通常较好)
    "total_timesteps": 3_500_000,
    "learning_rate": 2.5e-4,
    "num_steps": 128,        # 每个环境采样的步数
    "batch_size": 8 * 128,   # num_envs * num_steps = 1024
    "minibatch_size": 256,   # 每次更新从buffer里取多少数据
    "update_epochs": 4,      # PPO更新循环次数
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_coef": 0.1,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "seed": 42,
    "anneal_lr": False, # 新增：是否衰减学习率
}

# === 4. 训练主循环 ===
def train():
    run_name = f"Pong_PPO_{int(time.time())}"

    # 优化 CUDA
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # 1. 设置随机种子
    np.random.seed(ARGS["seed"])
    torch.manual_seed(ARGS["seed"])
    
    # 2. 创建并行环境
    # SyncVectorEnv 会同步运行多个环境， step() 返回的是数组
    envs = gym.vector.SyncVectorEnv(
        [make_env(ARGS["env_id"], i, run_name) for i in range(ARGS["num_envs"])]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Envs: {ARGS['num_envs']}")

    agent = ActorCritic(envs.single_action_space.n).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=ARGS["learning_rate"], eps=1e-5)

    # 3. 初始化存储 Buffer
    # 注意维度：(Steps, Envs, ...)
    obs = torch.zeros((ARGS["num_steps"], ARGS["num_envs"]) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((ARGS["num_steps"], ARGS["num_envs"])).to(device)
    logprobs = torch.zeros((ARGS["num_steps"], ARGS["num_envs"])).to(device)
    rewards = torch.zeros((ARGS["num_steps"], ARGS["num_envs"])).to(device)
    dones = torch.zeros((ARGS["num_steps"], ARGS["num_envs"])).to(device)
    values = torch.zeros((ARGS["num_steps"], ARGS["num_envs"])).to(device)

    # 记录分数的队列
    score_window = deque(maxlen=100) # 记录最近100局的分数
    current_episode_rewards = np.zeros(ARGS["num_envs"])

    # 初始化第一次观测
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=ARGS["seed"])
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(ARGS["num_envs"]).to(device)

    num_updates = ARGS["total_timesteps"] // ARGS["batch_size"]

    print(f"{'Step':>8} | {'FPS':>5} | {'Score':>8} | {'Entr':>6} | {'V_Loss':>8} | {'P_Loss':>8} | {'ExpVar':>6} | {'Clip%':>6}")
    print("-" * 90)

    for update in range(1, num_updates + 1):
        # === 学习率衰减 ===
        if ARGS["anneal_lr"]:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * ARGS["learning_rate"]
            optimizer.param_groups[0]["lr"] = lrnow
        else:
            lrnow = ARGS["learning_rate"]
 
        # === A. 数据收集 (Rollout) ===
        for step in range(ARGS["num_steps"]):
            global_step += 1 * ARGS["num_envs"]
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob

            # 执行动作 (SyncVectorEnv 会自动处理 Reset，如果 done=True)
            real_next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(real_next_obs).to(device)
            next_done = torch.Tensor(done).to(device)

            current_episode_rewards += reward

            # done 是一个布尔数组，比如 [False, True, False, ...]
            for i in range(ARGS["num_envs"]):
                if done[i]:
                    # 如果第 i 个环境结束了，把它的总分记下来
                    score_window.append(current_episode_rewards[i])
                    # 重置该环境的分数归零，准备下一局
                    current_episode_rewards[i] = 0


        # === B. 计算优势函数 (GAE) ===
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(ARGS["num_steps"])):
                if t == ARGS["num_steps"] - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + ARGS["gamma"] * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + ARGS["gamma"] * ARGS["gae_lambda"] * nextnonterminal * lastgaelam
            returns = advantages + values

        # === C. 数据展平 (Flatten) ===
        # 把 (Steps, Envs) 展平成 (Batch_Size)
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # === D. PPO 更新 (Update) ===
        b_inds = np.arange(ARGS["batch_size"])
        clipfracs = []
        
        for epoch in range(ARGS["update_epochs"]):
            np.random.shuffle(b_inds) # 打乱数据
            for start in range(0, ARGS["batch_size"], ARGS["minibatch_size"]):
                end = start + ARGS["minibatch_size"]
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # 调试用：查看有多少被clip了
                with torch.no_grad():
                    clipfracs += [((ratio - 1.0).abs() > ARGS["clip_coef"]).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                # 优势归一化 (重要技巧)
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy Loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - ARGS["clip_coef"], 1 + ARGS["clip_coef"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy Loss
                entropy_loss = entropy.mean()

                # Total Loss
                loss = pg_loss - ARGS["ent_coef"] * entropy_loss + ARGS["vf_coef"] * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), ARGS["max_grad_norm"])
                optimizer.step()

        # === E. 打印日志 ===
        if update % 5 == 0: # 每5次更新打印一次
            fps = int(global_step / (time.time() - start_time))
            avg_score = np.mean(score_window) if len(score_window) > 0 else 0.0
            # 1. 计算 Explained Variance (解释方差)
            # 公式: 1 - Var(y_true - y_pred) / Var(y_true)
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            # 2. 计算平均 Clip Fraction
            avg_clip_frac = np.mean(clipfracs)
            print(f"{global_step:>8} | {fps:>5} | {avg_score:>8.2f} | {entropy_loss.item():>6.2f} | {v_loss.item():>8.4f} | {pg_loss.item():>8.4f} | {explained_var:>6.4f} | {avg_clip_frac:>6.4f}")

    # 保存模型
    torch.save(agent.state_dict(), "ppo_pong_multi_env.pth")
    envs.close()
    print("训练完成！模型已保存。")

if __name__ == "__main__":
    train()