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
import os

# === 1. 环境配置 (使用官方 Wrapper 替代手动实现) ===
def make_env(env_id, seed, run_name, idx, capture_video=True, video_freq=100):
    def thunk():
        # 1. 创建基础环境
        env = gym.make(env_id, render_mode="rgb_array")

        # === 新增：视频录制 Wrapper ===
        # 关键位置：必须放在 AtariPreprocessing 之前，这样录下来的才是彩色原画
        # 关键逻辑：if idx == 0，只让第一个环境录像，避免并行环境冲突
        if capture_video and idx == 0:
            video_folder = f"videos/{run_name}"
            if not os.path.exists(video_folder):
                os.makedirs(video_folder)
            
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=video_folder,
                # 设置录制频率：这里是每 video_freq 个 episode 录制一次
                # 也会自动录制第 0 个 episode
                episode_trigger=lambda x: x % video_freq == 0,
                name_prefix=f"rl-video",
                # disable_logger=True 可以减少一些控制台噪音
                disable_logger=True
            )
        
        # 2. 统计真实回报 (必须在 ClipReward 之前)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        # 3. 官方 Atari 预处理 (包含了 NoopReset, MaxAndSkip, Resize, Grayscale, Scale)
        # scale_obs=True 会自动将像素除以 255 归一化
        env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, 
                                              screen_size=84, terminal_on_life_loss=False, 
                                              grayscale_obs=True, scale_obs=True)
        
        # 4. 关键：奖励截断 (将奖励限制在 -1 到 1 之间，稳定训练)
        env = gym.wrappers.TransformReward(env, lambda r: np.sign(r))
        
        # 5. 堆叠帧 (4帧)
        env = gym.wrappers.FrameStackObservation(env, 4)
        
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
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
    "env_id": "BreakoutNoFrameskip-v4",
    "num_envs": 8,           # 并行环境数量 (根据CPU核数调整，8-16通常较好)
    "total_timesteps": 10_000_000,
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
    run_name = f"Breakout_PPO_{int(time.time())}"

    # 优化 CUDA
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # 1. 设置随机种子
    np.random.seed(ARGS["seed"])
    torch.manual_seed(ARGS["seed"])
    
    # 2. 创建并行环境
    # SyncVectorEnv 会同步运行多个环境， step() 返回的是数组
    envs = gym.vector.SyncVectorEnv(
        [make_env(ARGS["env_id"], ARGS["seed"] + i, run_name, i) for i in range(ARGS["num_envs"])]
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
    torch.save(agent.state_dict(), "ppo_breakout_multi_env.pth")
    envs.close()
    print("训练完成！模型已保存。")

if __name__ == "__main__":
    train()