import gymnasium as gym
import ale_py
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# ==========================================
# 超参数 (Hyperparameters)
# ==========================================
# 注意：Pong 的 Policy Gradient 收敛非常慢，600轮肯定不够
# 建议至少 3000-5000 轮才能看到明显效果
NUM_EPOCHES = 5000  
BATCH_SIZE = 64    # 每 32 局更新一次参数
GAMMA = 0.99
LR = 1e-4

# 结果保存路径
result_dir = "./results"
video_dir = os.path.join(result_dir, "videos")
curve_dir = os.path.join(result_dir, "curves")
os.makedirs(video_dir, exist_ok=True)
os.makedirs(curve_dir, exist_ok=True)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 环境配置
# ==========================================
def make_env_with_video(env_name, video_folder):
    # render_mode 设为 rgb_array 用于录像
    env = gym.make(env_name, render_mode="rgb_array")
    
    # 录像触发器：每 100 轮录一次
    trigger = lambda ep_id: ep_id > 0 and ep_id % 100 == 0
    
    env = gym.wrappers.RecordVideo(
        env, 
        video_folder=video_folder, 
        episode_trigger=trigger,
        name_prefix="pg-pong",
        disable_logger=True
    )
    # 标准 Atari 预处理
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, 
                                          screen_size=84, terminal_on_life_loss=True, 
                                          grayscale_obs=True, scale_obs=True)
    env = gym.wrappers.FrameStackObservation(env, 4)
    return env

# ==========================================
# 网络定义
# ==========================================
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        # 输入 4帧, 输出 2个动作 (Up/Down)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        # 修正：输出改为 2，专门对应 Up 和 Down
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

# ==========================================
# 辅助函数
# ==========================================
def process_state(obs):
    """(4, 84, 84) numpy -> (1, 4, 84, 84) tensor on device"""
    state = np.array(obs)
    state = torch.from_numpy(state).unsqueeze(0).float().to(device)
    return state

def plot_and_save(rewards, losses, save_path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Reward")
    if len(rewards) >= 20:
        # 简单的移动平均
        means = np.convolve(rewards, np.ones(20)/20, mode='valid')
        plt.plot(range(19, len(rewards)), means, label="Avg (20)", color='red')
    plt.title("Episode Rewards")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(losses, color='orange', alpha=0.6)
    plt.title("Training Loss")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ==========================================
# 训练主循环
# ==========================================
def train():
    env_name = "PongNoFrameskip-v4"
    env = make_env_with_video(env_name, video_dir)
    
    # 初始化网络，输出维度强制设为 2
    policy = PolicyNetwork().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    
    # 缓冲池：存储多个 Episode 的数据用于一次 Update
    batch_log_probs = []
    batch_rewards = []
    batch_entropies = []
    
    episode_rewards = []
    episode_losses = []
    running_reward = None

    print(f"训练开始！Device: {device}")
    
    for i_episode in range(1, NUM_EPOCHES + 1):
        obs, info = env.reset()
        state = process_state(obs)
        
        ep_rewards = []     # 记录当前这一局的每一步 reward
        ep_log_probs = []   # 记录当前这一局的每一步 log_prob
        ep_entropies = []   # 记录当前这一局的每一步 entropy
        
        # 修正 1: 增加 while not done 循环
        while True:
            # 1. 预测
            probs = policy(state)
            m = Categorical(probs)
            action_index = m.sample()
            
            # 2. 映射动作 (0->Right/Up=2, 1->Left/Down=3)
            # 这样概率分布就是 50/50 起步，不会有偏见
            actual_action = 2 if action_index.item() == 0 else 3
            
            # 3. 执行
            next_obs, reward, terminated, truncated, info = env.step(actual_action)
            done = terminated or truncated
            
            # 4. 存储
            ep_log_probs.append(m.log_prob(action_index))
            ep_entropies.append(m.entropy()) # 【新增】收集熵
            ep_rewards.append(reward)
            
            state = process_state(next_obs)
            
            if done:
                break
        
        # --- Episode 结束 ---
        
        # 1. 计算这一局的 Returns (修正算法错误：单局独立计算)
        R = 0
        returns = []
        for r in reversed(ep_rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        
        # 2. 将这一局的数据加入 Batch 缓冲
        batch_log_probs.extend(ep_log_probs)
        batch_entropies.extend(ep_entropies)
        batch_rewards.extend(returns)
        
        # 3. 记录 Reward
        ep_total_reward = sum(ep_rewards)
        episode_rewards.append(ep_total_reward)
        
        if running_reward is None:
            running_reward = ep_total_reward
        else:
            running_reward = 0.99 * running_reward + 0.01 * ep_total_reward
            
        # 4. 执行更新 (每 BATCH_SIZE 局更新一次)
        if i_episode % BATCH_SIZE == 0:
            optimizer.zero_grad()
            
            # 拼接
            # batch_log_probs 是 list of scalars (tensors)
            # batch_rewards 是 list of scalars (tensors)
            
            # 堆叠成一维向量
            log_probs_tensor = torch.stack(batch_log_probs).view(-1)
            entropies_tensor = torch.stack(batch_entropies).view(-1)
            returns_tensor = torch.stack(batch_rewards).view(-1)
            
            # 计算 Loss: - sum(log_prob * return)
            # 因为是 Batch，通常取 mean 会让学习率更稳定，但 sum 也可以（取决于lr大小）
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
            loss_policy = -torch.mean(log_probs_tensor * returns_tensor)
            loss_entropy = -0.01 * entropies_tensor.mean()
            loss = loss_policy + loss_entropy
            loss.backward()

            optimizer.step()
            
            episode_losses.append(loss.item())
            
            # 清空 Batch
            batch_log_probs = []
            batch_rewards = []
            batch_entropies = []
            
            print(f"Ep: {i_episode}, Reward: {ep_total_reward}, PolicyL: {loss_policy.item():.5f}, EntropyL: {loss_entropy.item():.5f}")
            
            # 绘图
            plot_and_save(episode_rewards, episode_losses, os.path.join(curve_dir, "training_PG_curve.png"))
            
            # 保存模型
            if running_reward > 18: 
                torch.save(policy.state_dict(), os.path.join(result_dir, 'pong_pg_solved.pth'))
                print("Solved! Model saved.")

    env.close()
    print("训练结束。")

if __name__ == "__main__":
    train()