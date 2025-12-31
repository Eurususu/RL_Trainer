import gymnasium as gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import ale_py
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ==========================================
# 环境配置
# ==========================================
def make_env_with_video(env_name, video_folder):
    env = gym.make(env_name, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env, 
        video_folder=video_folder, 
        episode_trigger=lambda episode_id: episode_id % EVERY_EPOCHES == 0, # 每50轮录像
        name_prefix="dqn-breakout",
        disable_logger=True
    )
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, 
                                          screen_size=84, terminal_on_life_loss=False, 
                                          grayscale_obs=True, scale_obs=True)
    env = gym.wrappers.FrameStackObservation(env, 4)
    return env

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 超参数
# ==========================================
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY = 100000 
TAU = 0.005
LR = 1e-4
MEMORY_SIZE = 50000
NUM_EPOCHES = 5001
EVERY_EPOCHES = 500

# ==========================================
# 经验回放
# ==========================================
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

# ==========================================
# CNN 模型
# ==========================================
class CNN_DQN(nn.Module):
    def __init__(self, n_actions):
        super(CNN_DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x.to(device) # 确保在 forward 内部移动到设备
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        return self.head(x)


class Dueling_DQN(nn.Module):
    def __init__(self, n_actions):
        super(Dueling_DQN, self).__init__()
        # 卷积层不变
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # --- 变化在这里 ---
        # 展平后的维度是 3136 (64*7*7)
        
        # 1. Advantage 流 (计算每个动作的优势)
        self.fc_adv = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
        # 2. Value 流 (计算当前状态的整体价值)
        self.fc_val = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) 
        
        # 分别通过两个流
        adv = self.fc_adv(x) # [Batch, n_actions]
        val = self.fc_val(x) # [Batch, 1]
        
        # 组合: Q = V + (A - mean(A))
        return val + (adv - adv.mean(dim=1, keepdim=True))

# ==========================================
# 初始化
# ==========================================
env_name = "BreakoutNoFrameskip-v4"
# 这里先创建目录，防止 make_env 报错（虽然 RecordVideo 通常会自动创建）
result_dir = "./results"
video_dir = os.path.join(result_dir, "videos")
weight_dir = os.path.join(result_dir, "weights")
curve_dir = os.path.join(result_dir, "curves")
os.makedirs(video_dir, exist_ok=True)
os.makedirs(weight_dir, exist_ok=True)
os.makedirs(curve_dir, exist_ok=True)

env = make_env_with_video(env_name, video_dir)
n_actions = env.action_space.n

policy_net = Dueling_DQN(n_actions).to(device)
target_net = Dueling_DQN(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(MEMORY_SIZE)
steps_done = 0

# ==========================================
# 关键修复 1: 补充 process_state 函数
# ==========================================
def process_state(obs):
    """将 Gym 的 LazyFrame (numpy) 转换为 PyTorch Tensor"""
    # obs 是 (4, 84, 84) 的 numpy 数组
    # 转换为 Tensor，增加 Batch 维度 (1, 4, 84, 84)，转为 float32
    state = np.array(obs)
    state = torch.from_numpy(state).unsqueeze(0).float().to(device)
    return state

# ==========================================
# 工具函数
# ==========================================
def select_action(state, current_eps):
    """修改：将 epsilon 作为参数传入，解决作用域问题"""
    sample = random.random()
    if sample > current_eps:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return None
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        # 1. 使用 Policy Net 选择动作 (argmax a)
        # policy_net 输出的是 [Batch, Actions]，max(1)[1] 取索引
        # unsqueeze(1) 是为了变成列向量，方便 gather
        next_actions = policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
        # 2. 使用 Target Net 评估该动作的价值
        # gather(1, next_actions) 会根据上面选出的索引，去 Target Net 的输出里查 Q 值
        next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_actions).squeeze(1)
    
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()

def plot_and_save(rewards, losses, save_path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Episode Rewards")
    plt.plot(rewards, label="Reward")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    if len(rewards) >= 10:
        means = [np.mean(rewards[max(0, i-10):i+1]) for i in range(len(rewards))]
        plt.plot(means, label="Avg (10)", color='red', alpha=0.8)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.title("Training Loss")
    plt.plot(losses, label="Loss", color='orange', alpha=0.5)
    plt.xlabel("Step / Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    print(f"训练开始！结果将保存在 {result_dir}")
    
    all_rewards = []
    all_avg_losses = [] 
    
    num_episodes = NUM_EPOCHES

    for i_episode in range(num_episodes):
        obs, info = env.reset()
        state = process_state(obs) # 现在这个函数存在了
        total_reward = 0
        episode_losses = []
        
        for t in count():
            # 关键修复 2: 在主循环计算 epsilon，以便打印和使用
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1

            # 将 epsilon 传入 select_action
            action = select_action(state, eps_threshold)
            
            obs, reward, terminated, truncated, _ = env.step(action.item())
            reward = np.clip(reward, -1, 1)
            total_reward += reward
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = process_state(obs)

            reward_tensor = torch.tensor([reward], device=device)
            memory.push(state, action, next_state, reward_tensor)
            state = next_state

            loss_val = optimize_model() 
            if loss_val is not None:
                episode_losses.append(loss_val)

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                all_rewards.append(total_reward)
                avg_loss = np.mean(episode_losses) if episode_losses else 0
                all_avg_losses.append(avg_loss)
                
                # 现在的 eps_threshold 是可见的，不会报错了
                print(f"Episode {i_episode} | Reward: {total_reward} | Avg Loss: {avg_loss:.4f} | Epsilon: {eps_threshold:.3f}")
                break
        
        if i_episode % EVERY_EPOCHES == 0:
            torch.save(policy_net.state_dict(), os.path.join(weight_dir, f"dqn_breakout_ep{i_episode}.pth"))
            plot_and_save(all_rewards, all_avg_losses, os.path.join(curve_dir, "training_curve.png"))
            
            if total_reward >= max(all_rewards):
                 torch.save(policy_net.state_dict(), os.path.join(weight_dir, "best_model_breakout.pth"))
                 print(f"  >>> 新纪录！最佳模型已保存 (Reward: {total_reward})")

    print('训练全部完成！')
    torch.save(policy_net.state_dict(), os.path.join(weight_dir, "final_model_breakout.pth"))
    plot_and_save(all_rewards, all_avg_losses, os.path.join(curve_dir, "final_curve_breakout.png"))
    env.close()