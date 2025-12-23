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
        name_prefix="dqn-pong",
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
NUM_EPOCHES = 601
EVERY_EPOCHES = 50


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

# ==========================================
# 初始化
# ==========================================
env_name = "PongNoFrameskip-v4"
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

policy_net = CNN_DQN(n_actions).to(device)
target_net = CNN_DQN(n_actions).to(device)
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
    state_batch = torch.cat(batch.state) # 形状: [Batch_Size, Channel, Height, Width]
    action_batch = torch.cat(batch.action) # 形状: [Batch_Size, 1]
    reward_batch = torch.cat(batch.reward) # 形状: [Batch_Size]
    # ====================================================
    # 计算当前状态的 Q 值 (预测值)
    # ====================================================
    # policy_net(state_batch) 输出形状: [Batch_Size, n_actions] (每个动作的 Q 值)
    # .gather(1, action_batch) 作用: 从所有动作的 Q 值中，只提取出我们实际执行了的那个动作的 Q 值
    # 结果 state_action_values 形状: [Batch_Size, 1]
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    # ====================================================
    # 计算下一状态的 V 值 (目标值的一部分)
    # ====================================================
    # 初始化全 0 张量，用于存放下一状态的价值 V(s_{t+1})
    # 对于终止状态 (Done)，其价值保持为 0
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        # .max(1)[0] 作用: 找到每个状态下 Q 值最大的那个动作对应的 Q 值 (贪婪策略) 用最大的Q来代替加权平均的V(next_state)
        # max_a' Q*(s', a')
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # 贝尔曼方程: Q_target = Reward + Gamma * V(next_state) 这里的V就是上面的贪婪策略下的 max_a' Q*(s', a')
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
        
        if i_episode % 50 == 0:
            torch.save(policy_net.state_dict(), os.path.join(weight_dir, f"dqn_pong_ep{i_episode}.pth"))
            plot_and_save(all_rewards, all_avg_losses, os.path.join(curve_dir, "training_curve.png"))
            
            if total_reward >= max(all_rewards):
                 torch.save(policy_net.state_dict(), os.path.join(weight_dir, "best_model.pth"))
                 print(f"  >>> 新纪录！最佳模型已保存 (Reward: {total_reward})")

    print('训练全部完成！')
    torch.save(policy_net.state_dict(), os.path.join(weight_dir, "final_model.pth"))
    plot_and_save(all_rewards, all_avg_losses, os.path.join(curve_dir, "final_curve.png"))
    env.close()