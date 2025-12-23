import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ==========================================
# 1. 超参数设置 (就像 DL 里的 Learning Rate, Batch Size)
# ==========================================
BATCH_SIZE = 128
GAMMA = 0.99           # 折扣因子 (关注未来)
EPS_START = 0.9        # 初始探索率 (90% 概率乱走)
EPS_END = 0.05         # 最终探索率 (5% 概率乱走)
EPS_DECAY = 1000       # 探索率衰减速度
TAU = 0.005            # 软更新系数 (Target Net 慢慢追 Policy Net)
LR = 1e-4              # 学习率

    # 1. 采样 (相
# 设定设备 (你有 GPU 就用 cuda/mps)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 定义经验回放池 (Replay Buffer)
# ==========================================
# 作用：打破数据相关性，让数据变成立在这个 Buffer 里的 IID 数据
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """保存由 Transition 定义的转换"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """随机采样 Batch"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# ==========================================
# 3. 定义 Q-Network (模型结构)
# ==========================================
# 因为输入是 4维向量，不是图片，所以我们用 MLP (Linear) 而不是 Conv2d
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # 前向传播：输入 State -> 输出所有动作的 Q 值
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# ==========================================
# 4. 准备环境和模型
# ==========================================
env = gym.make("CartPole-v1")

# 获取状态数 (4) 和 动作数 (2: 左, 右)
n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

# 初始化两个网络：Policy Net 和 Target Net
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict()) # 初始时刻参数同步

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

# ==========================================
# 5. 核心工具函数：选择动作 (Epsilon-Greedy)
# ==========================================
def select_action(state):
    global steps_done
    sample = random.random()
    # 计算当前的 epsilon 阈值 (随时间衰减)
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    # 探索 (Exploration) vs 利用 (Exploitation)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) 返回最大值及其索引，[1] 取索引即动作 ID
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # 随机动作
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

# ==========================================
# 6. 核心工具函数：优化模型 (训练一步)
# ==========================================
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    # 1. 采样 (相当于 DataLoader)
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # 这里的操作是为了把 [(s1, a1...), (s2, a2...)] 变成 [s1, s2...], [a1, a2...]
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # 2. 计算预测值 Q(s, a)
    # gather(1, action_batch) 意思是：只提取我实际上执行的那个动作的 Q 值
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 3. 计算目标值 V(s_{t+1}) = max Q(s_{t+1}, a')
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        # 注意：这里用的是 target_net 来计算目标！(稳定目标)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    
    # 贝尔曼最优方程: expected_Q = R + gamma * max Q_target
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # 4. 计算 Loss (Huber Loss / Smooth L1)
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # 5. 反向传播
    optimizer.zero_grad()
    loss.backward()
    # 梯度裁剪 (防止梯度爆炸)
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# ==========================================
# 7. 主训练循环
# ==========================================
num_episodes = 600 # 训练的总回合数

print("开始训练...")
episode_durations = []

for i_episode in range(num_episodes):
    # 重置环境
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    for t in count():
        # 1. 选动作
        action = select_action(state)
        
        # 2. 执行动作，观察环境
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # 3. 存入经验回放池
        memory.push(state, action, next_state, reward)

        # 4. 状态更新
        state = next_state

        # 5. 训练模型 (Experience Replay)
        optimize_model()

        # 6. 软更新 Target Net 参数
        # θ_target = τ * θ_policy + (1 - τ) * θ_target
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            if i_episode % 50 == 0:
                print(f"Episode {i_episode}, 坚持了 {t+1} 步")
            break

print('训练完成！')
env.close()

# 简单的绘图
plt.plot(episode_durations)
plt.title('Training Result')
plt.xlabel('Episode')
plt.ylabel('Duration (Reward)')
plt.show()