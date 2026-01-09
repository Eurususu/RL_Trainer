import numpy as np
import pandas as pd
import time

# --- 1. 定义迷宫环境 ---
class MazeEnv:
    def __init__(self):
        # 3x3 网格
        # 0:路, 1:陷阱(Trap), 2:终点(Goal)
        self.map = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 2]
        ]
        self.n_rows = 3
        self.n_cols = 3
        self.start_pos = (0, 0)
        self.current_pos = self.start_pos

    def reset(self):
        self.current_pos = self.start_pos
        return self.current_pos

    def step(self, action):
        """
        Action: 0=上, 1=下, 2=左, 3=右
        """
        x, y = self.current_pos
        
        # 根据动作计算移动
        if action == 0:   # Up
            x = max(0, x - 1)
        elif action == 1: # Down
            x = min(self.n_rows - 1, x + 1)
        elif action == 2: # Left
            y = max(0, y - 1)
        elif action == 3: # Right
            y = min(self.n_cols - 1, y + 1)
            
        next_state = (x, y)
        self.current_pos = next_state
        
        # 获取奖励和判断是否结束
        cell_type = self.map[x][y]
        
        if cell_type == 1: # 掉进陷阱
            reward = -10
            done = True
        elif cell_type == 2: # 到达终点
            reward = 10
            done = True
        else: # 普通路，给予微小负奖励鼓励走最短路
            reward = -0.1
            done = False
            
        return next_state, reward, done

# --- 2. Q-Learning 算法主体 ---
def q_learning_maze(episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    env = MazeEnv()
    
    # 初始化 Q-Table: 3x3个状态，每个状态有4个动作
    # 维度: (3, 3, 4)
    q_table = np.zeros((env.n_rows, env.n_cols, 4))
    
    actions = [0, 1, 2, 3] # 上下左右

    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            sx, sy = state
            
            # --- 核心 1: Epsilon-Greedy 策略选择动作 ---
            if np.random.uniform(0, 1) < epsilon:
                # 探索模式：随机选动作
                action = np.random.choice(actions)
            else:
                # 利用模式：选当前Q值最大的动作
                # 这里的逻辑稍微复杂一点是为了处理初始全为0的情况，随机选一个
                current_q_values = q_table[sx, sy, :]
                max_q = np.max(current_q_values)
                # 如果有多个最大值，随机选一个（避免死板）
                action = np.random.choice(np.where(current_q_values == max_q)[0])

            # 执行动作
            next_state, reward, done = env.step(action)
            nx, ny = next_state
            
            # --- 核心 2: Q-Learning 更新公式 ---
            # Q(S, A) <- Q(S, A) + alpha * [R + gamma * max(Q(S', a')) - Q(S, A)]
            
            # 预测值 (当前Q)
            q_predict = q_table[sx, sy, action]
            
            # 目标值 (TD Target)
            if done:
                # 如果结束了，就没有下一步的价值了
                q_target = reward
            else:
                # 即使下一步会乱走，Q-Learning也假设你会走最好的一步 (max)
                q_target = reward + gamma * np.max(q_table[nx, ny, :])
            
            # 更新 Q 表
            q_table[sx, sy, action] += alpha * (q_target - q_predict)
            
            # 更新状态
            state = next_state

    return q_table

# --- 3. 结果展示 ---
def print_policy(q_table):
    action_icons = ['↑', '↓', '←', '→']
    policy_map = []
    
    for i in range(3):
        row_str = []
        for j in range(3):
            if (i, j) == (1, 1):
                row_str.append(" X ") # 陷阱
                continue
            if (i, j) == (2, 2):
                row_str.append(" G ") # 终点
                continue
                
            # 找到该状态下价值最大的动作
            best_action_idx = np.argmax(q_table[i, j, :])
            row_str.append(f" {action_icons[best_action_idx]} ")
        policy_map.append(row_str)
        
    print("\n学习到的最优策略 (地图):")
    for row in policy_map:
        print("".join(row))

if __name__ == "__main__":
    print("开始训练 Q-Learning Agent...")
    final_q_table = q_learning_maze(episodes=1000)
    
    print("\n训练完成!")
    # 打印 (0,0) 起点的 Q 值看看
    print("起点 (0,0) 的四个动作价值:", final_q_table[0,0])
    print("起点 (0,1) 的四个动作价值:", final_q_table[0,1])
    print("起点 (0,2) 的四个动作价值:", final_q_table[0,2])
    print("起点 (1,0) 的四个动作价值:", final_q_table[1,0])
    print("起点 (1,1) 的四个动作价值:", final_q_table[1,1])
    print("起点 (1,2) 的四个动作价值:", final_q_table[1,2])
    print("起点 (2,0) 的四个动作价值:", final_q_table[2,0])
    print("起点 (2,1) 的四个动作价值:", final_q_table[2,1])
    print("起点 (2,2) 的四个动作价值:", final_q_table[2,2])
    
    print_policy(final_q_table)