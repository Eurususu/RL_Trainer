import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# ==========================================
# 1. 超参数
# ==========================================
learning_rate = 0.0003
gamma = 0.99           # 折扣因子
lmbda = 0.95           # GAE 参数 (用于计算优势)
eps_clip = 0.2         # PPO 截断范围 (0.8 ~ 1.2)
K_epochs = 3           # 每次收集完数据，拿来训练几轮
T_horizon = 20         # 每隔多少步更新一次网络 (或者设长一点)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. PPO 网络结构 (Actor-Critic)
# ==========================================
class PPO(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPO, self).__init__()
        self.data = [] # 临时存储轨迹
        
        # Critic 网络: 输入状态 -> 输出 V 值 (标量)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Actor 网络: 输入状态 -> 输出动作概率 (Softmax)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float).to(device), \
                                          torch.tensor(a_lst).to(device), \
                                          torch.tensor(r_lst).to(device), \
                                          torch.tensor(s_prime_lst, dtype=torch.float).to(device), \
                                          torch.tensor(done_lst, dtype=torch.float).to(device), \
                                          torch.tensor(prob_a_lst).to(device)
        self.data = [] # 清空数据 (On-Policy 特性)
        return s, a, r, s_prime, done_mask, prob_a

    # 训练主函数
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        # 1. 计算 Target Value (TD Target)
        # V(s')
        v_prime = self.critic(s_prime) 
        td_target = r + gamma * v_prime * done_mask
        
        # V(s)
        v_s = self.critic(s)
        
        # 2. 计算优势函数 (Advantage)
        # A = td_target - V(s)
        # 这里用简单的 Advantage，也可以换成 GAE (Generalized Advantage Estimation)
        delta = td_target - v_s
        delta = delta.detach().cpu().numpy()

        advantage_lst = []
        advantage = 0.0
        # GAE 计算 (从后往前算)
        for delta_t in delta[::-1]:
            advantage = gamma * lmbda * advantage + delta_t[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage = torch.tensor(advantage_lst, dtype=torch.float).to(device)

        # 3. PPO 更新循环 (由 K_epochs 决定复用几次数据)
        for _ in range(K_epochs):
            # 重新计算当前的 pi(a|s) 和 V(s)
            pi = self.actor(s)
            cur_v = self.critic(s)
            
            # 获取实际执行动作的概率
            pi_a = pi.gather(1, a) 
            
            # 计算比率 Ratio = pi_new / pi_old
            # prob_a 是我们在收集数据时存下来的旧概率
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            # 4. PPO Loss 公式 (Clipping)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(cur_v, td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

# ==========================================
# 3. 主循环
# ==========================================
def main():
    env = gym.make('CartPole-v1')
    model = PPO(env.observation_space.shape[0], env.action_space.n).to(device)
    score = 0.0
    print_interval = 20

    for n_epi in range(1000):
        s, _ = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                # 1. Action 选择 (概率采样)
                prob = model.actor(torch.from_numpy(s).float().to(device))
                m = Categorical(prob)
                a = m.sample().item()
                
                s_prime, r, terminated, truncated, info = env.step(a)
                done = terminated or truncated

                # 2. 收集数据 (注意：要存下 log_prob 或者 prob，用于算 ratio)
                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                
                s = s_prime
                score += r
                if done:
                    break
            
            # 3. 只要收集够了数据，立马更新 (On-Policy)
            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print(f"# of episode :{n_epi}, avg score : {score/print_interval:.1f}")
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()