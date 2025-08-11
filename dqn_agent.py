import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple

# 定义 Q 网络结构
class QNetwork(nn.Module):
    """一个简单的全连接网络，用于逼近 Q 函数"""
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# 定义经验回放缓冲区
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    """一个固定大小的缓冲区，用于存储和采样经验"""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """保存一个 transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """从内存中随机采样一个 batch 的 transitions"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# 定义 DQN 智能体
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, buffer_size, batch_size, tau):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau  # 用于目标网络的软更新

        # 创建主网络和目标网络
        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络不参与训练

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.loss_fn = nn.SmoothL1Loss()  # Huber Loss，比 MSE 更稳定

    def select_action(self, state, epsilon):
        """使用 Epsilon-Greedy 策略选择动作"""
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state)
                # 选择具有最高 Q 值的动作
                return q_values.max(1)[1].item()
        else:
            # 随机选择一个动作进行探索
            return random.randint(0, self.action_dim - 1)

    def store_transition(self, state, action, next_state, reward, done):
        """将一个 transition 存入回放缓冲区"""
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.tensor([action], dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.float32)
        self.memory.push(state, action, next_state, reward, done)

    def update(self):
        """从回放缓冲区采样数据并更新网络"""
        if len(self.memory) < self.batch_size:
            return  # 如果数据不够一个 batch，则不学习

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.stack(batch.next_state)
        done_batch = torch.cat(batch.done)

        # 计算当前状态的 Q 值: Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # 使用目标网络计算下一状态的最大 Q 值: max_a Q(s_{t+1}, a)
        with torch.no_grad():
            next_state_q_values = self.target_net(next_state_batch).max(1)[0]
        
        # 计算期望的 Q 值 (TD Target)
        expected_state_action_values = (next_state_q_values * self.gamma * (1 - done_batch)) + reward_batch

        # 计算损失
        loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 软更新目标网络
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)