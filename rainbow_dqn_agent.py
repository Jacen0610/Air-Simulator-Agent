import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from collections import deque, namedtuple

# --- 1. Noisy Nets 组件 ---
# 自定义的带噪声的线性层，用于替代 nn.Linear
class NoisyLinear(nn.Module):
    """
    带参数化噪声的线性层，用于实现 Noisy Nets 探索。
    它取代了 Epsilon-Greedy 策略。
    """
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # 可学习的权重和偏置
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x):
        if self.training:
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return nn.functional.linear(x, weight, bias)

# --- 2. Dueling Network 组件 ---
class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network 结构。
    它将 Q 值分解为 V(s) (状态价值) 和 A(s, a) (动作优势)。
    """
    def __init__(self, state_dim, action_dim):
        super(DuelingQNetwork, self).__init__()
        
        # 使用 NoisyLinear 替代 nn.Linear
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )

        self.advantage_layer = nn.Sequential(
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, action_dim)
        )

        self.value_layer = nn.Sequential(
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, 1)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_layer(features)
        advantage = self.advantage_layer(features)
        # 组合 V(s) 和 A(s, a)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# 经验回放缓冲区 (与 DQN 相同)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- 3. Rainbow DQN 智能体 ---
class RainbowDQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, buffer_size, batch_size, tau):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        # 使用 DuelingQNetwork
        self.policy_net = DuelingQNetwork(state_dim, action_dim)
        self.target_net = DuelingQNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state):
        # 使用 Noisy Nets，不再需要 epsilon-greedy
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state)
            return q_values.max(1)[1].item()

    def store_transition(self, state, action, next_state, reward, done):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.tensor([action], dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.float32)
        self.memory.push(state, action, next_state, reward, done)

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.stack(batch.next_state)
        done_batch = torch.cat(batch.done)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # --- Double DQN 核心 ---
        # 1. 使用 policy_net 选择下一最佳动作
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            # 2. 使用 target_net 评估该动作的 Q 值
            next_state_q_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
        
        expected_state_action_values = (next_state_q_values * self.gamma * (1 - done_batch)) + reward_batch

        loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 软更新目标网络
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)