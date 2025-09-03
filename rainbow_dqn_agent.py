import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from collections import deque, namedtuple

# --- 新增：用于优先经验回放的 SumTree ---
class SumTree:
    """
    SumTree 数据结构，用于高效地根据优先级进行采样。
    """
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

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
        # 如果是训练模式，使用带噪声的权重和偏置
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else: # 如果是评估模式，使用确定的权重和偏置
            weight = self.weight_mu
            bias = self.bias_mu
        
        return nn.functional.linear(x, weight, bias)

# --- 2. Dueling Network 组件 ---
class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network 结构。
    它将 Q 值分解为 V(s) (状态价值) 和 A(s, a) (动作优势)。
    """
    def __init__(self, state_dim, action_dim, noisy_std, hidden_dim=128):
        super(DuelingQNetwork, self).__init__()
        
        # 使用 NoisyLinear 替代 nn.Linear
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        self.advantage_layer = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim, std_init=noisy_std),
            nn.ReLU(),
            NoisyLinear(hidden_dim, action_dim, std_init=noisy_std)
        )

        self.value_layer = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim, std_init=noisy_std),
            nn.ReLU(),
            NoisyLinear(hidden_dim, 1, std_init=noisy_std)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_layer(features)
        advantage = self.advantage_layer(features)
        # 组合 V(s) 和 A(s, a)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
    
    def reset_noise(self):
        """重置网络中所有 NoisyLinear 层的噪声"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

# 经验回放缓冲区 (与 DQN 相同)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# --- 修改：使用优先经验回放 ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.e = 0.01  # 小常数，避免优先级为0
        self.max_priority = 1.0

    def push(self, state, action, next_state, reward, done):
        data = Transition(state, action, next_state, reward, done)
        self.tree.add(self.max_priority, data)

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weight /= is_weight.max()

        return batch, idxs, torch.FloatTensor(is_weight)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            p = (priority + self.e) ** self.alpha
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return self.tree.n_entries

# --- 3. Rainbow DQN 智能体 ---
class RainbowDQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, buffer_size, batch_size, tau, noisy_std=0.1, hidden_dim=128, per_alpha=0.6):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        # 使用 DuelingQNetwork
        self.policy_net = DuelingQNetwork(state_dim, action_dim, noisy_std, hidden_dim)
        self.target_net = DuelingQNetwork(state_dim, action_dim, noisy_std, hidden_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = PrioritizedReplayBuffer(buffer_size, alpha=per_alpha)
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
        self.memory.push(state, action, next_state, reward, done) # PER push

    def update(self, beta):
        if len(self.memory) < self.batch_size:
            return

        # 在每次更新开始时，为网络重置一次噪声
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        transitions, batch_indices, is_weights = self.memory.sample(self.batch_size, beta)
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

        # 计算 TD-error 用于更新优先级
        td_errors = (state_action_values - expected_state_action_values.unsqueeze(1)).abs().detach()

        # 更新 PER 优先级
        self.memory.update_priorities(batch_indices, td_errors.squeeze(1).cpu().numpy())

        # 计算加权的损失
        loss = (self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1)) * is_weights.unsqueeze(1)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 软更新目标网络
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)