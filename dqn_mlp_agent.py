# dqn_mlp_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import collections


class ReplayBuffer:
    """
    DQN 使用的 Off-Policy 经验回放缓冲区。
    """

    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)


class QNetworkMLP(nn.Module):
    """
    DQN 的 Q-Value 网络，使用 MLP。
    """

    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetworkMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        q_values = self.net(state)
        return q_values


class DQNAgent:
    """
    使用 MLP 网络的 DQN 智能体。
    """

    def __init__(self, state_dim, action_dim, hidden_dim,
                 learning_rate=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update_freq=1000):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"MLP DQN Agent 使用设备: {self.device}")

        self.q_network = QNetworkMLP(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_network = QNetworkMLP(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()  # 目标网络不进行训练

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state):
        """
        根据 epsilon-greedy 策略选择动作。
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        transitions = self.replay_buffer.sample(self.batch_size)
        if transitions is None:
            return None

        states, actions, rewards, next_states, dones = transitions

        states = states.to(self.device)
        actions = actions.to(self.device).unsqueeze(1)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 计算当前 Q 值
        current_q_values = self.q_network(states).gather(1, actions).squeeze(1)

        # 计算下一个状态的最大 Q 值 (使用目标网络)
        with torch.no_grad():
            next_q_values = self.target_q_network(next_states).max(1)[0]
            # 如果 done 为 True，则下一个状态的 Q 值为 0
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # 计算损失
        loss = self.loss_fn(current_q_values, target_q_values)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)  # 梯度裁剪
        self.optimizer.step()

        # 更新 epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # 更新目标网络
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self):
        """
        将 Q 网络权重复制到目标 Q 网络。
        """
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load_model(self, path):
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.target_q_network.load_state_dict(self.q_network.state_dict())