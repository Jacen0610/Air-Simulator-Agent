# dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import collections


class ReplayBuffer:
    """
    经验回放缓冲区。
    存储 (状态历史, 动作, 奖励, 下一个状态历史, 完成标志) 元组。
    """

    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)

    def add(self, state_history, action, reward, next_state_history, done):
        self.buffer.append((state_history, action, reward, next_state_history, done))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        # 将 batch 中的元组解包成独立的列表，并转换为 numpy 数组
        state_histories, actions, rewards, next_state_histories, dones = zip(*batch)

        # 确保所有历史观测都是 numpy 数组，并且形状正确
        state_histories = np.array(state_histories, dtype=np.float32)
        next_state_histories = np.array(next_state_histories, dtype=np.float32)

        return (
            torch.tensor(state_histories, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_state_histories, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)


class RNNDQN(nn.Module):
    """
    基于 RNN 的 Q 网络。
    输入是历史观测序列，输出是每个动作的 Q 值。
    """

    def __init__(self, state_dim, action_dim, hidden_dim, sequence_length):
        super(RNNDQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length

        # RNN 层 (例如 LSTM 或 GRU)
        # batch_first=True 表示输入形状为 (batch, seq_len, features)
        self.rnn = nn.GRU(input_size=state_dim, hidden_size=hidden_dim, batch_first=True)

        # 线性层，从 RNN 的输出映射到 Q 值
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, action_dim)

    def forward(self, x):
        # x 的形状: (batch_size, sequence_length, state_dim)

        # 经过 RNN 层
        # output 的形状: (batch_size, sequence_length, hidden_dim)
        # hidden 的形状: (num_layers * num_directions, batch_size, hidden_dim)
        output, hidden = self.rnn(x)

        # 我们只关心序列的最后一个时间步的输出，或者 RNN 的最终隐藏状态
        # 对于 GRU/LSTM，hidden[0] 是最后一个时间步的隐藏状态 (如果 num_layers=1)
        # 或者直接使用 output[:, -1, :]

        # 使用最后一个时间步的输出
        q_values = self.fc1(output[:, -1, :])
        q_values = nn.functional.relu(q_values)
        q_values = self.fc2(q_values)
        return q_values


class RNNDQNAgent:
    """
    RNN-DQN 智能体。
    """

    def __init__(self, state_dim, action_dim, hidden_dim, sequence_length,
                 learning_rate=1e-3, gamma=0.99, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update_freq=100):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        self.q_network = RNNDQN(state_dim, action_dim, hidden_dim, sequence_length).to(self.device)
        self.target_q_network = RNNDQN(state_dim, action_dim, hidden_dim, sequence_length).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()  # 目标网络不进行训练

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, observation_history: np.ndarray) -> int:
        """
        根据 epsilon-greedy 策略选择动作。
        observation_history 形状: (sequence_length, state_dim)
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            # 将历史观测转换为 tensor，并添加 batch 维度
            state_history_tensor = torch.tensor(observation_history, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_history_tensor)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state_history, action, reward, next_state_history, done):
        """
        将经验存储到回放缓冲区。
        """
        self.replay_buffer.add(state_history, action, reward, next_state_history, done)

    def learn(self):
        """
        从回放缓冲区采样并更新 Q 网络。
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        if transitions is None:  # 缓冲区不足
            return

        state_histories, actions, rewards, next_state_histories, dones = transitions

        state_histories = state_histories.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_state_histories = next_state_histories.to(self.device)
        dones = dones.to(self.device)

        # 计算当前 Q 值
        # q_network(state_histories) 形状: (batch_size, action_dim)
        # actions.unsqueeze(1) 形状: (batch_size, 1)
        # gather(1, actions.unsqueeze(1)) 提取每个样本中实际执行动作的 Q 值
        current_q_values = self.q_network(state_histories).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算下一个状态的最大 Q 值 (使用目标网络)
        with torch.no_grad():
            next_q_values = self.target_q_network(next_state_histories).max(1)[0]
            # 如果 done 为 True，则下一个状态的 Q 值为 0
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # 计算损失
        loss = self.loss_fn(current_q_values, target_q_values)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
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
        self.target_q_network.eval()

    def save_model(self, path):
        """保存模型权重。"""
        torch.save(self.q_network.state_dict(), path)
        print(f"模型已保存到 {path}")

    def load_model(self, path):
        """加载模型权重。"""
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()
        print(f"模型已从 {path} 加载。")
