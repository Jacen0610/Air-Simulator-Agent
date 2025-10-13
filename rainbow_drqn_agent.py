import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# --- Noisy Layer (from original Rainbow) ---
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = torch.randn(self.in_features)
        epsilon_out = torch.randn(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)

# --- DRQN Network ---
class DuelingRecurrentQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, noisy_std=0.1):
        super(DuelingRecurrentQNetwork, self).__init__()
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Dueling Architecture
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim, noisy_std),
            nn.ReLU(),
            NoisyLinear(hidden_dim, action_dim, noisy_std)
        )

        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim, noisy_std),
            nn.ReLU(),
            NoisyLinear(hidden_dim, 1, noisy_std)
        )

    def forward(self, x, hidden):
        # x shape: (batch, seq_len, state_dim)
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size * seq_len, -1)
        x = F.relu(self.fc1(x))
        x = x.view(batch_size, seq_len, -1)

        lstm_out, hidden = self.lstm(x, hidden)
        
        # Dueling streams
        advantages = self.advantage_stream(lstm_out) # (batch, seq_len, action_dim)
        values = self.value_stream(lstm_out)      # (batch, seq_len, 1)

        # Combine to get Q-values
        qvals = values + (advantages - advantages.mean(dim=2, keepdim=True))
        return qvals, hidden

    def reset_noise(self):
        for name, module in self.named_children():
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, NoisyLinear):
                        layer.reset_noise()

# --- Sequence Replay Buffer ---
class SequenceReplayBuffer:
    def __init__(self, capacity, sequence_length=8):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.sequence_length = sequence_length

    def store_episode(self, episode):
        # episode is a list of (state, action, reward, done)
        if len(episode) >= self.sequence_length:
            self.buffer.append(episode)

    def sample(self, batch_size):
        # Sample full episodes
        sampled_episodes = random.sample(self.buffer, batch_size)

        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = [], [], [], [], []

        for episode in sampled_episodes:
            # Sample a random start index for the sequence
            start_idx = random.randint(0, len(episode) - self.sequence_length)
            sequence = episode[start_idx : start_idx + self.sequence_length]
            next_sequence = episode[start_idx + 1 : start_idx + self.sequence_length + 1]

            states, actions, rewards, dones = zip(*sequence)
            next_states, _, _, _ = zip(*next_sequence)

            batch_states.append(torch.FloatTensor(np.array(states)))
            batch_actions.append(torch.LongTensor(actions))
            batch_rewards.append(torch.FloatTensor(rewards))
            batch_next_states.append(torch.FloatTensor(np.array(next_states)))
            batch_dones.append(torch.FloatTensor(dones))

        return (
            torch.stack(batch_states),
            torch.stack(batch_actions),
            torch.stack(batch_rewards),
            torch.stack(batch_next_states),
            torch.stack(batch_dones)
        )

    def __len__(self):
        return len(self.buffer)

# --- Rainbow DRQN Agent ---
class RainbowDRQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.0001, gamma=0.99, buffer_size=10000, batch_size=32, tau=0.005, hidden_dim=64, noisy_std=0.1, sequence_length=8):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        self.policy_net = DuelingRecurrentQNetwork(state_dim, action_dim, hidden_dim, noisy_std)
        self.target_net = DuelingRecurrentQNetwork(state_dim, action_dim, hidden_dim, noisy_std)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = SequenceReplayBuffer(buffer_size, sequence_length)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def get_initial_hidden_state(self, batch_size=1):
        hidden_dim = self.policy_net.hidden_dim
        return (torch.zeros(1, batch_size, hidden_dim), torch.zeros(1, batch_size, hidden_dim))

    def select_action(self, state, hidden_state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0) # (1, 1, state_dim)
            q_values, new_hidden_state = self.policy_net(state, hidden_state)
            action = q_values.squeeze(0).argmax().item()
        return action, new_hidden_state

    def store_episode(self, episode):
        self.memory.store_episode(episode)

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Initial hidden states for the batch of sequences
        h0 = self.get_initial_hidden_state(self.batch_size)
        h1 = self.get_initial_hidden_state(self.batch_size)

        # Get Q-values for current states: Q(s_t, a_t)
        # q_values shape: (batch, seq_len, action_dim)
        q_values, _ = self.policy_net(states, h0)
        # Gather Q-values for the actions that were taken
        # actions shape: (batch, seq_len) -> (batch, seq_len, 1)
        q_values = q_values.gather(2, actions.unsqueeze(2)).squeeze(2)

        # Get Q-values for next states: max_a Q_target(s_{t+1}, a)
        with torch.no_grad():
            next_q_values, _ = self.target_net(next_states, h1)
            next_q_values = next_q_values.max(2)[0]

        # Compute the target Q-value
        # dones shape is (batch, seq_len)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss (Huber loss)
        loss = F.smooth_l1_loss(q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update the target network
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
