import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from collections import deque

# 复用 DRQN 的序列回放缓冲区
class SequenceReplayBuffer:
    def __init__(self, capacity, sequence_length=8):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.sequence_length = sequence_length

    def store_episode(self, episode):
        if len(episode) >= self.sequence_length:
            self.buffer.append(episode)

    def sample(self, batch_size):
        sampled_episodes = random.sample(self.buffer, batch_size)
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = [], [], [], [], []

        for episode in sampled_episodes:
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

# --- SAC-RNN 网络定义 ---
class ActorRNN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorRNN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, hidden):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size * seq_len, -1)
        x = F.relu(self.fc1(x))
        x = x.view(batch_size, seq_len, -1)
        lstm_out, hidden = self.lstm(x, hidden)
        logits = self.fc2(lstm_out)
        dist = Categorical(logits=logits)
        return dist, hidden

class CriticRNN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(CriticRNN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, hidden):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size * seq_len, -1)
        x = F.relu(self.fc1(x))
        x = x.view(batch_size, seq_len, -1)
        lstm_out, hidden = self.lstm(x, hidden)
        q_values = self.fc2(lstm_out)
        return q_values, hidden

# --- SAC-RNN 智能体 ---
class SACRNNAgent:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, buffer_size=10000, batch_size=32, tau=0.005, hidden_dim=64, sequence_length=8, alpha=0.2):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim

        # Actor
        self.actor = ActorRNN(state_dim, action_dim, hidden_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Critics (Double Q-learning)
        self.critic1 = CriticRNN(state_dim, action_dim, hidden_dim)
        self.critic1_target = CriticRNN(state_dim, action_dim, hidden_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)

        self.critic2 = CriticRNN(state_dim, action_dim, hidden_dim)
        self.critic2_target = CriticRNN(state_dim, action_dim, hidden_dim)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)

        # Entropy Temperature (alpha)
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -torch.tensor(action_dim, dtype=torch.float32).item()

        self.memory = SequenceReplayBuffer(buffer_size, sequence_length)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_initial_hidden_state(self, batch_size=1):
        return (torch.zeros(1, batch_size, self.hidden_dim), torch.zeros(1, batch_size, self.hidden_dim))

    def select_action(self, state, hidden_state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
            dist, new_hidden_state = self.actor(state, hidden_state)
            action = dist.sample()
        return action.item(), new_hidden_state

    def store_episode(self, episode):
        self.memory.store_episode(episode)

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        h0 = self.get_initial_hidden_state(self.batch_size)

        # --- Critic Loss ---
        with torch.no_grad():
            next_dist, _ = self.actor(next_states, h0)
            next_action_probs = next_dist.probs
            next_log_action_probs = next_dist.logits # Use logits for stability

            q1_target, _ = self.critic1_target(next_states, h0)
            q2_target, _ = self.critic2_target(next_states, h0)
            min_q_target = torch.min(q1_target, q2_target)
            
            # V_target(s') = E_{a'~pi}[Q_target(s', a') - alpha * log pi(a'|s')]
            v_target = (next_action_probs * (min_q_target - self.alpha * next_log_action_probs)).sum(dim=2)
            
            # Q_target(s, a) = r + gamma * V_target(s')
            q_target = rewards + self.gamma * (1 - dones) * v_target

        q1, _ = self.critic1(states, h0)
        q1 = q1.gather(2, actions.unsqueeze(2)).squeeze(2)
        critic1_loss = F.mse_loss(q1, q_target)

        q2, _ = self.critic2(states, h0)
        q2 = q2.gather(2, actions.unsqueeze(2)).squeeze(2)
        critic2_loss = F.mse_loss(q2, q_target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # --- Actor Loss ---
        dist, _ = self.actor(states, h0)
        action_probs = dist.probs
        log_action_probs = dist.logits

        with torch.no_grad():
            q1_val, _ = self.critic1(states, h0)
            q2_val, _ = self.critic2(states, h0)
            min_q = torch.min(q1_val, q2_val)

        # J_pi(phi) = E_{s~D, a~pi}[alpha * log pi(a|s) - Q(s,a)]
        actor_loss = (action_probs * (self.alpha * log_action_probs - min_q)).sum(dim=2).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Alpha (Entropy Temperature) Loss ---
        alpha_loss = -(self.log_alpha * (log_action_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --- Soft Update Target Networks ---
        self._soft_update(self.critic1_target, self.critic1)
        self._soft_update(self.critic2_target, self.critic2)

    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
