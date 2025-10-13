import torch
import torch.nn as nn
from torch.distributions import Categorical

# [修复] 将 PPOMemory 类移到此处，使其成为 PPO 智能体模块的一部分
class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
    
    def to_dict(self):
        return {
            'states': self.states,
            'actions': self.actions,
            'logprobs': self.logprobs,
            'rewards': self.rewards,
            'dones': self.dones
        }

class ActorCriticRNN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCriticRNN, self).__init__()

        # --- 共享层 ---
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        
        # --- LSTM 层 ---
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # --- Actor (策略) 头 ---
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # --- Critic (价值) 头 ---
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden):
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size * seq_len, -1)
        x = torch.relu(self.fc1(x))
        x = x.view(batch_size, seq_len, -1)

        out, hidden = self.lstm(x, hidden)
        
        lstm_out = out[:, -1, :] # 我们只关心序列中最后一个时间步的输出

        action_logits = self.actor_head(lstm_out)
        dist = Categorical(logits=action_logits)

        value = self.critic_head(lstm_out)

        return dist, value, hidden

class PPORNNAgent:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, K_epochs=4, eps_clip=0.2, hidden_dim=64):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCriticRNN(state_dim, action_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = ActorCriticRNN(state_dim, action_dim, hidden_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def get_initial_hidden_state(self, batch_size=1):
        hidden_dim = self.policy.lstm.hidden_size
        return (torch.zeros(1, batch_size, hidden_dim), torch.zeros(1, batch_size, hidden_dim))

    def select_action(self, state, hidden_state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
            dist, _, new_hidden_state = self.policy_old(state, hidden_state)
            action = dist.sample()
            action_logprob = dist.log_prob(action)

        return action.item(), action_logprob, new_hidden_state

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory['rewards']), reversed(memory['dones'])):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(memory['states'], dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(memory['actions'], dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory['logprobs'], dim=0)).detach()
        
        for _ in range(self.K_epochs):
            initial_hidden = self.get_initial_hidden_state()
            dist, values, _ = self.policy(old_states.unsqueeze(0), initial_hidden)
            advantages = rewards - values.detach().squeeze()
            logprobs = dist.log_prob(old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(values.squeeze(), rewards) - 0.01 * dist.entropy()

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
