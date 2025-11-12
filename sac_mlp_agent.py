# train_mlp_sac.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
import collections
import os
import matplotlib.pyplot as plt
from go_simulator_env import GoSimulatorEnv


# ==============================================================================
#  1. Replay Buffer
# ==============================================================================
class ReplayBuffer:
    """
    SAC 使用的 Off-Policy 经验回放缓冲区。
    """

    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        # 注意：这里存储的是单个状态，而不是状态历史
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


# ==============================================================================
#  2. Neural Network Architectures
# ==============================================================================
class ActorMLP(nn.Module):
    """
    SAC 的 Actor 网络 (策略网络)，使用 MLP。
    """

    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        action_logits = self.net(state)
        return action_logits


class CriticMLP(nn.Module):
    """
    SAC 的 Critic 网络 (Q值网络)，使用 MLP。
    """

    def __init__(self, state_dim, action_dim, hidden_dim):
        super(CriticMLP, self).__init__()
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


# ==============================================================================
#  3. SAC Agent
# ==============================================================================
class SACMLPAgent:
    """
    使用 MLP 网络的 SAC 智能体。
    """

    def __init__(self, state_dim, action_dim, hidden_dim,
                 lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
                 gamma=0.99, tau=0.005, buffer_size=100000, batch_size=256):

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"MLP SAC Agent 使用设备: {self.device}")

        # Actor
        self.actor = ActorMLP(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic
        self.critic1 = CriticMLP(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2 = CriticMLP(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr_critic)

        self.target_critic1 = CriticMLP(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic2 = CriticMLP(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # 熵温度 alpha 的自动调整
        self.target_entropy = -torch.tensor(action_dim, dtype=torch.float32).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        self.replay_buffer = ReplayBuffer(buffer_size)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            action_logits = self.actor(state_tensor)
            action_dist = Categorical(logits=action_logits)
            action = action_dist.sample()
        return action.item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device).unsqueeze(1)
        rewards = rewards.to(self.device).unsqueeze(1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).unsqueeze(1)

        # --- 更新 Critic ---
        with torch.no_grad():
            next_action_logits = self.actor(next_states)
            next_action_probs = F.softmax(next_action_logits, dim=1)
            next_log_action_probs = F.log_softmax(next_action_logits, dim=1)

            q1_target_next = self.target_critic1(next_states)
            q2_target_next = self.target_critic2(next_states)
            q_target_next = torch.min(q1_target_next, q2_target_next)

            v_target_next = torch.sum(next_action_probs * (q_target_next - self.alpha * next_log_action_probs), dim=1,
                                      keepdim=True)
            q_target = rewards + self.gamma * (1 - dones) * v_target_next

        q1_current = self.critic1(states).gather(1, actions)
        q2_current = self.critic2(states).gather(1, actions)

        critic1_loss = F.mse_loss(q1_current, q_target)
        critic2_loss = F.mse_loss(q2_current, q_target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # --- 更新 Actor 和 Alpha ---
        for p in self.critic1.parameters(): p.requires_grad = False
        for p in self.critic2.parameters(): p.requires_grad = False

        action_logits = self.actor(states)
        action_probs = F.softmax(action_logits, dim=1)
        log_action_probs = F.log_softmax(action_logits, dim=1)

        q1_new = self.critic1(states)
        q2_new = self.critic2(states)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = torch.sum(action_probs * (self.alpha.detach() * log_action_probs - q_new), dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for p in self.critic1.parameters(): p.requires_grad = True
        for p in self.critic2.parameters(): p.requires_grad = True

        alpha_loss = -torch.sum(
            action_probs.detach() * (self.alpha * (log_action_probs.detach() + self.target_entropy)), dim=1).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --- 软更新目标网络 ---
        self.soft_update_target_networks()

    def soft_update_target_networks(self):
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save_model(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
