# sac_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
import collections


class ReplayBuffer:
    """
    SAC 使用 Off-Policy 的经验回放缓冲区，与 DQN 类似。
    """

    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)

    def add(self, state_history, action, reward, next_state_history, done):
        self.buffer.append((state_history, action, reward, next_state_history, done))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)

        state_histories, actions, rewards, next_state_histories, dones = zip(*batch)

        return (
            torch.tensor(np.array(state_histories), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_state_histories), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)


class ActorRNN(nn.Module):
    """
    SAC 的 Actor 网络 (策略网络)。
    输出动作的概率分布。
    """

    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorRNN, self).__init__()
        self.rnn = nn.GRU(state_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state_sequence):
        rnn_out, _ = self.rnn(state_sequence)
        last_step_out = rnn_out[:, -1, :]
        x = F.relu(self.fc1(last_step_out))
        action_logits = self.fc2(x)
        return action_logits


class CriticRNN(nn.Module):
    """
    SAC 的 Critic 网络 (Q值网络)。
    输入是状态序列，输出是每个动作的 Q 值。
    """

    def __init__(self, state_dim, action_dim, hidden_dim):
        super(CriticRNN, self).__init__()
        self.rnn = nn.GRU(state_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state_sequence):
        rnn_out, _ = self.rnn(state_sequence)
        last_step_out = rnn_out[:, -1, :]
        x = F.relu(self.fc1(last_step_out))
        q_values = self.fc2(x)
        return q_values


class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, sequence_length,
                 lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
                 gamma=0.99, tau=0.005, buffer_size=100000, batch_size=256):

        self.gamma = gamma
        self.tau = tau  # 软更新系数
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"SAC Agent 使用设备: {self.device}")

        # Actor
        self.actor = ActorRNN(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic (包含两个 Q 网络和两个目标 Q 网络)
        self.critic1 = CriticRNN(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2 = CriticRNN(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr_critic)

        self.target_critic1 = CriticRNN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic2 = CriticRNN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # 熵温度 alpha 的自动调整
        # 我们希望熵至少为 target_entropy，如果实际熵小于它，就增加 alpha 来鼓励探索
        self.target_entropy = -torch.tensor(action_dim, dtype=torch.float32).item()  # 启发式设置
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        self.replay_buffer = ReplayBuffer(buffer_size)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state_history):
        with torch.no_grad():
            state_history_tensor = torch.tensor(state_history, dtype=torch.float32).unsqueeze(0).to(self.device)
            action_logits = self.actor(state_history_tensor)
            action_dist = Categorical(logits=action_logits)
            action = action_dist.sample()
        return action.item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        if transitions is None:
            return

        states, actions, rewards, next_states, dones = transitions

        states = states.to(self.device)
        actions = actions.to(self.device).unsqueeze(1)
        rewards = rewards.to(self.device).unsqueeze(1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).unsqueeze(1)

        # --- 更新 Critic ---
        with torch.no_grad():
            # 1. 从下一个状态计算动作概率和对数概率
            next_action_logits = self.actor(next_states)
            next_action_probs = F.softmax(next_action_logits, dim=1)
            next_log_action_probs = F.log_softmax(next_action_logits, dim=1)

            # 2. 计算目标 Q 值
            q1_target_next = self.target_critic1(next_states)
            q2_target_next = self.target_critic2(next_states)
            q_target_next = torch.min(q1_target_next, q2_target_next)

            # 3. 计算 V_target = E[Q_target - alpha * log_prob]
            # 对于离散动作，这是对所有动作的期望
            v_target_next = torch.sum(next_action_probs * (q_target_next - self.alpha * next_log_action_probs), dim=1,
                                      keepdim=True)

            # 4. 计算最终的 Q 目标
            q_target = rewards + self.gamma * (1 - dones) * v_target_next

        # 5. 计算当前 Q 值和 Critic 损失
        q1_current = self.critic1(states).gather(1, actions)
        q2_current = self.critic2(states).gather(1, actions)

        critic1_loss = F.mse_loss(q1_current, q_target)
        critic2_loss = F.mse_loss(q2_current, q_target)

        # 6. 更新 Critic 网络
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # --- 更新 Actor 和 Alpha ---
        # 为了减少计算，冻结 Critic 的梯度
        for p in self.critic1.parameters(): p.requires_grad = False
        for p in self.critic2.parameters(): p.requires_grad = False

        # 1. 计算新策略下的动作概率和 Q 值
        action_logits = self.actor(states)
        action_probs = F.softmax(action_logits, dim=1)
        log_action_probs = F.log_softmax(action_logits, dim=1)

        q1_new = self.critic1(states)
        q2_new = self.critic2(states)
        q_new = torch.min(q1_new, q2_new)

        # 2. 计算 Actor 损失
        actor_loss = torch.sum(action_probs * (self.alpha.detach() * log_action_probs - q_new), dim=1).mean()

        # 3. 更新 Actor 网络
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 恢复 Critic 的梯度
        for p in self.critic1.parameters(): p.requires_grad = True
        for p in self.critic2.parameters(): p.requires_grad = True

        # 4. 更新 Alpha (熵温度)
        # 我们希望最大化 E[-log_prob]，所以 alpha_loss 是 E[log_prob]
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