# ppo_transformer_agent.py
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import math


class RolloutBuffer:
    """
    PPO的经验存储区 (on-policy)。
    """

    def __init__(self):
        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.state_values = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.state_values[:]


class PositionalEncoding(nn.Module):
    """
    为序列中的每个位置添加位置信息。
    Transformer本身不处理顺序，所以我们需要手动告诉它每个状态的位置。
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 形状: (sequence_length, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return x


class ActorCriticTransformer(nn.Module):
    """
    [终极创新网络] 基于 Transformer Encoder 的 Actor-Critic 网络。
    """

    def __init__(self, state_dim, action_dim, embed_dim, nhead, num_encoder_layers, dim_feedforward):
        super(ActorCriticTransformer, self).__init__()
        self.embed_dim = embed_dim

        # 1. 输入嵌入层: 将7维的状态向量映射到更高维度的嵌入空间
        self.input_embedding = nn.Linear(state_dim, embed_dim)

        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(embed_dim)

        # 3. Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 4. 决策头
        self.actor_head = nn.Linear(embed_dim, action_dim)
        self.critic_head = nn.Linear(embed_dim, 1)

    def forward(self, state_sequence):
        # state_sequence 形状: (batch_size, sequence_length, state_dim)

        # 嵌入输入
        embedded_input = self.input_embedding(state_sequence)  # -> (batch_size, sequence_length, embed_dim)

        # 添加位置编码
        # Transformer的输入期望形状是 (sequence_length, batch_size, embed_dim)
        embedded_input = embedded_input.permute(1, 0, 2)
        embedded_input_with_pos = self.pos_encoder(embedded_input)

        # 通过 Transformer Encoder
        transformer_output = self.transformer_encoder(
            embedded_input_with_pos)  # -> (sequence_length, batch_size, embed_dim)

        # 切换回 batch_first
        transformer_output = transformer_output.permute(1, 0, 2)  # -> (batch_size, sequence_length, embed_dim)

        # 我们使用序列的第一个位置的输出作为整个序列的聚合表示
        # 这在BERT等模型中是常见做法，第一个token (CLS token) 用来做分类任务
        aggregated_representation = transformer_output[:, 0, :]  # -> (batch_size, embed_dim)

        # 最终决策
        state_value = self.critic_head(aggregated_representation)
        action_logits = self.actor_head(aggregated_representation)
        action_dist = Categorical(logits=action_logits)

        return action_dist, state_value


class PPOTransformerAgent:
    """
    使用 Transformer 网络的 PPO 智能体。
    """

    def __init__(self, state_dim, action_dim, embed_dim, nhead, num_encoder_layers, dim_feedforward,
                 lr_actor_critic, gamma, lambda_gae, eps_clip, k_epochs):

        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Transformer PPO Agent 使用设备: {self.device}")

        self.buffer = RolloutBuffer()

        # 实例化新的 Transformer 网络
        self.policy = ActorCriticTransformer(state_dim, action_dim, embed_dim, nhead, num_encoder_layers,
                                             dim_feedforward).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor_critic)

        self.policy_old = ActorCriticTransformer(state_dim, action_dim, embed_dim, nhead, num_encoder_layers,
                                                 dim_feedforward).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.loss_fn = nn.MSELoss()

    def select_action(self, state_history):
        with torch.no_grad():
            state_history_tensor = torch.tensor(state_history, dtype=torch.float32).unsqueeze(0).to(self.device)
            action_dist, state_value = self.policy_old(state_history_tensor)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
        return action.item(), log_prob.item(), state_value.item()

    def store_transition(self, state, action, log_prob, reward, done, state_value):
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.log_probs.append(log_prob)
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(done)
        self.buffer.state_values.append(state_value)

    def update(self):
        # 这部分的PPO更新逻辑与之前的版本完全相同
        advantages = []
        last_advantage = 0
        for i in reversed(range(len(self.buffer.rewards))):
            reward = self.buffer.rewards[i]
            done = self.buffer.dones[i]
            v_s = self.buffer.state_values[i]
            v_s_next = self.buffer.state_values[i + 1] if i < len(self.buffer.rewards) - 1 else 0
            delta = reward + self.gamma * v_s_next * (1 - done) - v_s
            last_advantage = delta + self.gamma * self.lambda_gae * (1 - done) * last_advantage
            advantages.insert(0, last_advantage)
        returns = (torch.tensor(advantages, dtype=torch.float32) + torch.tensor(self.buffer.state_values,
                                                                                dtype=torch.float32)).detach()
        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        old_states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32).to(self.device)
        old_actions = torch.tensor(self.buffer.actions, dtype=torch.int64).to(self.device)
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32).to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        for _ in range(self.k_epochs):
            action_dist, state_values = self.policy(old_states)
            log_probs = action_dist.log_prob(old_actions)
            dist_entropy = action_dist.entropy()
            ratios = torch.exp(log_probs - old_log_probs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.loss_fn(state_values.squeeze(), returns)
            entropy_bonus = -0.01 * dist_entropy.mean()  # 可以从一个较小的值开始
            loss = actor_loss + 0.5 * critic_loss + entropy_bonus
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save_model(self, path):
        torch.save(self.policy_old.state_dict(), path)

    def load_model(self, path):
        self.policy_old.load_state_dict(torch.load(path, map_location=self.device))
        self.policy.load_state_dict(torch.load(path, map_location=self.device))