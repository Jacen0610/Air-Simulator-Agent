# ppo_attention_mlp_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from torch.utils.data import BatchSampler, SubsetRandomSampler


class RolloutBuffer:
    """
    PPO的经验存储区 (on-policy)。
    收集一个完整的轨迹，更新策略后清空。
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


class ActorCriticAttentionMLP(nn.Module):
    """
    [2024-05-22 优化] 直接在状态历史序列上应用注意力机制，并加入了位置编码。
    """

    def __init__(self, state_dim, action_dim, hidden_dim, sequence_length):
        super(ActorCriticAttentionMLP, self).__init__()
        self.state_dim = state_dim
        self.sequence_length = sequence_length

        # [新] 创建可学习的位置编码，让模型感知时序关系
        # 它的维度与状态序列相同，可以被直接加上去
        self.positional_encoding = nn.Parameter(
            torch.randn(1, sequence_length, state_dim), 
            requires_grad=True
        )

        # 注意力网络: 计算每个历史状态与当前状态的相关性分数
        self.attention_net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Dropout(p=0.2)
        )

        # 决策网络的主体
        self.fc_main = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),  # 输入是注意力加权后的上下文向量
            nn.Tanh()
        )

        # Actor 和 Critic 头
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state_sequence, return_weights=False):
        # state_sequence 形状: (batch_size, sequence_length, state_dim)

        # [核心修改] 将位置信息加入到状态序列中
        state_sequence_with_pos = state_sequence + self.positional_encoding

        # --- 后续所有计算都使用包含了位置信息的新序列 ---

        # 提取当前状态作为 "查询" (Query)
        current_state = state_sequence_with_pos[:, -1, :].unsqueeze(1)  # 形状: (batch_size, 1, state_dim)

        # 将当前状态扩展，以便与历史序列中的每个状态进行拼接
        current_state_expanded = current_state.repeat(1, self.sequence_length, 1)

        # 拼接，准备计算注意力分数
        attention_input = torch.cat([state_sequence_with_pos, current_state_expanded], dim=2)

        # 计算分数
        scores = self.attention_net(attention_input)

        # 转换为权重
        attention_weights = F.softmax(scores, dim=1)

        # 计算上下文向量 (对包含了位置信息的序列进行加权)
        context_vector = torch.sum(attention_weights * state_sequence_with_pos, dim=1)

        # 将上下文向量送入决策网络
        main_features = self.fc_main(context_vector)

        # 最终决策
        state_value = self.critic_head(main_features)
        action_logits = self.actor_head(main_features)
        action_dist = Categorical(logits=action_logits)

        if return_weights:
            return action_dist, state_value, attention_weights
        else:
            return action_dist, state_value


class PPOAttentionMLPAgent:
    """
    使用 Attention-MLP 网络的 PPO 智能体。
    """

    def __init__(self, state_dim, action_dim, hidden_dim, sequence_length,
                 lr_actor_critic, gamma, lambda_gae, eps_clip, k_epochs, batch_size, max_grad_norm): # 新增 max_grad_norm

        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm # 保存 max_grad_norm

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Attention-MLP PPO Agent 使用设备: {self.device}")

        self.buffer = RolloutBuffer()

        # 实例化新的 Attention-MLP 网络
        self.policy = ActorCriticAttentionMLP(state_dim, action_dim, hidden_dim, sequence_length).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor_critic)

        self.policy_old = ActorCriticAttentionMLP(state_dim, action_dim, hidden_dim, sequence_length).to(self.device)
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
        # 1. 计算优势 (Advantage) 和回报 (Return) - 使用 GAE
        advantages = []
        
        # 获取所有数据长度
        data_len = len(self.buffer.rewards)
        
        # 确保 buffer.state_values 长度正确
        if len(self.buffer.state_values) != data_len:
            raise ValueError("Buffer state_values length mismatch with rewards length.")

        # 计算 GAE 优势
        for i in reversed(range(data_len)):
            reward = self.buffer.rewards[i]
            done = self.buffer.dones[i]
            v_s = self.buffer.state_values[i]
            v_s_next = self.buffer.state_values[i + 1] if i < data_len - 1 else 0
            
            delta = reward + self.gamma * v_s_next * (1 - done) - v_s
            
            if i == data_len - 1: # 最后一个时间步
                last_advantage = delta
            else:
                last_advantage = delta + self.gamma * self.lambda_gae * (1 - done) * advantages[0]
            advantages.insert(0, last_advantage) # 插入到列表开头

        returns = (torch.tensor(advantages, dtype=torch.float32) + torch.tensor(self.buffer.state_values,
                                                                                dtype=torch.float32)).detach()

        # 归一化优势
        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 2. 转换数据为 Tensor (所有数据)
        old_states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32).to(self.device)
        old_actions = torch.tensor(self.buffer.actions, dtype=torch.int64).to(self.device)
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32).to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        
        # 获取所有数据的索引
        sampler = BatchSampler(
            SubsetRandomSampler(range(data_len)),
            self.batch_size,
            drop_last=True # 丢弃不足 batch_size 的最后一个批次
        )

        # 3. 在同一个 rollout 数据上进行 K 轮优化 (使用 Mini-batch)
        for _ in range(self.k_epochs):
            for indices in sampler:
                # 提取 Mini-batch 数据
                mb_old_states = old_states[indices]
                mb_old_actions = old_actions[indices]
                mb_old_log_probs = old_log_probs[indices]
                mb_advantages = advantages[indices]
                mb_returns = returns[indices]

                action_dist, state_values = self.policy(mb_old_states)
                log_probs = action_dist.log_prob(mb_old_actions)
                dist_entropy = action_dist.entropy()

                ratios = torch.exp(log_probs - mb_old_log_probs.detach())

                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.loss_fn(state_values.squeeze(), mb_returns)
                entropy_bonus = -0.1 * dist_entropy.mean() # 熵奖励系数可以调整

                loss = actor_loss + 0.5 * critic_loss + entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                # --- 核心修改：应用梯度裁剪 ---
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        # 4. 将当前策略的权重复制到旧策略网络
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 5. 清空缓冲区
        self.buffer.clear()

    def save_model(self, path):
        torch.save(self.policy_old.state_dict(), path)

    def load_model(self, path):
        self.policy_old.load_state_dict(torch.load(path, map_location=self.device))
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
