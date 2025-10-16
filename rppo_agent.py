import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np


class RolloutBuffer:
    """
    PPO 的经验存储区。
    PPO 是 on-policy 算法，它收集一个完整的轨迹（rollout），然后用它来更新策略，
    更新完毕后就丢弃这些数据。这与 DQN 的经验回放缓冲区（ReplayBuffer）不同。
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


class ActorCriticRNN(nn.Module):
    """
    基于 RNN 的 Actor-Critic 网络。
    它有一个共享的 RNN 层来处理状态序列，然后是两个独立的头：
    1. Actor Head: 输出动作的概率分布。
    2. Critic Head: 输出当前状态的价值（V-value）。
    """

    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCriticRNN, self).__init__()

        # RNN 层 (GRU)
        self.rnn = nn.GRU(state_dim, hidden_dim, batch_first=True)

        # Actor Head
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Critic Head
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state_sequence):
        # state_sequence 形状: (batch_size, sequence_length, state_dim)
        rnn_out, _ = self.rnn(state_sequence)

        # 我们只关心序列的最后一个时间步的输出
        last_step_out = rnn_out[:, -1, :]

        # Critic Head: 评估状态价值
        state_value = self.critic_head(last_step_out)

        # Actor Head: 得到动作的概率分布
        action_logits = self.actor_head(last_step_out)
        action_dist = Categorical(logits=action_logits)

        return action_dist, state_value


class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, sequence_length,
                 lr_actor_critic, gamma, lambda_gae, eps_clip, k_epochs):

        self.gamma = gamma
        self.lambda_gae = lambda_gae  # GAE (Generalized Advantage Estimation) 的 lambda 参数
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs  # 在一个 rollout 上进行多轮更新

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PPO Agent 使用设备: {self.device}")

        self.buffer = RolloutBuffer()

        self.policy = ActorCriticRNN(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor_critic)

        # PPO 需要一个旧策略来计算概率比率，所以我们创建另一个网络
        self.policy_old = ActorCriticRNN(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.loss_fn = nn.MSELoss()

    def select_action(self, state_history):
        """
        根据当前策略选择一个动作。
        返回：动作，动作的对数概率，状态价值。
        """
        with torch.no_grad():
            state_history_tensor = torch.tensor(state_history, dtype=torch.float32).unsqueeze(0).to(self.device)
            action_dist, state_value = self.policy_old(state_history_tensor)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

        return action.item(), log_prob.item(), state_value.item()

    def store_transition(self, state, action, log_prob, reward, done, state_value):
        """将单步经验存入缓冲区。"""
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.log_probs.append(log_prob)
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(done)
        self.buffer.state_values.append(state_value)

    def update(self):
        """
        使用收集到的整个 rollout 数据来更新 Actor 和 Critic 网络。
        """
        # 1. 计算优势 (Advantage) 和回报 (Return) - 使用 GAE
        rewards = []
        advantages = []
        last_advantage = 0

        # 从后往前计算
        for i in reversed(range(len(self.buffer.rewards))):
            reward = self.buffer.rewards[i]
            done = self.buffer.dones[i]
            v_s = self.buffer.state_values[i]
            v_s_next = self.buffer.state_values[i + 1] if i < len(self.buffer.rewards) - 1 else 0

            # 计算 TD-error (delta)
            delta = reward + self.gamma * v_s_next * (1 - done) - v_s

            # 计算 GAE
            last_advantage = delta + self.gamma * self.lambda_gae * (1 - done) * last_advantage
            advantages.insert(0, last_advantage)

        # 计算回报 (Returns)
        returns = (torch.tensor(advantages, dtype=torch.float32) + torch.tensor(self.buffer.state_values,
                                                                                dtype=torch.float32)).detach()

        # 归一化优势
        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 2. 转换数据为 Tensor
        old_states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32).to(self.device)
        old_actions = torch.tensor(self.buffer.actions, dtype=torch.int64).to(self.device)
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32).to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        # 3. 在同一个 rollout 数据上进行 K 轮优化
        for _ in range(self.k_epochs):
            # 使用当前策略评估旧动作
            action_dist, state_values = self.policy(old_states)
            log_probs = action_dist.log_prob(old_actions)
            dist_entropy = action_dist.entropy()

            # 计算概率比率 r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
            ratios = torch.exp(log_probs - old_log_probs.detach())

            # 计算 PPO 的核心损失 (Clipped Surrogate Objective)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # 最终损失 = Actor损失 + Critic损失 - 熵奖励
            # 我们希望最大化 Actor 目标和熵，所以取负号
            # 我们希望最小化 Critic 损失，所以取正号
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.loss_fn(state_values.squeeze(), returns)
            entropy_bonus = -0.01 * dist_entropy.mean()  # 熵奖励系数 c2

            loss = actor_loss + 0.5 * critic_loss + entropy_bonus  # 价值损失系数 c1

            # 梯度更新
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 4. 将当前策略的权重复制到旧策略网络，为下一次 rollout 做准备
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 5. 清空缓冲区
        self.buffer.clear()

    def save_model(self, path):
        torch.save(self.policy_old.state_dict(), path)

    def load_model(self, path):
        self.policy_old.load_state_dict(torch.load(path, map_location=self.device))
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
