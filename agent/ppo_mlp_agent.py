# ppo_mlp_agent.py
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np


class RolloutBuffer:
    """
    PPO的经验存储区 (on-policy)。
    这个缓冲区现在存储的是单个状态，而不是状态历史。
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


class ActorCriticMLP(nn.Module):
    """
    [新] 基于前馈网络 (MLP) 的 Actor-Critic 网络。
    这是一个“无记忆”的网络，只处理当前状态。
    """

    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCriticMLP, self).__init__()

        # 共享的主干网络
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Actor Head
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Critic Head
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        # state 形状: (batch_size, state_dim)
        shared_features = self.shared_net(state)

        # Critic Head: 评估状态价值
        state_value = self.critic_head(shared_features)

        # Actor Head: 得到动作的概率分布
        action_logits = self.actor_head(shared_features)
        action_dist = Categorical(logits=action_logits)

        return action_dist, state_value


class PPOMLPAgent:
    """
    使用 MLP 网络的 PPO 智能体。
    """

    def __init__(self, state_dim, action_dim, hidden_dim,
                 lr_actor_critic, gamma, lambda_gae, eps_clip, k_epochs):

        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"MLP PPO Agent 使用设备: {self.device}")

        self.buffer = RolloutBuffer()

        # 实例化新的 MLP 网络
        self.policy = ActorCriticMLP(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor_critic)

        self.policy_old = ActorCriticMLP(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        """
        根据当前状态选择动作。
        """
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            action_dist, state_value = self.policy_old(state_tensor)
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
            action_dist, state_values = self.policy(old_states)
            log_probs = action_dist.log_prob(old_actions)
            dist_entropy = action_dist.entropy()

            ratios = torch.exp(log_probs - old_log_probs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.loss_fn(state_values.squeeze(), returns)
            entropy_bonus = -0.01 * dist_entropy.mean()

            loss = actor_loss + 0.5 * critic_loss + entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
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