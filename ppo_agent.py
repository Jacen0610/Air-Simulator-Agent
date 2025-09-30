import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

class ActorCritic(nn.Module):
    """
    PPO 的 Actor-Critic 网络。
    它接收观测并输出动作 logits 和一个价值估计。
    """
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Actor 网络 (策略)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )

        # Critic 网络 (价值)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        """
        根据当前状态选择一个动作。
        """
        action_logits = self.actor(state)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob

    def evaluate(self, state, action):
        """
        评估旧状态和动作，用于计算损失。
        """
        action_logits = self.actor(state)
        dist = Categorical(logits=action_logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # 旧策略，用于计算 PPO 的比率
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def save(self, filepath):
        """
        保存模型状态 (只保存旧策略网络，因为新策略会从中复制)
        """
        torch.save(self.policy_old.state_dict(), filepath)

    def load(self, filepath):
        """
        加载模型状态
        """
        self.policy_old.load_state_dict(torch.load(filepath))
        # 确保 policy 网络也同步更新
        self.policy.load_state_dict(self.policy_old.state_dict())

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action, action_logprob = self.policy_old.act(state)
        return action, action_logprob

    def update(self, memory):
        # GAE (Generalized Advantage Estimation) 计算回报
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory['rewards']), reversed(memory['dones'])):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)

        # 转换为 tensor
        returns = torch.tensor(returns, dtype=torch.float32)

        # 转换 list 为 tensor
        old_states = torch.squeeze(torch.stack(memory['states'], dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(memory['actions'], dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory['logprobs'], dim=0)).detach()

        # 优化策略 K 个 epochs
        for _ in range(self.K_epochs):
            # 评估旧动作和价值
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # 计算优势
            advantages = returns - state_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 计算比率 (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # 计算 PPO 裁剪损失 (Surrogate Loss)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - 0.01 * dist_entropy

            # 梯度下降
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # 将新权重复制到旧策略中
        self.policy_old.load_state_dict(self.policy.state_dict())
