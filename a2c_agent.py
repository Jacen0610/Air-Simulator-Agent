import torch
import torch.nn as nn
from torch.distributions import Categorical

# Actor-Critic 网络结构
class ActorCritic(nn.Module):
    """
    A2C 的 Actor-Critic 网络。
    它接收观测并输出动作概率和一个价值估计。
    """
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Actor 网络 (策略)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
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
        # A2C 不直接使用 forward，而是通过 act 和 evaluate
        raise NotImplementedError

    def act(self, state):
        """
        根据当前状态选择一个动作。
        返回动作和其对数概率。
        """
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob

    def evaluate(self, state, action):
        """
        评估状态和动作。
        返回动作的对数概率、状态的价值和分布的熵。
        """
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy


# A2C 智能体
class A2CAgent:
    def __init__(self, state_dim, action_dim, lr, gamma):
        self.lr = lr
        self.gamma = gamma

        # A2C 使用单个策略网络
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        """
        使用当前策略选择动作
        """
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action, action_logprob = self.policy.act(state)
        return action, action_logprob

    def update(self, memory):
        """
        使用收集到的经验更新网络
        """
        # 计算折扣回报 (Returns)
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory['rewards']), reversed(memory['dones'])):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # 标准化回报
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # 转换 list 为 tensor
        old_states = torch.squeeze(torch.stack(memory['states'], dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(memory['actions'], dim=0)).detach()
        
        # 评估动作和价值
        logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

        # 计算优势 (Advantage)
        advantages = rewards - state_values.detach()
        
        # 计算 Actor (策略) 损失和 Critic (价值) 损失
        loss = -(logprobs * advantages).mean() + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy.mean()

        # 梯度下降
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()