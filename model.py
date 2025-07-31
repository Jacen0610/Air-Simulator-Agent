import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(ActorCritic, self).__init__()

        # 一个共享网络层，用于提取特征
        self.shared_layers = nn.Sequential(
            nn.Linear(observation_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # 演员（Actor）头：输出动作的概率分布
        self.actor_head = nn.Linear(64, action_dim)

        # 评论家（Critic）头：输出当前状态的价值（V-value）
        self.critic_head = nn.Linear(64, 1)

    def forward(self, observation):
        """
        前向传播，根据观测输出动作分布和状态价值。
        """
        # 确保输入是 Tensor
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32)

        features = self.shared_layers(observation)

        # 计算动作的 logits
        action_logits = self.actor_head(features)
        # 使用 Softmax 将 logits 转换为概率分布
        action_dist = Categorical(logits=action_logits)

        # 计算状态价值
        state_value = self.critic_head(features)

        return action_dist, state_value