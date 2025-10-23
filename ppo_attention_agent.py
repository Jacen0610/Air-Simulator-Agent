# ppo_attention_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


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


class ActorCriticAttentionRNN(nn.Module):
    """
    [创新网络] 基于注意力机制的 RNN Actor-Critic 网络。
    """
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCriticAttentionRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(state_dim, hidden_dim, batch_first=True)
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    # [核心修改] 增加一个可选参数，用于返回注意力权重
    def forward(self, state_sequence, return_weights=False):
        rnn_outputs, last_hidden = self.rnn(state_sequence)
        last_hidden_expanded = last_hidden.permute(1, 0, 2).repeat(1, rnn_outputs.size(1), 1)
        attention_input = torch.cat([rnn_outputs, last_hidden_expanded], dim=2)
        scores = self.attention_net(attention_input)
        attention_weights = F.softmax(scores, dim=1)
        context_vector = torch.sum(attention_weights * rnn_outputs, dim=1, keepdim=True)
        focused_representation = context_vector.squeeze(1)
        state_value = self.critic_head(focused_representation)
        action_logits = self.actor_head(focused_representation)
        action_dist = Categorical(logits=action_logits)

        # 根据需要返回权重
        if return_weights:
            return action_dist, state_value, attention_weights
        else:
            return action_dist, state_value


class PPOAttentionAgent:
    """
    使用带注意力机制的 Actor-Critic 网络的 PPO 智能体。
    """
    def __init__(self, state_dim, action_dim, hidden_dim, sequence_length,
                 lr_actor_critic, gamma, lambda_gae, eps_clip, k_epochs):
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Attention PPO Agent 使用设备: {self.device}")
        self.buffer = RolloutBuffer()
        self.policy = ActorCriticAttentionRNN(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor_critic)
        self.policy_old = ActorCriticAttentionRNN(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.loss_fn = nn.MSELoss()

    # [核心修改] 新增一个方法，用于在评估时获取权重
    def select_action_and_get_weights(self, state_history):
        with torch.no_grad():
            state_history_tensor = torch.tensor(state_history, dtype=torch.float32).unsqueeze(0).to(self.device)
            # 调用可以返回权重的前向传播方法
            action_dist, state_value, attention_weights = self.policy_old(state_history_tensor, return_weights=True)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
        # 返回动作以及注意力权重
        return action.item(), log_prob.item(), state_value.item(), attention_weights.squeeze().cpu().numpy()

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
        returns = (torch.tensor(advantages, dtype=torch.float32) + torch.tensor(self.buffer.state_values, dtype=torch.float32)).detach()
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
            entropy_bonus = -0.05 * dist_entropy.mean()
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