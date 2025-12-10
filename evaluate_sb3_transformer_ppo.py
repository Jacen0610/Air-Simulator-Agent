import gymnasium as gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import math

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from gym_env import GymEnv

# --- 1. 从训练脚本中引入必要的自定义模块 ---
# 这些模块对于加载使用自定义特征提取器的模型至关重要

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        
        pe[0, :, 0::2] = torch.sin(position * div_term)
        
        num_odd_indices = d_model // 2
        if num_odd_indices > 0:
            pe[0, :, 1::2] = torch.cos(position * div_term[:num_odd_indices])
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class AttentionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        seq_len, features_in = observation_space.shape
        self.positional_encoding = PositionalEncoding(d_model=features_in, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=features_in, nhead=1, dim_feedforward=256,
            batch_first=True, activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.linear = nn.Sequential(
            nn.Linear(seq_len * features_in, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations_with_pe = self.positional_encoding(observations)
        encoded_features = self.transformer_encoder(observations_with_pe)
        flattened_features = torch.flatten(encoded_features, start_dim=1)
        return self.linear(flattened_features)

# --- 2. 从评估脚本中引入绘图函数 ---
def plot_evaluation_rewards(rewards: list, title: str, filename: str):
    if not rewards:
        print("没有可供绘制的奖励数据。")
        return

    episodes = range(1, len(rewards) + 1)
    
    plt.figure(figsize=(12, 7))
    
    plt.plot(episodes, rewards, color='dodgerblue', linestyle='-', linewidth=2, label='Episode Reward')
    plt.scatter(episodes, rewards, color='red', zorder=5)

    for i, reward in enumerate(rewards):
        plt.text(episodes[i], reward, f' {reward:.2f}', va='center')

    plt.title(title, fontsize=16)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(filename)
    print(f"评估奖励图表已保存至: {filename}")
    plt.close()

# --- 3. 主评估流程 ---
def main():
    # --- 配置 ---
    EVAL_EPISODES = 10
    MODEL_DIR = "sb3_models"
    PLOT_DIR = "sb3_plots"
    # 关键：确保模型路径指向正确的 Transformer 模型文件
    MODEL_PATH = os.path.join(MODEL_DIR, "attention_ppo_model_with_pe.zip")
    PLOT_FILENAME = os.path.join(PLOT_DIR, "evaluation_rewards_transformer_ppo.png")

    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型文件 '{MODEL_PATH}'。")
        print("请先运行 train_gym_attention_ppo.py 脚本来训练并保存一个模型。")
        return

    os.makedirs(PLOT_DIR, exist_ok=True)

    # --- 1. 创建环境并加载模型 ---
    print("正在初始化 Gym 环境...")
    env = GymEnv()

    # 关键：在加载模型时，必须提供与训练时相同的 policy_kwargs
    policy_kwargs = {
        "features_extractor_class": AttentionExtractor,
        "features_extractor_kwargs": dict(features_dim=128),
    }

    print(f"正在从 {MODEL_PATH} 加载已训练的 Transformer PPO 模型...")
    try:
        # 关键：传入 policy_kwargs 和 device
        model = PPO.load(MODEL_PATH, env=env, policy_kwargs=policy_kwargs, device='cpu')
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        env.close()
        return

    # --- 2. 运行评估循环 ---
    print(f"\n开始评估模型，共 {EVAL_EPISODES} 个 episodes...")
    
    eval_rewards = []
    for i in range(EVAL_EPISODES):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        while not done:
            # 使用确定性动作进行评估
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
        
        eval_rewards.append(episode_reward)
        print(f"评估 Episode {i + 1}/{EVAL_EPISODES} | Reward: {episode_reward:.2f} | Steps: {step_count}")

    # --- 3. 绘制并保存奖励图表 ---
    print("\n评估完成。正在绘制奖励图表...")
    plot_evaluation_rewards(
        eval_rewards,
        f"Transformer PPO Evaluation Rewards ({len(eval_rewards)} Episodes)",
        PLOT_FILENAME
    )

    # --- 4. 清理 ---
    print("\n流程结束，关闭环境。")
    env.close()

if __name__ == '__main__':
    main()
