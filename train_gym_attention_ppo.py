import gymnasium as gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker # 导入 ticker 库用于美化坐标轴
import os
import math

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from gym_env import GymEnv

# --- 1. 位置编码模块 ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# --- 2. 集成位置编码的 AttentionExtractor ---
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

# --- 3. 回调函数 (保持不变) ---
class EpisodeTerminationCallback(BaseCallback):
    def __init__(self, target_episodes: int, verbose: int = 0):
        super().__init__(verbose)
        self.target_episodes = target_episodes
        self.episode_count = 0
        self.episode_rewards = []
        self.last_timestep = 0

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_count += 1
                episode_steps = self.num_timesteps - self.last_timestep
                self.last_timestep = self.num_timesteps
                if self.verbose > 0:
                    print(f"Episode {self.episode_count}/{self.target_episodes} | "
                          f"Reward: {info['episode']['r']:.2f} | "
                          f"Steps: {episode_steps} | "
                          f"Total Steps: {self.num_timesteps}")
        if self.episode_count >= self.target_episodes:
            print(f"\n已达到目标 {self.target_episodes} 个 episodes，停止训练。")
            return False
        return True

# --- 4. 更新后的绘图函数 ---
def plot_rewards(rewards: list, title: str, filename: str):
    """
    为训练过程绘制奖励曲线，并确保 Y 轴清晰易读。
    """
    if not rewards:
        print("没有可供绘制的奖励数据。")
        return

    episodes = range(1, len(rewards) + 1)
    
    plt.figure(figsize=(15, 8)) # 增大画布尺寸以容纳更多标签
    
    # 绘制线图和散点图
    plt.plot(episodes, rewards, color='dodgerblue', linestyle='-', linewidth=1.5, alpha=0.7, label='Episode Reward')
    plt.scatter(episodes, rewards, color='red', zorder=5, s=20) # s是点的大小

    # 仅为部分点添加标签，避免过于拥挤
    # 例如，每隔 N 个点或者只为最高/最低点添加标签
    if len(rewards) > 50:
        label_interval = len(rewards) // 25 # 大约显示25个标签
    else:
        label_interval = 1 # 如果点不多，全部显示

    for i, reward in enumerate(rewards):
        if i % label_interval == 0:
            plt.text(episodes[i], reward, f' {reward:.1f}', fontsize=9, va='center')

    plt.title(title, fontsize=16)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    
    # 设置 X 轴为整数刻度
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(filename)
    print(f"已将图表保存至: {filename}")
    plt.close()

# --- 5. 主训练流程 (保持不变) ---
def main():
    TRAIN_EPISODES = 50
    MODEL_DIR = "sb3_models"
    PLOT_DIR = "sb3_plots"
    MODEL_PATH = os.path.join(MODEL_DIR, "attention_ppo_model_with_pe.zip")
    PLOT_PATH = os.path.join(PLOT_DIR, "training_rewards_attention_ppo_with_pe.png")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("正在初始化 Gym 环境...")
    env = Monitor(GymEnv())

    policy_kwargs = {
        "features_extractor_class": AttentionExtractor,
        "features_extractor_kwargs": dict(features_dim=128),
    }

    print("开始使用带位置编码的 Attention 策略进行训练...")
    train_callback = EpisodeTerminationCallback(target_episodes=TRAIN_EPISODES, verbose=1)

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        n_steps=8192,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        learning_rate=3e-4,
        verbose=0,
        tensorboard_log="./attention_ppo_pe_tensorboard_sb3/"
    )

    try:
        model.learn(total_timesteps=int(1e12), callback=train_callback)
        
        print(f"\n训练完成。正在保存模型至 {MODEL_PATH}...")
        model.save(MODEL_PATH)

        print("正在绘制训练奖励图表...")
        plot_rewards(
            train_callback.episode_rewards,
            f"Attention PPO Training Rewards ({len(train_callback.episode_rewards)} Episodes)",
            PLOT_PATH
        )

    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
    finally:
        print("\n流程结束，关闭环境。")
        env.close()

if __name__ == '__main__':
    main()
