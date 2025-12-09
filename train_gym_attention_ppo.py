import gymnasium as gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from gym_env import GymEnv


# --- 1. 构建自定义的 Attention 特征提取器 ---
# 这是实现“专家路径”的核心

class AttentionExtractor(BaseFeaturesExtractor):
    """
    一个使用 Transformer Encoder (Attention) 的自定义特征提取器。

    这个网络的目标是将形状为 (N, 10, 7) 的观测序列，
    通过注意力机制，转换为一个形状为 (N, features_dim) 的高质量特征向量。
    这个特征向量随后会被送入 PPO 的 Actor 和 Critic 头部网络。

    :param observation_space: 环境的观测空间。
    :param features_dim: 您希望从这个提取器输出的特征向量维度。
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        # features_dim 是您希望输出的特征维度，可以自由设置
        super().__init__(observation_space, features_dim)

        # 从观测空间获取输入的维度信息
        # 对于我们的环境，obs_shape 是 (10, 7)
        seq_len, features_in = observation_space.shape

        # --- 定义您的注意力网络层 ---

        # 使用 PyTorch 内置的 TransformerEncoderLayer 作为注意力核心。
        # 这是构建 Transformer 的标准模块。
        # d_model: Transformer 输入特征的维度，必须是 7。
        # nhead: 多头注意力的头数。对于较小的 d_model，1 或 7 的因子是好的选择。
        # dim_feedforward: Transformer 内部前馈网络的维度。
        # batch_first=True: 关键！确保输入形状是 (批量大小, 序列长度, 特征数)。
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=features_in,
            nhead=1,
            dim_feedforward=256,
            batch_first=True,
            activation='relu'
        )

        # 将单个 Encoder Layer 堆叠成一个完整的 Transformer Encoder。
        # num_layers 控制堆叠的层数，这是模型复杂度的关键参数。
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 定义一个线性层，用于将 Attention 的输出映射到最终的特征维度。
        # Transformer 的输出形状仍然是 (N, 10, 7)。我们需要将其压平。
        # 压平后的维度是 seq_len * features_in (10 * 7 = 70)。
        self.linear = nn.Sequential(
            nn.Linear(seq_len * features_in, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        这是网络的前向传播方法。
        """
        # observations 的输入形状: (N, 10, 7)

        # 1. 将观测数据通过 Transformer Encoder
        # 输出的 encoded_features 形状仍然是 (N, 10, 7)
        # 但它内部的每个时间步的特征都包含了对整个序列的注意力信息。
        encoded_features = self.transformer_encoder(observations)

        # 2. 将带有注意力信息的序列特征压平，以便送入最后的线性层。
        # 形状从 (N, 10, 7) 变为 (N, 70)
        flattened_features = torch.flatten(encoded_features, start_dim=1)

        # 3. 通过线性层得到最终的特征向量。
        # 输出形状: (N, features_dim)，例如 (N, 128)
        return self.linear(flattened_features)


# --- 2. 复用之前的回调和绘图函数 ---

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


def plot_rewards(rewards: list, title: str, filename: str):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"已将图表保存至: {filename}")
    plt.close()


# --- 3. 主训练流程 ---

def main():
    TRAIN_EPISODES = 50
    MODEL_DIR = "sb3_models"
    PLOT_DIR = "sb3_plots"
    MODEL_PATH = os.path.join(MODEL_DIR, "attention_ppo_model.zip")
    PLOT_PATH = os.path.join(PLOT_DIR, "training_rewards_attention_ppo.png")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("正在初始化 Gym 环境...")
    env = Monitor(GymEnv())

    # --- 关键步骤：定义 policy_kwargs 来使用我们的自定义提取器 ---
    policy_kwargs = {
        "features_extractor_class": AttentionExtractor,
        "features_extractor_kwargs": dict(features_dim=128),  # 设置输出特征维度
    }

    print("开始使用自定义 Attention 策略进行训练...")
    train_callback = EpisodeTerminationCallback(target_episodes=TRAIN_EPISODES, verbose=1)

    # 仍然使用标准的 PPO 算法，但通过 policy_kwargs 注入了我们的 Attention 网络。
    # 策略名称仍然是 "MlpPolicy"，因为我们只替换了特征提取器（身体），
    # 而 Actor-Critic 的头部（Head）仍然是 MLP。
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        n_steps=8192,
        batch_size=256,  # 对于复杂模型，稍大的 batch_size 可能更稳定
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # 鼓励探索
        learning_rate=3e-4,
        verbose=0,
        tensorboard_log="./attention_ppo_tensorboard_sb3/"
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