import sys
import os

# --- 动态添加项目根目录 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# --- 核心修改：导入更新后的 Transformer Extractor ---
from agent.attention_feature_extractor import AviationTransformerExtractor
from env.gym_env import GymEnv


class RewardAndEpisodeCallback(BaseCallback):
    def __init__(self, total_episodes: int, verbose=0):
        super(RewardAndEpisodeCallback, self).__init__(verbose)
        self.total_episodes = total_episodes
        self.episode_count = 0
        self.episode_rewards = []
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_count += 1
                if self.verbose > 0:
                    elapsed_time = time.time() - self.start_time
                    sps = self.num_timesteps / elapsed_time if elapsed_time > 0 else 0
                    print(f"Episode {self.episode_count}/{self.total_episodes} | "
                          f"Reward: {info['episode']['r']:.2f} | SPS: {sps:.2f}")

        # 修正：判断停止条件
        if self.episode_count >= self.total_episodes:
            print(f"\n已达到目标 {self.total_episodes} 个 episodes，停止训练。")
            return False
        return True


class AttentionVisualizationCallback(RewardAndEpisodeCallback):
    def __init__(self, total_episodes: int, viz_freq: int, verbose=0):
        super().__init__(total_episodes, verbose)
        self.viz_freq = viz_freq

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        if not continue_training: return False

        if self.episode_count > 0 and self.episode_count % self.viz_freq == 0:
            if self.locals['dones'][0]:
                self.visualize_attention()
        return True

    def visualize_attention(self):
        """
        注意：针对 TransformerEncoder，获取权重需要 hook 或者在 forward 中显式返回。
        这里我们展示一个通用的热力图逻辑占位。
        """
        print(f"\n--- Episode {self.episode_count}: 记录训练状态 ---")
        # 记录当前的平均奖励到 TensorBoard
        if len(self.episode_rewards) > 0:
            avg_rew = np.mean(self.episode_rewards[-self.viz_freq:])
            self.logger.record("train/avg_episode_reward", avg_rew)


def plot_rewards(rewards, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(rewards) + 1), rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Transformer-PPO Training Curve")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def main():
    # --- 1. 核心参数设置 ---
    TOTAL_TRAINING_EPISODES = 500  # Transformer 需要略多一点的训练量
    SEQUENCE_LENGTH = 32
    FEATURES_DIM = 256  # Transformer 输出的特征向量长度
    EMBED_DIM = 128  # 内部 Embedding 维度

    MODEL_DIR = os.path.join(project_root, "SB3/models")
    PLOT_DIR = os.path.join(project_root, "SB3/plots/train")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # --- 2. 环境初始化 (12 维状态已经在 GymEnv 中适配) ---
    vec_env = make_vec_env(lambda: GymEnv(
        grpc_server_address='localhost:50051',
        sequence_length=SEQUENCE_LENGTH
    ), n_envs=1)

    # 注意：对于异步时间 delta，建议开启 clip_obs 以增强稳定性
    env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, gamma=0.999)

    # --- 3. 定义 Transformer-PPO 策略参数 ---
    policy_kwargs = dict(
        features_extractor_class=AviationTransformerExtractor,
        features_extractor_kwargs=dict(
            features_dim=FEATURES_DIM,
            embed_dim=EMBED_DIM
        ),
        # 决策头：Transformer 已经提取了很强的特征，后端的 MLP 不需要太深
        net_arch=dict(pi=[256, 128], vf=[256, 128])
    )

    # --- 4. 实例化模型 (使用为你推荐的 12 维优化超参) ---
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=1e-5,  # 推荐值
        gamma=0.999,  # 针对长周期
        n_steps=16384,  # 每次更新采集的样本量
        batch_size=2048, # 增加 Batch 以平滑碰撞脉冲
        n_epochs=5, # 每次更新迭代10遍
        ent_coef=0.0005,  # 稍作降低，让模型更聚焦于已发现的空隙
        gae_lambda=0.98,
        clip_range=0.1,
        max_grad_norm=0.1,
        device="cuda",
        tensorboard_log="./sb3_logs/"
    )

    train_callback = AttentionVisualizationCallback(
        total_episodes=TOTAL_TRAINING_EPISODES,
        viz_freq=25,
        verbose=1
    )

    try:
        print("开始 Transformer-PPO 训练...")
        model.learn(
            total_timesteps=int(1e12),
            callback=train_callback,
            tb_log_name="PPO_Aviation_Transformer_12D"
        )
    except Exception as e:
        print(f"报错: {e}")
    finally:
        model.save(os.path.join(MODEL_DIR, "sb3_transformer_ppo_final"))
        env.save(os.path.join(MODEL_DIR, "sb3_transformer_ppo_vec_normalize_final.pkl"))
        if train_callback.episode_rewards:
            plot_rewards(train_callback.episode_rewards, os.path.join(PLOT_DIR, "reward_curve.png"))
        env.close()


if __name__ == '__main__':
    main()