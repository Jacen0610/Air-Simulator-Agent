import sys
import os
# --- 动态添加项目根目录到 sys.path ---
# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 假设项目根目录是脚本所在目录的父目录 (Air-Simulator-Agent/)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
# 将项目根目录添加到 sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# --- 核心修改：导入新的 AviationAttentionExtractor ---
from agent.attention_feature_extractor import AviationAttentionExtractor
from env.gym_env import GymEnv 

# --- 自定义回调：记录奖励并按 Episode 数量停止训练 ---
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
                          f"Normalized Reward: {info['episode']['r']:.2f} | "
                          f"Total Steps: {self.num_timesteps} | "
                          f"SPS: {sps:.2f}")

        if self.episode_count >= self.target_episodes:
            print(f"\n已达到目标 {self.target_episodes} 个 episodes，停止训练。")
            return False
        return True

# --- 新增：用于可视化注意力权重的回调 ---
class AttentionVisualizationCallback(RewardAndEpisodeCallback):
    def __init__(self, total_episodes: int, viz_freq: int, verbose=0):
        super().__init__(total_episodes, verbose)
        self.viz_freq = viz_freq

    def _on_step(self) -> bool:
        # 首先调用父类的 _on_step 方法
        continue_training = super()._on_step()
        if not continue_training:
            return False

        # 每隔 viz_freq 个 episode 触发一次
        if self.episode_count > 0 and self.episode_count % self.viz_freq == 0:
            # 确保在 episode 结束时才执行
            if self.locals['dones'][0]:
                self.visualize_attention()
        
        return True

    def visualize_attention(self):
        print(f"\n--- Episode {self.episode_count}: 绘制注意力热力图 ---")

        feature_extractor = self.model.policy.features_extractor
        self.model.policy.eval()

        # 获取当前最新观测并转换为 Tensor
        obs_tensor, _ = self.model.policy.obs_to_tensor(self.locals['new_obs'])

        # 触发前向传播以更新权重
        with th.no_grad():
            feature_extractor(obs_tensor)

        # 获取权重 (修改后的 Extractor 返回的是最后一行，即当前对历史的关注)
        # 形状应该是 (seq_len,)
        attn_weights = feature_extractor.last_attn_weights

        self.model.policy.train()

        if attn_weights is not None:
            plt.figure(figsize=(12, 3))

            # 将 1D 权重转换为 2D 矩阵形状 (1, seq_len) 方便绘制热力图
            data = attn_weights.reshape(1, -1)

            # 绘制热力图
            ax = sns.heatmap(data, annot=True, fmt=".2f", cmap="YlGnBu",
                             cbar_kws={'label': 'Attention Weight'})

            # 设置坐标轴
            ax.set_title(f'Attention Focus at Episode {self.episode_count}\n(What the Agent is looking at RIGHT NOW)')
            ax.set_xlabel('Steps back in History (0 is oldest, 31 is newest)')
            ax.set_yticklabels(['Current Decision'])

            # 强调最后几个 Step（微观）和较早的 Step（宏观）
            plt.tight_layout()

            # 写入 TensorBoard
            self.logger.record(f"attention/heatmap_ep_{self.episode_count}",
                               plt.gcf(), exclude=("stdout", "log", "json", "csv"))
            plt.close()

# --- 绘制奖励曲线图的函数 ---
def plot_rewards(rewards, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(rewards) + 1), rewards)
    plt.xlabel("Episode Number")
    plt.ylabel("Normalized Episode Reward")
    plt.title("SB3 Multi-Head Attention PPO Normalized Reward Curve")
    
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_major_formatter().set_useOffset(False)
    
    plt.grid(True)
    plt.savefig(filename)
    print(f"奖励曲线图已保存至: {filename}")

# --- 主训练函数 ---
def main():
    # --- 训练设置 ---
    TOTAL_TRAINING_EPISODES = 500
    SEQUENCE_LENGTH = 32
    FEATURES_DIM = 128 # 最终从提取器输出的特征维度
    
    # --- 文件路径设置 ---
    MODEL_DIR = os.path.join(project_root, "SB3/models")
    PLOT_DIR = os.path.join(project_root, "SB3/plots/train")
    TENSORBOARD_LOG_DIR = os.path.join(project_root, "sb3_logs")
    
    MODEL_PATH = os.path.join(MODEL_DIR, "sb3_attention_ppo.zip")
    STATS_PATH = os.path.join(MODEL_DIR, "sb3_attention_ppo_vec_normalize.pkl")
    PLOT_PATH = os.path.join(PLOT_DIR, "sb3_attention_ppo.png")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # --- 1. 创建并包装环境 ---
    # 使用 GymEnv，因为它返回的是序列数据
    vec_env = make_vec_env(lambda: GymEnv(grpc_server_address='localhost:50050', sequence_length=SEQUENCE_LENGTH), n_envs=1)
    env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, gamma=0.999) # 使用新的 gamma

    # --- 2. 定义模型 ---
    # 定义策略参数
    policy_kwargs = dict(
        features_extractor_class=AviationAttentionExtractor,
        features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
        net_arch=dict(pi=[128, 64], vf=[128, 64]) # 后端的 MLP 决策头 (注意: SB3 中价值函数是 vf)
    )

    # 定义PPO模型
    model = PPO(
        "MlpPolicy", # 即使是自定义提取器，基类通常选 MlpPolicy
        env, 
        policy_kwargs=policy_kwargs,
        verbose=0,
        learning_rate=2e-4,
        gamma=0.999,      # 重要：针对 90s 长周期，Gamma 必须大
        n_steps=4096,     # 每次更新采集的步数
        batch_size=128,
        ent_coef=0.02,    # 增加探索，防止 Agent 变“胆小”
        tensorboard_log="./sb3_logs/"
    )
    
    # --- 3. 训练模型 ---
    # 使用新的回调函数，每 10 个 episode 可视化一次注意力
    train_callback = AttentionVisualizationCallback(total_episodes=TOTAL_TRAINING_EPISODES, viz_freq=10, verbose=1)
    
    try:
        model.learn(
            total_timesteps=int(1e9),
            callback=train_callback,
            tb_log_name="PPO_MultiHeadAttention_Normalized"
        )
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- 4. 训练后操作 ---
        print("\n训练完成或中断。正在保存模型和统计数据...")
        
        model.save(MODEL_PATH)
        print(f"模型已保存至: {MODEL_PATH}")

        env.save(STATS_PATH)
        print(f"环境统计数据已保存至: {STATS_PATH}")
        
        if train_callback.episode_rewards:
            plot_rewards(train_callback.episode_rewards, PLOT_PATH)
        else:
            print("没有足够的奖励数据来生成图表。")

        env.close()
        print("环境已关闭。")

# --- 脚本入口 ---
if __name__ == '__main__':
    main()
