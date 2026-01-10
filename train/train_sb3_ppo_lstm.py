import sys
import os
import argparse # 导入 argparse

# --- 动态添加项目根目录到 sys.path ---
# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 假设项目根目录是脚本所在目录的父目录 (Air-Simulator-Agent/)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
# 将项目根目录添加到 sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import time

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
# --- 新增导入 ---
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# 导入为 LSTM 策略准备的新环境
from env.gym_env_for_lstm import GymEnvForLSTM


class EpisodeTerminationCallback(BaseCallback):
    def __init__(self, target_episodes: int, verbose: int = 0):
        super(EpisodeTerminationCallback, self).__init__(verbose)
        self.target_episodes = target_episodes
        self.episode_count = 0
        self.episode_rewards = []
        self.start_time = time.time()

    def _on_step(self) -> bool:
        # 当 VecEnv 自动重置时，info 字典中会包含 'episode' 键
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            if 'episode' in info:
                # 注意：由于使用了 VecNormalize，这里的奖励是归一化后的
                self.episode_rewards.append(info['episode']['r'])
                self.episode_count += 1

                if self.verbose > 0:
                    elapsed_time = time.time() - self.start_time
                    sps = self.num_timesteps / elapsed_time if elapsed_time > 0 else 0
                    print(f"Episode {self.episode_count}/{self.target_episodes} | "
                          f"Normalized Reward: {info['episode']['r']:.2f} | "
                          f"Total Steps: {self.num_timesteps} | "
                          f"SPS: {sps:.2f}")

        if self.episode_count >= self.target_episodes:
            print(f"\n已达到目标 {self.target_episodes} 个 episodes，停止训练。")
            return False

        return True


def plot_rewards(rewards: list, title: str, filename: str):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards)
    if len(rewards) >= 50:
        moving_avg = np.convolve(rewards, np.ones(50)/50, mode='valid')
        plt.plot(np.arange(len(moving_avg)) + 49, moving_avg, color='red', linewidth=2, label='Moving Average (50 episodes)')
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Normalized Total Reward") # Y轴标签更新为归一化奖励
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"已将图表保存至: {filename}")
    plt.close()


def main():
    """
    主训练流程 - RecurrentPPO (带归一化)。
    """
    # --- 解析命令行参数 ---
    parser = argparse.ArgumentParser(description='Train SB3 Recurrent PPO Agent')
    parser.add_argument('--grpc_port', type=str, default='50051', help='gRPC server port (default: 50051)')
    args = parser.parse_args()
    grpc_address = f'localhost:{args.grpc_port}'
    print(f"Using gRPC server address: {grpc_address}")

    # --- 配置 ---
    TRAIN_EPISODES = 50
    GAMMA = 0.95
    SEQUENCE_LENGTH = 32
    # --- 使用绝对路径 ---
    MODEL_DIR = os.path.join(project_root, "SB3/models")
    PLOT_DIR = os.path.join(project_root, "SB3/plots/train")
    TENSORBOARD_LOG_DIR = os.path.join(project_root, "sb3_logs/lstm_ppo/")
    
    MODEL_PATH = os.path.join(MODEL_DIR, "recurrent_ppo_lstm.zip")
    STATS_PATH = os.path.join(MODEL_DIR, "recurrent_ppo_lstm_vec_normalize.pkl")
    PLOT_PATH = os.path.join(PLOT_DIR, "sb3_ppo_LSTM_training_rewards_recurrent.png")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # --- 1. 创建并封装环境 ---
    print("正在初始化为 LSTM 优化的 Gym 环境并应用归一化...")
    # 使用 make_vec_env 创建矢量化环境
    vec_env = make_vec_env(lambda: GymEnvForLSTM(grpc_server_address=grpc_address, sequence_length=SEQUENCE_LENGTH), n_envs=1)
    # 使用 VecNormalize 包装器来归一化观测值和奖励
    env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, gamma=GAMMA)

    # --- 2. 定义并训练模型 ---
    print(f"开始使用 RecurrentPPO 进行训练，目标为 {TRAIN_EPISODES} 个 episodes...")
    train_callback = EpisodeTerminationCallback(target_episodes=TRAIN_EPISODES, verbose=1)

    policy_kwargs = dict(
        lstm_hidden_size=128,
        n_lstm_layers=1,
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=0,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        gamma=GAMMA,
        n_steps=2048,  # 增加更新频率
        batch_size=512,  # 适配 FPS
        n_epochs=10,  # 充分利用每批数据
        clip_range=0.2,
        gae_lambda=0.95,
        ent_coef=0.1,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.015,
        device="cuda",
        tensorboard_log=TENSORBOARD_LOG_DIR
    )

    try:
        model.learn(total_timesteps=int(1e9), callback=train_callback)

        print(f"\n训练完成。正在保存模型至 {MODEL_PATH}...")
        model.save(MODEL_PATH)
        
        print(f"正在保存 VecNormalize 统计数据至 {STATS_PATH}...")
        env.save(STATS_PATH)

        print("正在绘制训练奖励图表...")
        plot_rewards(
            train_callback.episode_rewards,
            f"RecurrentPPO(LSTM) Training Rewards ({len(train_callback.episode_rewards)} Episodes)",
            PLOT_PATH
        )

    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n流程结束，关闭环境。")
        env.close()


if __name__ == '__main__':
    main()
