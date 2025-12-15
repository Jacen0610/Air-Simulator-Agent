import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# 导入为 LSTM 策略准备的新环境
from gym_env_for_lstm import GymEnvForLSTM


class EpisodeTerminationCallback(BaseCallback):
    def __init__(self, target_episodes: int, verbose: int = 0):
        super(EpisodeTerminationCallback, self).__init__(verbose)
        self.target_episodes = target_episodes
        self.episode_count = 0
        self.episode_rewards = []
        self.start_time = time.time()

    def _on_step(self) -> bool:
        # 这个回调实现对于单个或多个环境都是健壮的
        # 当使用单个环境时, dones 是一个 [bool] 形式的数组
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_count += 1

                if self.verbose > 0:
                    elapsed_time = time.time() - self.start_time
                    sps = self.num_timesteps / elapsed_time if elapsed_time > 0 else 0
                    print(f"Episode {self.episode_count}/{self.target_episodes} | "
                          f"Reward: {info['episode']['r']:.2f} | "
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
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"已将图表保存至: {filename}")
    plt.close()


def main():
    """
    主训练流程 - RecurrentPPO (单环境优化版)。
    """
    # --- 配置 ---
    TRAIN_EPISODES = 1000
    MODEL_DIR = "sb3_models"
    PLOT_DIR = "sb3_plots"
    MODEL_PATH = os.path.join(MODEL_DIR, "recurrent_ppo_lstm_single_env.zip")
    PLOT_PATH = os.path.join(PLOT_DIR, "training_rewards_recurrent_ppo_single_env.png")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # --- 1. 创建并封装单个环境 ---
    print("正在初始化为 LSTM 优化的 Gym 环境 (单实例)...")
    # [核心修改] 回退到使用单个 Monitor 封装的环境
    env = Monitor(GymEnvForLSTM())

    # --- 2. 定义并训练模型 ---
    print(f"开始使用 RecurrentPPO 进行训练，目标为 {TRAIN_EPISODES} 个 episodes...")
    train_callback = EpisodeTerminationCallback(target_episodes=TRAIN_EPISODES, verbose=1)

    # [优化] 保留对 LSTM 友好的超参数
    policy_kwargs = dict(
        lstm_hidden_size=128,
        n_lstm_layers=1,
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        policy_kwargs=policy_kwargs,
        n_steps=2048,  # 对于RNN，一个不太大的 n_steps 有助于更频繁地更新
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        learning_rate=1e-4,
        verbose=0,
        tensorboard_log="./recurrent_ppo_lstm_tensorboard_sb3/"
    )

    try:
        # 使用一个足够大的数，让训练由 callback 控制
        model.learn(total_timesteps=int(5e6), callback=train_callback)

        print(f"\n训练完成。正在保存模型至 {MODEL_PATH}...")
        model.save(MODEL_PATH)

        print("正在绘制训练奖励图表...")
        plot_rewards(
            train_callback.episode_rewards,
            f"RecurrentPPO Training Rewards ({len(train_callback.episode_rewards)} Episodes)",
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
