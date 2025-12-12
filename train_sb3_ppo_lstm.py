import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import os

from gym_env_for_lstm import GymEnvForLSTM


# ... (回调函数和绘图函数保持不变) ...
class EpisodeTerminationCallback(BaseCallback):
    def __init__(self, target_episodes: int, verbose: int = 0):
        super(EpisodeTerminationCallback, self).__init__(verbose)
        self.target_episodes = target_episodes
        self.episode_count = 0
        self.episode_rewards = []
        self.last_timestep = 0

    def _on_step(self) -> bool:
        if np.any(self.locals['dones']):
            for i, done in enumerate(self.locals['dones']):
                if done:
                    info = self.locals['infos'][i]
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


def main():
    """
    主训练流程 - PPO + LSTM 版本 (使用 SB3 v2.x 的最终正确配置)。
    """
    # --- 配置 ---
    TRAIN_EPISODES = 50
    MODEL_DIR = "sb3_models"
    PLOT_DIR = "sb3_plots"
    MODEL_PATH = os.path.join(MODEL_DIR, "ppo_lstm_gym_env.zip")
    PLOT_PATH = os.path.join(PLOT_DIR, "training_rewards_ppo_lstm.png")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # --- 1. 创建并封装环境 ---
    print("正在初始化为 LSTM 优化的 Gym 环境...")
    env = Monitor(GymEnvForLSTM())

    # --- 2. 训练模型 ---
    print(f"开始使用 PPO + LSTM 进行训练，目标为 {TRAIN_EPISODES} 个 episodes...")
    train_callback = EpisodeTerminationCallback(target_episodes=TRAIN_EPISODES, verbose=1)

    # [核心修改] 为 PPO 模型配置 LSTM 网络
    # 通过直接传递 lstm_hidden_size 来激活 LSTM
    policy_kwargs = dict(
        lstm_hidden_size=64,  # 直接指定 LSTM 隐藏层大小来启用它
        net_arch=dict(
            pi=[64],  # 定义 LSTM 之后的策略网络 MLP 层
            vf=[64]  # 定义 LSTM 之后的价值网络 MLP 层
        )
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,  # 传入 LSTM 配置
        n_steps=8192,
        verbose=0,
        tensorboard_log="./ppo_lstm_tensorboard_sb3/"
    )

    try:
        model.learn(total_timesteps=int(1e12), callback=train_callback)

        # --- 3. 保存模型和绘制训练图表 ---
        print(f"\n训练完成。正在保存模型至 {MODEL_PATH}...")
        model.save(MODEL_PATH)

        print("正在绘制训练奖励图表...")
        plot_rewards(
            train_callback.episode_rewards,
            f"PPO+LSTM Training Rewards ({len(train_callback.episode_rewards)} Episodes)",
            PLOT_PATH
        )

    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        print("请确保 Go 模拟器正在运行。")
    finally:
        # --- 4. 清理 ---
        print("\n流程结束，关闭环境。")
        env.close()


if __name__ == '__main__':
    main()