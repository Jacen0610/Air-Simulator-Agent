import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import FlattenObservation
import os

from gym_env import GymEnv

class EpisodeTerminationCallback(BaseCallback):
    def __init__(self, target_episodes: int, verbose: int = 0):
        super(EpisodeTerminationCallback, self).__init__(verbose)
        self.target_episodes = target_episodes
        self.episode_count = 0
        self.episode_rewards = []
        self.last_timestep = 0

    def _on_step(self) -> bool:
        # [代码优化] 对于单环境，直接检查 dones[0] 即可
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

def main():
    """
    主训练流程 (单环境 PPO)。
    """
    # --- 配置 ---
    TRAIN_EPISODES = 20 # 增加训练轮数以获得更好效果
    # [代码优化] 统一目录结构
    MODEL_DIR = "SB3/sb3_models"
    PLOT_DIR = "SB3/sb3_plots"
    MODEL_PATH = os.path.join(MODEL_DIR, "ppo_gym_env.zip")
    PLOT_PATH = os.path.join(PLOT_DIR, "sb3_ppo_training_rewards.png")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # --- 1. 创建并封装单个环境 ---
    print("正在初始化 Gym 环境 (单实例)...")
    env = GymEnv()
    env = FlattenObservation(env)
    env = Monitor(env)

    # --- 2. 训练模型 ---
    print(f"开始训练，目标为 {TRAIN_EPISODES} 个 episodes...")
    train_callback = EpisodeTerminationCallback(target_episodes=TRAIN_EPISODES, verbose=1)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=8192,
        verbose=0,
        tensorboard_log="./ppo_tensorboard_sb3/"
    )

    try:
        model.learn(total_timesteps=int(1e9), callback=train_callback)
        
        print(f"\n训练完成。正在保存模型至 {MODEL_PATH}...")
        model.save(MODEL_PATH)

        print("正在绘制训练奖励图表...")
        plot_rewards(
            train_callback.episode_rewards,
            f"SB3 PPO Training Rewards ({len(train_callback.episode_rewards)} Episodes)",
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
