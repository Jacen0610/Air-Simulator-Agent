import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# --- 1. 从共享文件中导入自定义模块 ---
from custom_policy import AttentionExtractor
from gym_env import GymEnv

# --- 2. 回调函数 (保持不变) ---
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

# --- 3. 绘图函数 (保持不变) ---
def plot_rewards(rewards: list, title: str, filename: str):
    if not rewards:
        print("没有可供绘制的奖励数据。")
        return

    episodes = range(1, len(rewards) + 1)
    
    plt.figure(figsize=(15, 8))
    
    plt.plot(episodes, rewards, color='dodgerblue', linestyle='-', linewidth=1.5, alpha=0.7, label='Episode Reward')
    plt.scatter(episodes, rewards, color='red', zorder=5, s=20)

    if len(rewards) > 50:
        label_interval = len(rewards) // 25
    else:
        label_interval = 1

    for i, reward in enumerate(rewards):
        if i % label_interval == 0:
            plt.text(episodes[i], reward, f' {reward:.1f}', fontsize=9, va='center')

    plt.title(title, fontsize=16)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(filename)
    print(f"已将图表保存至: {filename}")
    plt.close()

# --- 4. 主训练流程 ---
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

    # 关键：policy_kwargs 现在使用从 custom_policy.py 导入的 AttentionExtractor
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
        tensorboard_log="./attention_ppo_pe_tensorboard_sb3/",
        device='cpu'
    )

    try:
        model.learn(total_timesteps=int(1e12), callback=train_callback)
        
        print(f"\n训练完成。正在保存模型至 {MODEL_PATH}...")
        model.save(MODEL_PATH)

        print("正在绘制训练奖励图表...")
        plot_rewards(
            train_callback.episode_rewards,
            f"Attention PPO (with PE) Training Rewards ({len(train_callback.episode_rewards)} Episodes)",
            PLOT_PATH
        )

    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
    finally:
        print("\n流程结束，关闭环境。")
        env.close()

if __name__ == '__main__':
    main()
