import sys
import os
import argparse # 导入 argparse

# --- 动态添加项目根目录到 sys.path ---
# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 假设项目根目录是脚本所在目录的父目录的父目录 (Air-Simulator-Agent/)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
# 将项目根目录添加到 sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from env.gym_env import GymEnv # 导入正确的环境类名

# --- 自定义回调：记录奖励并按 Episode 数量停止训练 ---
class RewardAndEpisodeCallback(BaseCallback):
    def __init__(self, total_episodes: int, verbose=0):
        super(RewardAndEpisodeCallback, self).__init__(verbose)
        self.total_episodes = total_episodes
        self.episode_count = 0
        self.episode_rewards = []
        self.current_reward = 0.0

    def _on_step(self) -> bool:
        self.current_reward += self.locals['rewards'][0]
        
        if any(self.locals.get("dones", [])):
            self.episode_count += 1
            self.episode_rewards.append(self.current_reward)
            if self.verbose > 0:
                print(f"Episode {self.episode_count}/{self.total_episodes} finished. Normalized Reward: {self.current_reward}")
            self.current_reward = 0.0
        
        if self.episode_count >= self.total_episodes:
            print(f"Reached {self.total_episodes} episodes. Stopping training.")
            return False
        return True

# --- 绘制奖励曲线图的函数 ---
def plot_rewards(rewards, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(rewards) + 1), rewards)
    plt.xlabel("Episode Number")
    plt.ylabel("Normalized Episode Reward")
    plt.title("SB3 MLP PPO Normalized Reward Curve")
    
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_major_formatter().set_useOffset(False)
    
    plt.grid(True)
    plt.savefig(filename)
    print(f"奖励曲线图已保存至: {filename}")

# --- 主训练函数 ---
def main():
    # --- 解析命令行参数 ---
    parser = argparse.ArgumentParser(description='Train SB3 PPO Agent')
    parser.add_argument('--grpc_port', type=str, default='50051', help='gRPC server port (default: 50051)')
    args = parser.parse_args()
    grpc_address = f'localhost:{args.grpc_port}'
    print(f"Using gRPC server address: {grpc_address}")

    # --- 训练设置 ---
    TOTAL_TRAINING_EPISODES = 50 
    TENSORBOARD_LOG_DIR = os.path.join(project_root, "sb3_logs/mlp_ppo/") # 使用绝对路径
    
    # --- 核心改动：创建并包装环境以进行归一化 ---
    vec_env = make_vec_env(lambda: GymEnv(grpc_server_address=grpc_address), n_envs=1)
    env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, gamma=0.99)

    # 创建回调实例
    reward_callback = RewardAndEpisodeCallback(total_episodes=TOTAL_TRAINING_EPISODES, verbose=1)
    
    # 定义PPO模型，并加入稳定性调整
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=16384,
        verbose=0,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        learning_rate=3e-5,      
        max_grad_norm=0.5        
    )
    
    # 训练模型
    try:
        model.learn(
            total_timesteps=int(1e9),
            callback=reward_callback,
            tb_log_name="PPO_GymEnv_Normalized"
        )
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
    finally:
        # --- 训练后操作 ---
        
        # 1. 保存模型 (使用绝对路径)
        model_save_path = os.path.join(project_root, "SB3/models/sb3_mlp_ppo.zip")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True) # 确保目录存在
        model.save(model_save_path)
        print(f"模型已保存至: {model_save_path}")

        # 2. 保存 VecNormalize 的统计数据 (使用绝对路径)
        stats_path = os.path.join(project_root, "SB3/models/sb3_mlp_vec_normalize.pkl")
        env.save(stats_path)
        print(f"环境统计数据已保存至: {stats_path}")
        
        # 3. 生成并保存奖励曲线图 (使用绝对路径)
        if reward_callback.episode_rewards:
            reward_plot_path = os.path.join(project_root, "SB3/plots/train/sb3_mlp_ppo.png")
            os.makedirs(os.path.dirname(reward_plot_path), exist_ok=True) # 确保目录存在
            plot_rewards(reward_callback.episode_rewards, reward_plot_path)
        else:
            print("没有足够的奖励数据来生成图表。")

        # 关闭环境
        env.close()
        print("环境已关闭。")

    print("训练完成。")

# --- 脚本入口 ---
if __name__ == '__main__':
    main()
