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
    TOTAL_TRAINING_EPISODES = 1000
    SEQUENCE_LENGTH = 10
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
        verbose=1,
        learning_rate=3e-4,
        gamma=0.999,      # 重要：针对 90s 长周期，Gamma 必须大
        n_steps=2048,     # 每次更新采集的步数
        batch_size=64,
        ent_coef=0.01,    # 增加探索，防止 Agent 变“胆小”
        device="cuda",
        tensorboard_log=TENSORBOARD_LOG_DIR
    )
    
    # --- 3. 训练模型 ---
    train_callback = RewardAndEpisodeCallback(total_episodes=TOTAL_TRAINING_EPISODES, verbose=1)
    
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
