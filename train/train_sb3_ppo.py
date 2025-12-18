import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
# --- 新增导入，用于环境包装和归一化 ---
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
# ------------------------------------
# --- 修正：导入正确的环境类名 GymEnv ---
from env.gym_env import GymEnv

# --- 自定义回调：记录奖励并按 Episode 数量停止训练 ---
class RewardAndEpisodeCallback(BaseCallback):
    def __init__(self, total_episodes: int, verbose=0):
        super(RewardAndEpisodeCallback, self).__init__(verbose)
        self.total_episodes = total_episodes
        self.episode_count = 0
        self.episode_rewards = []
        self.current_reward = 0.0

    def _on_step(self) -> bool:
        # 注意：由于使用了 VecNormalize，这里的奖励是已经被缩放过的
        self.current_reward += self.locals['rewards'][0]
        
        if any(self.locals.get("dones", [])):
            self.episode_count += 1
            self.episode_rewards.append(self.current_reward)
            if self.verbose > 0:
                # 打印的是归一化后的奖励
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
    plt.ylabel("Normalized Episode Reward") # Y轴标签更新为归一化奖励
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
    # --- 训练设置 ---
    TOTAL_TRAINING_EPISODES = 50 # 您可以根据需要调整总轮数
    TENSORBOARD_LOG_DIR = "./sb3_logs/"
    
    # --- 核心改动：创建并包装环境以进行归一化 ---
    # 1. 使用 make_vec_env 创建矢量化环境
    # --- 修正：使用正确的环境类名 GymEnv ---
    vec_env = make_vec_env(lambda: GymEnv(grpc_server_address='localhost:50051'), n_envs=1)
    
    # 2. 使用 VecNormalize 包装器来归一化观测值和奖励
    env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, gamma=0.99)
    # ---------------------------------------------

    # 创建回调实例
    reward_callback = RewardAndEpisodeCallback(total_episodes=TOTAL_TRAINING_EPISODES, verbose=1)
    
    # 定义PPO模型，并加入稳定性调整
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=4096,
        verbose=0,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        learning_rate=3e-5,      # 使用更低的学习率以增加稳定性
        max_grad_norm=0.5        # 加入梯度裁剪防止梯度爆炸
    )
    
    # 训练模型
    try:
        model.learn(
            total_timesteps=int(1e9),
            callback=reward_callback,
            tb_log_name="PPO_GymEnv_Normalized" # 更新日志名称以反映正确的环境
        )
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
    finally:
        # --- 训练后操作 ---
        
        # 1. 保存模型
        model_save_path = f"../SB3/models/sb3_mlp_ppo.zip"
        model.save(model_save_path)
        print(f"模型已保存至: {model_save_path}")

        # 2. !! 必须保存 VecNormalize 的统计数据 !!
        stats_path = f"../SB3/models/sb3_mlp_vec_normalize.pkl"
        env.save(stats_path)
        print(f"环境统计数据已保存至: {stats_path}")
        
        # 3. 生成并保存奖励曲线图
        if reward_callback.episode_rewards:
            reward_plot_path = f"../SB3/plots/train/sb3_mlp_ppo.png"
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
