import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
import os
import numpy as np

from env.gym_env import GymEnv # 导入正确的环境类名

def plot_evaluation_rewards(rewards: list, title: str, filename: str):
    """
    为评估过程绘制奖励曲线，并确保 Y 轴清晰易读。

    :param rewards: 包含每个 episode 奖励的列表。
    :param title: 图表标题。
    :param filename: 保存图表的文件路径。
    """
    if not rewards:
        print("没有可供绘制的奖励数据。")
        return

    episodes = range(1, len(rewards) + 1)
    
    plt.figure(figsize=(12, 7))
    
    # 绘制线图和散点图，确保每个点都清晰可见
    plt.plot(episodes, rewards, color='dodgerblue', linestyle='-', linewidth=2, label='Episode Reward')
    plt.scatter(episodes, rewards, color='red', zorder=5) # zorder确保点在最上层

    # 在每个点旁边标注奖励值
    for i, reward in enumerate(rewards):
        plt.text(episodes[i], reward, f' {reward:.2f}', va='center', ha='center') # 居中对齐

    plt.title(title, fontsize=16)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Original Reward", fontsize=12) # 修正Y轴标签
    
    # 设置 X 轴为整数刻度
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    # 自动调整 Y 轴以适应奖励范围，并添加网格线
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.legend()
    plt.tight_layout() # 自动调整布局，防止标签重叠
    
    plt.savefig(filename)
    print(f"评估奖励图表已保存至: {filename}")
    plt.close()

def main():
    """
    主评估流程。
    """
    # --- 配置 ---
    EVAL_EPISODES = 10
    # 修正模型和统计数据路径，与训练脚本保持一致
    MODEL_PATH = "../SB3/models/sb3_ppo.zip"
    STATS_PATH = "../SB3/models/vec_normalize.pkl"
    PLOT_DIR = "../SB3/plots/eval" # 评估图表保存目录
    
    # 确保保存目录存在
    os.makedirs(PLOT_DIR, exist_ok=True)

    # 检查模型文件和统计数据文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型文件 '{MODEL_PATH}'。")
        print("请先运行 train_sb3_ppo.py 脚本来训练并保存一个模型。")
        return
    if not os.path.exists(STATS_PATH):
        print(f"错误：找不到环境统计数据文件 '{STATS_PATH}'。")
        print("请确保 train_sb3_ppo.py 脚本已运行并保存了 VecNormalize 统计数据。")
        return

    # --- 1. 创建环境并加载统计数据 ---
    print("正在初始化 Gym 环境并加载归一化统计数据...")
    # 创建基础环境 (不需要 FlattenObservation)
    eval_env = make_vec_env(lambda: GymEnv(grpc_server_address='localhost:50050'), n_envs=1)
    
    # 加载 VecNormalize 统计数据并包装环境
    env = VecNormalize.load(STATS_PATH, eval_env)
    # 设置为评估模式：不更新统计数据，并返回原始奖励
    env.training = False
    env.norm_reward = False
    
    # --- 2. 加载训练好的模型 ---
    print(f"正在从 {MODEL_PATH} 加载已训练的模型...")
    try:
        # env=env 是必要的，因为模型需要知道它所训练的环境的结构
        model = PPO.load(MODEL_PATH, env=env)
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        env.close()
        return

    # --- 3. 运行评估循环 ---
    print(f"\n开始评估模型，共 {EVAL_EPISODES} 个 episodes...")
    
    eval_rewards = []
    for i in range(EVAL_EPISODES):
        obs = env.reset() # env.reset() 返回的是归一化后的观测
        done = False
        episode_reward = 0.0 # 使用浮点数
        step_count = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward[0] # VecNormalize 返回的 reward 是一个数组，取第一个元素
            step_count += 1
        
        eval_rewards.append(episode_reward)
        print(f"评估 Episode {i + 1}/{EVAL_EPISODES} | Reward: {episode_reward:.2f} | Steps: {step_count}")

    # --- 4. 绘制并保存奖励图表 ---
    print("\n评估完成。正在绘制奖励图表...")

    PLOT_FILENAME = os.path.join(PLOT_DIR, f"sb3_ppo_evaluation_rewards.png")
    plot_evaluation_rewards(
        eval_rewards,
        f"SB3 PPO Model Evaluation Rewards (Avg: {np.mean(eval_rewards):.2f})",
        PLOT_FILENAME
    )

    # --- 5. 清理 ---
    print("\n流程结束，关闭环境。")
    env.close()

if __name__ == '__main__':
    main()
