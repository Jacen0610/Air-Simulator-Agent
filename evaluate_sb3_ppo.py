import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from stable_baselines3 import PPO
import os

# [核心修改] 导入 FlattenObservation 封装器
from gymnasium.wrappers import FlattenObservation
from gym_env import GymEnv

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
        plt.text(episodes[i], reward, f' {reward:.2f}', va='center')

    plt.title(title, fontsize=16)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    
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
    MODEL_DIR = "SB3/sb3_models"
    PLOT_DIR = "SB3/sb3_plots"
    MODEL_PATH = os.path.join(MODEL_DIR, "ppo_gym_env.zip")
    PLOT_FILENAME = os.path.join(PLOT_DIR, "sb3_ppo_evaluation_rewards.png")

    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型文件 '{MODEL_PATH}'。")
        print("请先运行 train_sb3_ppo.py 脚本来训练并保存一个模型。")
        return

    os.makedirs(PLOT_DIR, exist_ok=True)

    # --- 1. 创建环境并加载模型 ---
    print("正在初始化 Gym 环境...")
    # [核心修改] 必须应用与训练时完全相同的封装器
    env = GymEnv()
    env = FlattenObservation(env)

    print(f"正在从 {MODEL_PATH} 加载已训练的模型...")
    try:
        model = PPO.load(MODEL_PATH, env=env)
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        env.close()
        return

    # --- 2. 运行评估循环 ---
    print(f"\n开始评估模型，共 {EVAL_EPISODES} 个 episodes...")
    
    eval_rewards = []
    for i in range(EVAL_EPISODES):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        while not done:
            # 使用确定性动作进行评估，以获得模型最稳定的表现
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
        
        eval_rewards.append(episode_reward)
        print(f"评估 Episode {i + 1}/{EVAL_EPISODES} | Reward: {episode_reward:.2f} | Steps: {step_count}")

    # --- 3. 绘制并保存奖励图表 ---
    print("\n评估完成。正在绘制奖励图表...")
    plot_evaluation_rewards(
        eval_rewards,
        f"SB3 PPO Model Evaluation Rewards ({len(eval_rewards)} Episodes)",
        PLOT_FILENAME
    )

    # --- 4. 清理 ---
    print("\n流程结束，关闭环境。")
    env.close()

if __name__ == '__main__':
    main()
