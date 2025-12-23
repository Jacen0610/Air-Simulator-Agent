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

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import time

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
    
    # --- 使用基于项目根目录的绝对路径 ---
    # 注意：这里的文件名需要与 train_sb3_ppo.py 中保存的文件名一致
    MODEL_PATH = os.path.join(project_root, "SB3/models/sb3_mlp_ppo.zip")
    STATS_PATH = os.path.join(project_root, "SB3/models/sb3_mlp_vec_normalize.pkl")
    PLOT_DIR = os.path.join(project_root, "SB3/plots/eval") # 评估图表保存目录
    
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
    # 创建基础环境，并禁用 Monitor 的自动重置功能
    eval_env = make_vec_env(
        lambda: GymEnv(grpc_server_address='localhost:50051'),
        n_envs=1,
        monitor_kwargs={"auto_reset": False} # 禁用 Monitor 的自动重置
    )
    
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
        # 显式地重置环境，确保每个 episode 都从一个 reset 开始
        obs = env.reset() 
        episode_reward = 0.0
        step_count = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            # VecEnv 的 step 返回 4 个值: obs, reward, done, info
            obs, reward, done, info = env.step(action)
            
            # is_done 标志现在直接来自 env.step() 的 done 数组
            is_done = done[0]
            step_reward = reward[0]
            
            episode_reward += step_reward
            step_count += 1
            
            if is_done:
                # 当 is_done 为 True 时，表示当前 episode 结束
                # 由于 Monitor 的 auto_reset=False，底层环境不会自动重置
                # 我们将记录这个 episode 的奖励，并在下一个 for 循环中显式重置
                break
        
        # 记录当前 episode 的总奖励
        # 注意：这里 episode_reward 累加的是 VecNormalize 缩放后的奖励
        # 但由于 Monitor 的 auto_reset=False，info['episode'] 仍然会包含原始奖励
        if 'episode' in info[0]:
            original_episode_reward = info[0]['episode']['r']
            eval_rewards.append(original_episode_reward)
            print(f"评估 Episode {i + 1}/{EVAL_EPISODES} | Reward: {original_episode_reward:.2f} | Steps: {info[0]['episode']['l']}")
        else:
            # 如果没有 'episode' 信息，说明可能在 episode 结束前循环中断，或者 Monitor 配置有问题
            print(f"Warning: Episode {i + 1} finished but 'episode' info not found. Using accumulated scaled reward: {episode_reward:.2f}")
            eval_rewards.append(episode_reward) # 作为备用，记录缩放后的奖励

    # --- 4. 绘制并保存奖励图表 ---
    print("\n评估完成。正在绘制奖励图表...")
    # 使用时间戳确保文件名唯一
    timestamp = int(time.time())
    PLOT_FILENAME = os.path.join(PLOT_DIR, f"sb3_ppo_evaluation_rewards_{timestamp}.png")
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
