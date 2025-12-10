import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

from stable_baselines3 import PPO

# --- 1. 从共享文件中导入自定义模块 ---
from custom_policy import AttentionExtractor
from gym_env import GymEnv

# --- 2. 绘图函数 (保持不变) ---
def plot_evaluation_rewards(rewards: list, title: str, filename: str):
    if not rewards:
        print("没有可供绘制的奖励数据。")
        return

    episodes = range(1, len(rewards) + 1)
    
    plt.figure(figsize=(12, 7))
    
    plt.plot(episodes, rewards, color='dodgerblue', linestyle='-', linewidth=2, label='Episode Reward')
    plt.scatter(episodes, rewards, color='red', zorder=5)

    for i, reward in enumerate(rewards):
        plt.text(episodes[i], reward, f' {reward:.2f}', va='center')

    plt.title(title, fontsize=16)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(filename)
    print(f"评估奖励图表已保存至: {filename}")
    plt.close()

# --- 3. 主评估流程 ---
def main():
    # --- 配置 ---
    EVAL_EPISODES = 10
    MODEL_DIR = "sb3_models"
    PLOT_DIR = "sb3_plots"
    MODEL_PATH = os.path.join(MODEL_DIR, "attention_ppo_model_with_pe.zip")
    PLOT_FILENAME = os.path.join(PLOT_DIR, "evaluation_rewards_transformer_ppo.png")

    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型文件 '{MODEL_PATH}'。")
        print("请先运行 train_gym_attention_ppo.py 脚本来训练并保存一个模型。")
        return

    os.makedirs(PLOT_DIR, exist_ok=True)

    # --- 1. 创建环境并加载模型 ---
    print("正在初始化 Gym 环境...")
    env = GymEnv()

    # 关键：定义与训练时完全相同的 policy_kwargs。
    # 这是确保 SB3 能够正确重建模型结构的关键。
    policy_kwargs = {
        "features_extractor_class": AttentionExtractor,
        "features_extractor_kwargs": dict(features_dim=128),
    }

    print(f"正在从 {MODEL_PATH} 加载已训练的 Transformer PPO 模型...")
    try:
        # 正确的加载方式：传入 policy_kwargs，而不是 custom_objects。
        # 这会使 SB3 在加载时使用与训练时相同的结构，从而通过检查。
        model = PPO.load(MODEL_PATH, env=env, policy_kwargs=policy_kwargs, device='cpu')
        print("模型加载成功！")
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
        f"Transformer PPO Evaluation Rewards ({len(eval_rewards)} Episodes)",
        PLOT_FILENAME
    )

    # --- 4. 清理 ---
    print("\n流程结束，关闭环境。")
    env.close()

if __name__ == '__main__':
    main()
