# evaluate_sb3_ppo_lstm.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# [核心修改] 导入 RecurrentPPO 和对应的 LSTM 环境
from sb3_contrib import RecurrentPPO
from env.gym_env_for_lstm import GymEnvForLSTM

def plot_evaluation_rewards(rewards: list, title: str, filename: str):
    """
    为评估过程绘制奖励曲线。
    (此函数与 evaluate_sb3_ppo.py 中的版本完全相同)
    """
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

def main():
    """
    主评估流程 - 专门用于 RecurrentPPO (LSTM) 模型。
    """
    # --- 配置 ---
    EVAL_EPISODES = 10
    MODEL_DIR = "SB3/sb3_models"
    PLOT_DIR = "SB3/sb3_plots"
    
    # [核心修改] 指向 LSTM 模型文件
    MODEL_PATH = os.path.join(MODEL_DIR, "recurrent_ppo_lstm_single_env.zip")
    PLOT_FILENAME = os.path.join(PLOT_DIR, "evaluation_rewards_lstm.png")

    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型文件 '{MODEL_PATH}'。")
        print("请先运行 train_sb3_ppo_lstm.py 脚本来训练并保存一个模型。")
        return

    os.makedirs(PLOT_DIR, exist_ok=True)

    # --- 1. 创建环境并加载模型 ---
    print("正在初始化为 LSTM 优化的 Gym 环境...")
    # [核心修改] 使用 GymEnvForLSTM，它提供模型所需的一维观测
    env = GymEnvForLSTM()

    print(f"正在从 {MODEL_PATH} 加载已训练的 RecurrentPPO 模型...")
    try:
        # [核心修改] 使用 RecurrentPPO.load
        model = RecurrentPPO.load(MODEL_PATH, env=env)
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        env.close()
        return

    # --- 2. 运行评估循环 ---
    print(f"\n开始评估模型，共 {EVAL_EPISODES} 个 episodes...")
    
    eval_rewards = []
    for i in range(EVAL_EPISODES):
        obs, info = env.reset()
        
        # [核心修改] RecurrentPPO 需要额外处理 lstm_states 和 episode_starts
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        
        done = False
        episode_reward = 0
        step_count = 0
        while not done:
            # [核心修改] 在 predict 调用中传递状态
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states, 
                episode_start=episode_starts,
                deterministic=True
            )
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
            # 在 episode 的后续步骤中，episode_starts 应为 False
            episode_starts = np.zeros((1,), dtype=bool)
        
        eval_rewards.append(episode_reward)
        print(f"评估 Episode {i + 1}/{EVAL_EPISODES} | Reward: {episode_reward:.2f} | Steps: {step_count}")

    # --- 3. 绘制并保存奖励图表 ---
    print("\n评估完成。正在绘制奖励图表...")
    plot_evaluation_rewards(
        eval_rewards,
        f"LSTM PPO Model Evaluation Rewards ({len(eval_rewards)} Episodes)",
        PLOT_FILENAME
    )

    # --- 4. 清理 ---
    print("\n流程结束，关闭环境。")
    env.close()

if __name__ == '__main__':
    main()
