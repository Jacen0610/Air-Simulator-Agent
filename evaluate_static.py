# evaluate_static.py
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from go_simulator_env import GoSimulatorEnv
from static_agent import CSMAAgent

# --- 配置 ---
NUM_EVAL_EPISODES = 10
SEQUENCE_LENGTH = 10
P_VALUE = 0.05
SLOT_TIME_SECONDS = 0.0045

# [新增] 绘图配置
PLOT_DIR = "SB3/sb3_plots" # 复用现有的绘图目录
PLOT_FILENAME = os.path.join(PLOT_DIR, "evaluation_rewards_static_agent.png")

# --- [新增] 绘图函数 ---
def plot_evaluation_rewards(rewards: list, title: str, filename: str):
    """
    为评估过程绘制奖励曲线。
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
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.savefig(filename)
    print(f"\n评估奖励图表已保存至: {filename}")
    plt.close()

# --- 评估逻辑 ---
def evaluate():
    """
    加载静态CSMA智能体，运行多个episode，并记录和绘制性能。
    """
    env = GoSimulatorEnv(sequence_length=SEQUENCE_LENGTH)
    agent = CSMAAgent(p_value=P_VALUE)

    eval_rewards = [] # [修改] 使用更明确的变量名

    print(f"开始评估 P={P_VALUE} 的静态 CSMA 策略，时隙时间: {SLOT_TIME_SECONDS * 1000}ms...")

    for episode in range(1, NUM_EVAL_EPISODES + 1):
        obs_history = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done:
            action = agent.select_action(obs_history)
            next_obs_history, reward, done, _ = env.step(action)

            obs_history = next_obs_history
            episode_reward += reward
            episode_steps += 1

            time.sleep(SLOT_TIME_SECONDS)

        eval_rewards.append(episode_reward) # [修改] 记录每轮得分
        print(f"Episode {episode}/{NUM_EVAL_EPISODES} | Total Reward: {episode_reward:.2f} | Steps: {episode_steps}")

    env.close()

    # --- 结果分析与输出 ---
    avg_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)

    print("\n" + "=" * 50)
    print(f"  Static CSMA (p={P_VALUE}) Evaluation Results")
    print("=" * 50)
    print(f"  Number of Episodes: {NUM_EVAL_EPISODES}")
    print(f"  Slot Time: {SLOT_TIME_SECONDS * 1000} ms")
    print(f"  Average Total Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print("=" * 50)

    # --- [新增] 调用绘图函数 ---
    plot_evaluation_rewards(
        eval_rewards,
        f"Static CSMA (p={P_VALUE}) Evaluation Rewards ({len(eval_rewards)} Episodes)",
        PLOT_FILENAME
    )


if __name__ == '__main__':
    evaluate()
