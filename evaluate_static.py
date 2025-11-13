# evaluate_static.py
import numpy as np
import time
from go_simulator_env import GoSimulatorEnv
from static_agent import CSMAAgent  # [关键] 导入我们的静态Agent


# --- 配置 ---
NUM_EVAL_EPISODES = 10  # 您希望运行多少个循环来取平均值
SEQUENCE_LENGTH = 10
P_VALUE = 0.05  # 您想要测试的 P-坚持概率
SLOT_TIME_SECONDS = 0.0045  # [新] 定义时隙时间为 4.5ms


# --- 评估逻辑 ---
def evaluate():
    """
    加载静态CSMA智能体，运行多个episode，并记录性能。
    """
    env = GoSimulatorEnv(sequence_length=SEQUENCE_LENGTH)

    # 实例化静态 Agent
    agent = CSMAAgent(p_value=P_VALUE)

    total_rewards = []

    print(f"开始评估 P={P_VALUE} 的静态 CSMA 策略，时隙时间: {SLOT_TIME_SECONDS * 1000}ms...")

    for episode in range(1, NUM_EVAL_EPISODES + 1):
        obs_history = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done:
            # 使用静态策略选择动作
            action = agent.select_action(obs_history)

            next_obs_history, reward, done, _ = env.step(action)

            obs_history = next_obs_history
            episode_reward += reward
            episode_steps += 1

            # --- [核心修改] ---
            # 在每个决策步骤后，暂停一个时隙的时间，以模拟真实的CSMA决策频率

            time.sleep(SLOT_TIME_SECONDS)
            # --------------------

        total_rewards.append(episode_reward)
        print(f"Episode {episode}/{NUM_EVAL_EPISODES} | Total Reward: {episode_reward:.2f} | Steps: {episode_steps}")

    env.close()

    # --- 结果分析与输出 ---
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)

    print("\n" + "=" * 50)
    print(f"  Static CSMA (p={P_VALUE}) Evaluation Results")
    print("=" * 50)
    print(f"  Number of Episodes: {NUM_EVAL_EPISODES}")
    print(f"  Slot Time: {SLOT_TIME_SECONDS * 1000} ms")
    print(f"  Average Total Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print("=" * 50)


if __name__ == '__main__':
    evaluate()