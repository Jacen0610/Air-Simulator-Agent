# train_mlp_dqn.py
import torch
import numpy as np
import os
import collections
from go_simulator_env import GoSimulatorEnv
from dqn_mlp_agent import DQNAgent  # [关键] 从分离的文件中导入 Agent
import matplotlib.pyplot as plt

# --- 超参数设置 ---
NUM_EPISODES = 500
SEQUENCE_LENGTH = 10  # 环境需要，但Agent只使用最后一个状态
STATE_DIM = 7
ACTION_DIM = 2
HIDDEN_DIM = 128  # DQN通常使用比PPO/SAC稍小或同等大小的网络

# DQN 特有超参数
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0  # 初始探索率
EPSILON_END = 0.05  # 最终探索率
EPSILON_DECAY = 0.999  # 探索率衰减因子 (每次 update 时衰减)
BUFFER_SIZE = 50000
BATCH_SIZE = 128
TARGET_UPDATE_FREQ = 1000  # 每隔 N 次更新，同步目标网络
LEARNING_STARTS = 1000  # 在收集到 N 步经验后才开始学习

MODEL_SAVE_PATH = "mlp_dqn_model.pth"
PLOT_SAVE_PATH = "training_rewards_mlp_dqn.png"


def train():
    """
    训练使用 MLP 网络的 DQN 智能体。
    """
    env = GoSimulatorEnv(sequence_length=SEQUENCE_LENGTH)

    agent = DQNAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ
    )

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"--- 发现已存在的模型 {MODEL_SAVE_PATH}，加载权重继续训练。 ---")
        agent.load_model(MODEL_SAVE_PATH)

    total_rewards = []
    avg_rewards_window = collections.deque(maxlen=100)
    total_steps = 0

    print("开始训练 MLP-DQN 智能体...")

    for episode in range(1, NUM_EPISODES + 1):
        obs_history = env.reset()
        current_state = obs_history[-1]  # 只取当前时刻的状态
        episode_reward = 0
        done = False

        while not done:
            # 使用 epsilon-greedy 策略选择动作
            action = agent.select_action(current_state)

            next_obs_history, reward, done, _ = env.step(action)
            next_state = next_obs_history[-1]

            # 存储经验
            agent.store_transition(current_state, action, reward, next_state, done)

            total_steps += 1

            # 在收集到足够的数据后，每一步都进行学习
            if total_steps > LEARNING_STARTS:
                agent.update()

            current_state = next_state
            episode_reward += reward

        total_rewards.append(episode_reward)
        avg_rewards_window.append(episode_reward)
        avg_reward = np.mean(avg_rewards_window)

        print(
            f"Episode {episode} | Total Steps: {total_steps} | Reward: {episode_reward:.2f} | Avg Reward (100): {avg_reward:.2f} | Epsilon: {agent.epsilon:.4f}")

        if episode % 50 == 0:
            print(f"--- Episode {episode}，保存模型到 {MODEL_SAVE_PATH} ---")
            agent.save_model(MODEL_SAVE_PATH)

    env.close()
    print("训练完成！")
    agent.save_model(MODEL_SAVE_PATH)

    # 绘制奖励曲线
    plt.figure(figsize=(12, 6))
    plt.plot(total_rewards, label='Episode Reward')
    plt.plot(np.convolve(total_rewards, np.ones(100) / 100, mode='valid'), label='Moving Average (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('MLP-DQN Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_SAVE_PATH)
    print(f"奖励曲线已保存到 {PLOT_SAVE_PATH}")


if __name__ == '__main__':
    train()