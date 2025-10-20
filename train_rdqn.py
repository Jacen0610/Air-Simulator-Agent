# train_dqn.py
import torch
import numpy as np
import os
import collections  # <--- 添加这一行来修复错误
from go_simulator_env import GoSimulatorEnv
from rdqn_agent import RNNDQNAgent
import matplotlib.pyplot as plt

# --- 超参数设置 ---
NUM_EPISODES = 50
MAX_STEPS_PER_EPISODE = 5000000  # 每个 episode 的最大步数，防止无限循环
SEQUENCE_LENGTH = 10  # RNN 观测历史长度
STATE_DIM = 7  # AgentObservation 的维度
ACTION_DIM = 2  # 0: ACTION_WAIT, 1: ACTION_SEND

HIDDEN_DIM = 128  # RNN 隐藏层维度

LEARNING_RATE = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.998  # 每 episode 衰减
BUFFER_SIZE = 50000
BATCH_SIZE = 128
TARGET_UPDATE_FREQ = 200  # 每隔多少次学习更新目标网络

MODEL_SAVE_PATH = "rnn_dqn_model.pth"
PLOT_SAVE_PATH = "training_rewards.png"


def train():
    env = GoSimulatorEnv(sequence_length=SEQUENCE_LENGTH)
    agent = RNNDQNAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        sequence_length=SEQUENCE_LENGTH,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ
    )

    # 如果存在已保存的模型，可以加载继续训练
    # if os.path.exists(MODEL_SAVE_PATH):
    #     agent.load_model(MODEL_SAVE_PATH)
    #     print("加载现有模型继续训练。")

    total_rewards = []
    avg_rewards_window = collections.deque(maxlen=100)  # 记录最近100个 episode 的平均奖励

    print("开始训练 RNN-DQN 智能体...")

    for episode in range(NUM_EPISODES):
        # env.reset() 返回的是 (sequence_length, state_dim) 形状的初始观测历史
        current_obs_history = env.reset()
        episode_reward = 0
        done = False
        step_count = 0

        while not done and step_count < MAX_STEPS_PER_EPISODE:
            # agent.select_action 接收 (sequence_length, state_dim) 形状的观测历史
            action = agent.select_action(current_obs_history)

            # env.step() 返回的是 (sequence_length, state_dim) 形状的下一个观测历史
            next_obs_history, reward, done, _ = env.step(action)

            # 存储经验到回放缓冲区
            agent.store_transition(current_obs_history, action, reward, next_obs_history, done)

            # 学习
            if len(agent.replay_buffer) > agent.batch_size:
                agent.learn()

            current_obs_history = next_obs_history
            episode_reward += reward
            step_count += 1

        total_rewards.append(episode_reward)
        avg_rewards_window.append(episode_reward)
        avg_reward = np.mean(avg_rewards_window)

        print(f"Episode {episode + 1}/{NUM_EPISODES}, "
              f"Reward: {episode_reward:.2f}, "
              f"Avg Reward (100 episodes): {avg_reward:.2f}, "
              f"Epsilon: {agent.epsilon:.4f}, "
              f"Steps: {step_count}")

        # 每隔一定 episode 保存模型
        if (episode + 1) % 20 == 0:
            agent.save_model(MODEL_SAVE_PATH)

    env.close()
    print("训练完成！")
    agent.save_model(MODEL_SAVE_PATH)  # 训练结束后保存最终模型

    # 绘制奖励曲线
    plt.figure(figsize=(12, 6))
    plt.plot(total_rewards, label='Episode Reward')
    plt.plot(np.convolve(total_rewards, np.ones(100) / 100, mode='valid'), label='Moving Average (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('RNN-DQN Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_SAVE_PATH)
    print(f"奖励曲线已保存到 {PLOT_SAVE_PATH}")


if __name__ == '__main__':
    train()