# train_sac.py
import torch
import numpy as np
import os
import collections
from go_simulator_env import GoSimulatorEnv
from rsac_agent import SACAgent
import matplotlib.pyplot as plt

# --- 超参数设置 ---
NUM_EPISODES = 40
MAX_STEPS_PER_EPISODE = 5_000_000  # 安全网
SEQUENCE_LENGTH = 10
STATE_DIM = 7
ACTION_DIM = 2
HIDDEN_DIM = 256  # SAC 通常需要更大的网络

# SAC 特有超参数
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
LR_ALPHA = 3e-4
GAMMA = 0.99
TAU = 0.005  # 目标网络软更新系数
BUFFER_SIZE = 100_000
BATCH_SIZE = 256
LEARNING_STARTS = 1000  # 在收集到 N 步经验后才开始学习

MODEL_SAVE_PATH = "rnn_sac_model.pth"
PLOT_SAVE_PATH = "training_rewards_sac.png"


def train():
    env = GoSimulatorEnv(sequence_length=SEQUENCE_LENGTH)
    agent = SACAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        sequence_length=SEQUENCE_LENGTH,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        lr_alpha=LR_ALPHA,
        gamma=GAMMA,
        tau=TAU,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE
    )

    total_rewards = []
    avg_rewards_window = collections.deque(maxlen=100)
    total_steps = 0

    print("开始训练 RNN-SAC 智能体...")

    for episode in range(1, NUM_EPISODES + 1):
        current_obs_history = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # 在学习开始前，随机选择动作以填充缓冲区
            if total_steps < LEARNING_STARTS:
                action = np.random.randint(ACTION_DIM)
            else:
                action = agent.select_action(current_obs_history)

            next_obs_history, reward, done, _ = env.step(action)

            agent.store_transition(current_obs_history, action, reward, next_obs_history, done)

            # 在收集到足够的数据后开始学习
            if total_steps > LEARNING_STARTS:
                agent.update()

            current_obs_history = next_obs_history
            episode_reward += reward
            total_steps += 1

        total_rewards.append(episode_reward)
        avg_rewards_window.append(episode_reward)
        avg_reward = np.mean(avg_rewards_window)

        print(
            f"Episode {episode}, Reward: {episode_reward:.2f}, Avg Reward (100 episodes): {avg_reward:.2f}, Alpha: {agent.alpha.item():.4f}")

        if episode % 50 == 0:
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
    plt.title('RNN-SAC Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_SAVE_PATH)
    print(f"奖励曲线已保存到 {PLOT_SAVE_PATH}")


if __name__ == '__main__':
    train()