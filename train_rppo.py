# train_ppo.py
import torch
import numpy as np
import os
import collections
from go_simulator_env import GoSimulatorEnv
from rppo_agent import PPOAgent
import matplotlib.pyplot as plt

# --- 超参数设置 ---
NUM_EPISODES = 1000
UPDATE_TIMESTEP = 4000  # 每收集 N 步数据后进行一次策略更新
SEQUENCE_LENGTH = 10  # RNN 观测历史长度
STATE_DIM = 7  # AgentObservation 的维度
ACTION_DIM = 2  # 0: ACTION_WAIT, 1: ACTION_SEND

HIDDEN_DIM = 128  # RNN 隐藏层维度

# PPO 特有超参数
LR_ACTOR_CRITIC = 3e-4
GAMMA = 0.99
LAMBDA_GAE = 0.95  # GAE 的 lambda 参数
EPS_CLIP = 0.2  # PPO 裁剪范围
K_EPOCHS = 10  # 在一个 rollout 上更新的轮次

MODEL_SAVE_PATH = "rnn_ppo_model.pth"
PLOT_SAVE_PATH = "training_rewards_ppo.png"


def train():
    env = GoSimulatorEnv(sequence_length=SEQUENCE_LENGTH)
    agent = PPOAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        sequence_length=SEQUENCE_LENGTH,
        lr_actor_critic=LR_ACTOR_CRITIC,
        gamma=GAMMA,
        lambda_gae=LAMBDA_GAE,
        eps_clip=EPS_CLIP,
        k_epochs=K_EPOCHS
    )

    total_rewards = []
    avg_rewards_window = collections.deque(maxlen=100)
    time_step = 0

    print("开始训练 RNN-PPO 智能体...")

    for episode in range(1, NUM_EPISODES + 1):
        current_obs_history = env.reset()
        episode_reward = 0
        done = False

        # PPO 的 episode 循环
        while not done:
            time_step += 1

            # 1. 选择动作
            action, log_prob, state_val = agent.select_action(current_obs_history)

            # 2. 与环境交互
            next_obs_history, reward, done, _ = env.step(action)

            # 3. 存储经验
            agent.store_transition(current_obs_history, action, log_prob, reward, done, state_val)

            current_obs_history = next_obs_history
            episode_reward += reward

            # 4. 如果收集到足够的数据，则进行更新
            if time_step % UPDATE_TIMESTEP == 0:
                agent.update()
                time_step = 0  # 重置步数计数器

        total_rewards.append(episode_reward)
        avg_rewards_window.append(episode_reward)
        avg_reward = np.mean(avg_rewards_window)

        print(f"Episode {episode}, Reward: {episode_reward:.2f}, Avg Reward (100 episodes): {avg_reward:.2f}")

        # 每隔一定 episode 保存模型
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
    plt.title('RNN-PPO Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_SAVE_PATH)
    print(f"奖励曲线已保存到 {PLOT_SAVE_PATH}")


if __name__ == '__main__':
    train()