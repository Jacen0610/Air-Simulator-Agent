# train_curriculum_ppo.py
import torch
import numpy as np
import os
import collections
import argparse  # [新] 导入命令行参数库
from go_simulator_env import GoSimulatorEnv
from ppo_attention_agent import PPOAttentionAgent
import matplotlib.pyplot as plt

# --- 超参数设置 ---
# [修改] 将 NUM_EPISODES 分配到不同阶段
PHASE1_EPISODES = 20  # 简单环境下的训练轮数
PHASE2_EPISODES = 50  # 困难环境下的微调轮数

UPDATE_TIMESTEP = 8192  # PPO 更新步数
SEQUENCE_LENGTH = 10
STATE_DIM = 7
ACTION_DIM = 2
HIDDEN_DIM = 128

# PPO 特有超参数
LR_ACTOR_CRITIC = 3e-4  # [建议] 使用我们之前讨论的较小学习率
GAMMA = 0.99
LAMBDA_GAE = 0.95
EPS_CLIP = 0.2
K_EPOCHS = 10

# [修改] 为不同阶段定义不同的模型和图表路径
BASE_MODEL_NAME = "attention_ppo_curriculum"


def get_paths(phase):
    """根据阶段生成模型和图表的保存路径"""
    model_path = f"{BASE_MODEL_NAME}_phase{phase}.pth"
    plot_path = f"training_rewards_{BASE_MODEL_NAME}_phase{phase}.png"
    return model_path, plot_path


def train(phase, total_episodes, load_from_phase=None):
    """
    执行一个训练阶段。

    :param phase: 当前阶段 (1 或 2)
    :param total_episodes: 当前阶段要运行的总轮数
    :param load_from_phase: (可选) 从哪个阶段加载模型权重
    """
    print(f"=============== STARTING PHASE {phase} ===============")

    # 1. 设置环境和智能体
    env = GoSimulatorEnv(sequence_length=SEQUENCE_LENGTH)
    agent = PPOAttentionAgent(
        state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM,
        sequence_length=SEQUENCE_LENGTH, lr_actor_critic=LR_ACTOR_CRITIC,
        gamma=GAMMA, lambda_gae=LAMBDA_GAE, eps_clip=EPS_CLIP, k_epochs=K_EPOCHS
    )

    # 2. 加载模型（如果需要）
    if load_from_phase is not None:
        load_model_path, _ = get_paths(load_from_phase)
        if os.path.exists(load_model_path):
            agent.load_model(load_model_path)
            print(f"--- Successfully loaded model from: {load_model_path} ---")
        else:
            print(f"--- WARNING: Model file not found at {load_model_path}. Starting from scratch. ---")

    # 3. 获取当前阶段的保存路径
    model_save_path, plot_save_path = get_paths(phase)

    # 4. 训练循环
    total_rewards = []
    avg_rewards_window = collections.deque(maxlen=100)
    time_step = 0

    print(f"--- Training for {total_episodes} episodes in Phase {phase}... ---")

    for episode in range(1, total_episodes + 1):
        current_obs_history = env.reset()
        episode_reward = 0
        done = False

        while not done:
            time_step += 1
            action, log_prob, state_val = agent.select_action(current_obs_history)
            next_obs_history, reward, done, _ = env.step(action)
            agent.store_transition(current_obs_history, action, log_prob, reward, done, state_val)
            current_obs_history = next_obs_history
            episode_reward += reward

            if time_step % UPDATE_TIMESTEP == 0:
                agent.update()
                time_step = 0

        total_rewards.append(episode_reward)
        avg_rewards_window.append(episode_reward)
        avg_reward = np.mean(avg_rewards_window)

        print(
            f"Phase {phase} | Episode {episode}/{total_episodes} | Reward: {episode_reward:.2f} | Avg Reward (100): {avg_reward:.2f}")

        if episode % 50 == 0:
            print(f"--- Saving model at episode {episode} to {model_save_path} ---")
            agent.save_model(model_save_path)

    # 5. 结束和收尾
    env.close()
    print(f"=============== PHASE {phase} FINISHED ===============")
    agent.save_model(model_save_path)
    print(f"Final model for Phase {phase} saved to {model_save_path}")

    # 绘制奖励曲线
    plt.figure(figsize=(12, 6))
    plt.plot(total_rewards, label=f'Phase {phase} Episode Reward')
    plt.plot(np.convolve(total_rewards, np.ones(50) / 50, mode='valid'), label=f'Moving Average (50 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Attention RNN-PPO Training Progress - Phase {phase}')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_save_path)
    print(f"Reward curve for Phase {phase} saved to {plot_save_path}")


if __name__ == '__main__':
    # [新] 使用 argparse 来解析命令行参数
    parser = argparse.ArgumentParser(description="Run Curriculum Learning for PPO Agent.")
    parser.add_argument('--phase', type=int, choices=[1, 2], required=True,
                        help="Specify the training phase to run (1 or 2).")
    args = parser.parse_args()

    if args.phase == 1:
        # --- 运行第一阶段 ---
        print("=" * 60)
        print("  IMPORTANT: Make sure the Go simulator is running in 'EASY MODE'")
        print("  (with low-density background traffic) before proceeding.")
        print("=" * 60)
        input("Press Enter to start Phase 1 training...")
        train(phase=1, total_episodes=PHASE1_EPISODES, load_from_phase=None)

    elif args.phase == 2:
        # --- 运行第二阶段 ---
        print("=" * 60)
        print("  IMPORTANT: Make sure the Go simulator is running in 'HARD MODE'")
        print("  (with high-density background traffic) before proceeding.")
        print("=" * 60)
        input("Press Enter to start Phase 2 training...")
        train(phase=2, total_episodes=PHASE2_EPISODES, load_from_phase=1)