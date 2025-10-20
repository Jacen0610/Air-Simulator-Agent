# train_mlp_ppo.py
import torch
import numpy as np
import os
import collections
from go_simulator_env import GoSimulatorEnv
from ppo_mlp_agent import PPOMLPAgent  # 导入新的 MLP Agent
import matplotlib.pyplot as plt

# --- 超参数设置 ---
NUM_EPISODES = 40
UPDATE_TIMESTEP = 4000  # 每收集 N 步数据后进行一次策略更新
SEQUENCE_LENGTH = 10  # 环境仍然需要这个参数，但我们只用最后一个状态
STATE_DIM = 7  # AgentObservation 的维度
ACTION_DIM = 2  # 0: ACTION_WAIT, 1: ACTION_SEND

HIDDEN_DIM = 128  # MLP 隐藏层维度

# PPO 特有超参数
LR_ACTOR_CRITIC = 3e-4
GAMMA = 0.99
LAMBDA_GAE = 0.95
EPS_CLIP = 0.2
K_EPOCHS = 10

MODEL_SAVE_PATH = "mlp_ppo_model.pth"
PLOT_SAVE_PATH = "training_rewards_mlp_ppo.png"


def train():
    # 注意：环境仍然需要 sequence_length，但我们只使用返回序列的最后一个元素
    env = GoSimulatorEnv(sequence_length=SEQUENCE_LENGTH)

    # 实例化新的 MLP Agent
    agent = PPOMLPAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        lr_actor_critic=LR_ACTOR_CRITIC,
        gamma=GAMMA,
        lambda_gae=LAMBDA_GAE,
        eps_clip=EPS_CLIP,
        k_epochs=K_EPOCHS
    )

    total_rewards = []
    avg_rewards_window = collections.deque(maxlen=100)
    time_step = 0

    print("开始训练无记忆的 MLP-PPO 智能体...")

    for episode in range(1, NUM_EPISODES + 1):
        obs_history = env.reset()
        current_state = obs_history[-1]  # [关键] 只取当前时刻的状态

        episode_reward = 0
        done = False

        while not done:
            time_step += 1

            # 1. 选择动作 (只传入当前状态)
            action, log_prob, state_val = agent.select_action(current_state)

            # 2. 与环境交互
            next_obs_history, reward, done, _ = env.step(action)
            next_state = next_obs_history[-1]  # [关键] 只取下一个时刻的状态

            # 3. 存储经验 (只存储单个状态)
            agent.store_transition(current_state, action, log_prob, reward, done, state_val)

            current_state = next_state
            episode_reward += reward

            # 4. 如果收集到足够的数据，则进行更新
            if time_step % UPDATE_TIMESTEP == 0:
                print(f"  [Episode {episode}] 收集到 {UPDATE_TIMESTEP} 步数据，开始更新策略...")
                agent.update()
                time_step = 0

        total_rewards.append(episode_reward)
        avg_rewards_window.append(episode_reward)
        avg_reward = np.mean(avg_rewards_window)

        print(f"Episode {episode} 结束, 总奖励: {episode_reward:.2f}, 平均奖励 (最近100轮): {avg_reward:.2f}")

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
    plt.title('MLP-PPO (No RNN) Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_SAVE_PATH)
    print(f"奖励曲线已保存到 {PLOT_SAVE_PATH}")


if __name__ == '__main__':
    train()