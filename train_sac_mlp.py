# train_mlp_sac.py
import torch
import numpy as np
import os
import collections
from go_simulator_env import GoSimulatorEnv
from sac_mlp_agent import SACMLPAgent  # [关键] 从分离的文件中导入 Agent
import matplotlib.pyplot as plt

# --- 超参数设置 ---
NUM_EPISODES = 50
SEQUENCE_LENGTH = 10
STATE_DIM = 7
ACTION_DIM = 2
HIDDEN_DIM = 256  # SAC 通常从稍大的网络中受益

# SAC 特有超参数
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
LR_ALPHA = 3e-4
GAMMA = 0.99
TAU = 0.005  # 目标网络软更新系数
BUFFER_SIZE = 100_000
BATCH_SIZE = 256
LEARNING_STARTS = 1000  # 在收集到 N 步经验后才开始学习

MODEL_SAVE_PATH = "mlp_sac_model.pth"
PLOT_SAVE_PATH = "training_rewards_mlp_sac.png"


def train():
    """
    训练使用 MLP 网络的 SAC 智能体。
    """
    env = GoSimulatorEnv(sequence_length=SEQUENCE_LENGTH)

    agent = SACMLPAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        lr_alpha=LR_ALPHA,
        gamma=GAMMA,
        tau=TAU,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE
    )

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"--- 发现已存在的模型 {MODEL_SAVE_PATH}，加载权重继续训练。 ---")
        agent.load_model(MODEL_SAVE_PATH)

    total_rewards = []
    avg_rewards_window = collections.deque(maxlen=100)
    total_steps = 0

    print("开始训练 MLP-SAC 智能体...")

    for episode in range(1, NUM_EPISODES + 1):
        obs_history = env.reset()
        current_state = obs_history[-1]  # [关键] 只取当前时刻的状态
        episode_reward = 0
        done = False

        while not done:
            # 在学习开始前，随机选择动作以填充缓冲区
            if total_steps < LEARNING_STARTS:
                action = np.random.randint(ACTION_DIM)
            else:
                action = agent.select_action(current_state)

            next_obs_history, reward, done, _ = env.step(action)
            next_state = next_obs_history[-1]  # [关键] 只取下一个时刻的状态

            # 存储经验 (只存储单个状态)
            agent.store_transition(current_state, action, reward, next_state, done)

            # 在收集到足够的数据后，每一步都进行学习
            if total_steps > LEARNING_STARTS:
                agent.update()

            current_state = next_state
            episode_reward += reward
            total_steps += 1

        total_rewards.append(episode_reward)
        avg_rewards_window.append(episode_reward)
        avg_reward = np.mean(avg_rewards_window)

        print(
            f"Episode {episode} | Total Steps: {total_steps} | Reward: {episode_reward:.2f} | Avg Reward (100): {avg_reward:.2f} | Alpha: {agent.alpha.item():.4f}")

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
    plt.title('MLP-SAC Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_SAVE_PATH)
    print(f"奖励曲线已保存到 {PLOT_SAVE_PATH}")


if __name__ == '__main__':
    # 确保 go_simulator_env.py 和 sac_mlp_agent.py 文件在同一目录下
    train()