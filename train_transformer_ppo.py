# train_transformer_ppo.py
import torch
import numpy as np
import os
import collections
from go_simulator_env import GoSimulatorEnv
from ppo_transformer_agent import PPOTransformerAgent  # [关键] 导入新的 Transformer Agent
import matplotlib.pyplot as plt

# --- 超参数设置 ---
NUM_EPISODES = 500
UPDATE_TIMESTEP = 8192
SEQUENCE_LENGTH = 10
STATE_DIM = 7
ACTION_DIM = 2

# --- [新] Transformer 特有超参数 ---
EMBED_DIM = 64  # 嵌入维度 (Transformer内部的工作维度)
NHEAD = 4  # 多头注意力的头数 (必须能被 EMBED_DIM 整除)
NUM_ENCODER_LAYERS = 2  # Transformer Encoder 层的数量
DIM_FEEDFORWARD = 128  # Transformer 内部前馈网络的维度

# --- PPO 超参数 ---
LR_ACTOR_CRITIC = 5e-5  # Transformer 对学习率更敏感，建议从一个更小的值开始
GAMMA = 0.99
LAMBDA_GAE = 0.95
EPS_CLIP = 0.2
K_EPOCHS = 10

MODEL_SAVE_PATH = "transformer_ppo_model.pth"
PLOT_SAVE_PATH = "training_rewards_transformer_ppo.png"


def train():
    """
    训练使用 Transformer 网络的 PPO 智能体。
    """
    env = GoSimulatorEnv(sequence_length=SEQUENCE_LENGTH)

    # 实例化新的 Transformer Agent
    agent = PPOTransformerAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        embed_dim=EMBED_DIM,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        lr_actor_critic=LR_ACTOR_CRITIC,
        gamma=GAMMA,
        lambda_gae=LAMBDA_GAE,
        eps_clip=EPS_CLIP,
        k_epochs=K_EPOCHS
    )

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"--- 发现已存在的模型 {MODEL_SAVE_PATH}，加载权重继续训练。 ---")
        agent.load_model(MODEL_SAVE_PATH)

    total_rewards = []
    avg_rewards_window = collections.deque(maxlen=100)
    time_step = 0

    print("开始训练 Transformer-PPO 智能体...")

    for episode in range(1, NUM_EPISODES + 1):
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
    plt.title('Transformer-PPO Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_SAVE_PATH)
    print(f"奖励曲线已保存到 {PLOT_SAVE_PATH}")


if __name__ == '__main__':
    train()