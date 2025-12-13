# train_attention_mlp_ppo.py
import torch
import numpy as np
import os
import collections
from go_simulator_env import GoSimulatorEnv
from ppo_attention_mlp_agent import PPOAttentionMLPAgent  # [关键] 导入新的、不含RNN的Agent
import matplotlib.pyplot as plt

# --- 超参数设置 (与之前的实验保持一致以进行公平比较) ---
NUM_EPISODES = 30
UPDATE_TIMESTEP = 8192
SEQUENCE_LENGTH = 10
STATE_DIM = 7
ACTION_DIM = 2
HIDDEN_DIM = 128
LR_ACTOR_CRITIC = 3e-5
GAMMA = 0.99
LAMBDA_GAE = 0.95
EPS_CLIP = 0.2
K_EPOCHS = 10

# [修改] 为这个新模型设置独立的文件名
MODEL_SAVE_PATH = "attention_mlp_ppo_model.pth"
PLOT_SAVE_PATH = "training_rewards_attention_mlp_ppo.png"


def train():
    """
    训练使用 Attention-MLP 网络的 PPO 智能体。
    """
    env = GoSimulatorEnv(sequence_length=SEQUENCE_LENGTH)

    # 实例化新的 Attention-MLP Agent
    agent = PPOAttentionMLPAgent(
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
    scheduler = torch.optim.lr_scheduler.StepLR(agent.optimizer, step_size=10, gamma=0.9)

    # 如果存在已保存的模型，可以加载继续训练
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"--- 发现已存在的模型 {MODEL_SAVE_PATH}，加载权重继续训练。 ---")
        agent.load_model(MODEL_SAVE_PATH)

    total_rewards = []
    avg_rewards_window = collections.deque(maxlen=100)
    time_step = 0

    print("开始训练 Attention-MLP-PPO 智能体 (无RNN)...")

    for episode in range(1, NUM_EPISODES + 1):
        current_obs_history = env.reset()
        episode_reward = 0
        done = False

        while not done:
            time_step += 1

            # 1. 选择动作
            # 注意：即使网络没有RNN，输入仍然是状态历史序列，因为注意力机制需要它
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
        scheduler.step()
        print(f"Episode {episode} 结束, 总奖励: {episode_reward:.2f}, 平均奖励 (最近100轮): {avg_reward:.2f}")

        # 每隔一定 episode 保存模型
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
    plt.title('Attention-MLP-PPO (No RNN) Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_SAVE_PATH)
    print(f"奖励曲线已保存到 {PLOT_SAVE_PATH}")


if __name__ == '__main__':
    # 确保您已经创建了 ppo_attention_mlp_agent.py 文件
    train()