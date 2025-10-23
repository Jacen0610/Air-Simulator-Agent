# train_attention_ppo.py
import torch
import numpy as np
import os
import collections
from go_simulator_env import GoSimulatorEnv
from ppo_attention_agent import PPOAttentionAgent  # 导入我们修改过的 Agent
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- 超参数设置 ---
NUM_EPISODES = 50
UPDATE_TIMESTEP = 8192
SEQUENCE_LENGTH = 10
STATE_DIM = 7
ACTION_DIM = 2
HIDDEN_DIM = 128

# PPO 特有超参数
LR_ACTOR_CRITIC = 3e-4
GAMMA = 0.99
LAMBDA_GAE = 0.95
EPS_CLIP = 0.2
K_EPOCHS = 10

MODEL_SAVE_PATH = "rnn_attention_ppo_model.pth"
PLOT_SAVE_PATH = "training_rewards_attention_ppo.png"
VISUALIZATION_DIR = "training_visualizations"  # [新] 可视化结果的输出文件夹

# 状态特征的名称 (用于绘图)
STATE_FEATURE_NAMES = [
    'has_message',
    'primary_channel_busy',
    'backup_channel_busy',
    'pending_acks_count',
    'outbound_queue_length',
    'top_message_wait_time',
    'is_retransmission'
]


# [新] 绘图函数
def plot_attention_heatmap(state_history, attention_weights, step, action, trigger_reason, episode):
    """
    绘制并保存状态历史和对应的注意力权重热力图。
    """
    output_dir_episode = os.path.join(VISUALIZATION_DIR, f"episode_{episode}")
    if not os.path.exists(output_dir_episode):
        os.makedirs(output_dir_episode)

    df_states = pd.DataFrame(state_history, columns=STATE_FEATURE_NAMES)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(
        f'Attention at Ep {episode}, Step {step} (Action: {"SEND" if action == 1 else "WAIT"})\\nTrigger: {trigger_reason}',
        fontsize=16)

    sns.heatmap(df_states.T, ax=axes[0], cmap="viridis", annot=True, fmt=".2f", linewidths=.5)
    axes[0].set_title("State History (t-9 to t)")
    axes[0].set_xlabel("Time Steps")
    axes[0].set_ylabel("State Features")

    time_steps = [f't-{i}' for i in range(SEQUENCE_LENGTH - 1, -1, -1)]
    axes[1].bar(time_steps, attention_weights, color='skyblue')
    axes[1].set_title("Attention Weights")
    axes[1].set_ylabel("Weight")
    axes[1].set_ylim(0, 1.0)
    for i, w in enumerate(attention_weights):
        axes[1].text(i, w + 0.02, f'{w:.2f}', ha='center')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = os.path.join(output_dir_episode, f"attention_step_{step}_{trigger_reason.replace(' ', '_')}.png")
    plt.savefig(filename)
    print(f"--- Attention heatmap saved to {filename} ---")
    plt.close(fig)


# [新] 评估与可视化函数
def evaluate_and_plot_attention(agent, env, episode):
    """
    在指定的 episode 运行一个评估循环，并绘制关键时刻的注意力热力图。
    """
    print(f"\n--- [Starting Attention Evaluation for Episode {episode}] ---")
    # 确保加载的是最新的模型权重
    agent.load_model(MODEL_SAVE_PATH)

    obs_history = env.reset()
    done = False
    step_count = 0
    last_action = 0

    while not done:
        step_count += 1
        # [关键] 调用可以返回权重的新方法
        action, _, _, weights = agent.select_action_and_get_weights(obs_history)

        # --- 可视化触发条件 ---
        # 条件1: 当智能体在等待后决定发送 (抓住机会)
        if last_action == 0 and action == 1:
            plot_attention_heatmap(obs_history, weights, step_count, action, "Decided to SEND after WAITING", episode)

        # 条件2: 当消息等待时间首次超过一个阈值时 (情况紧急)
        wait_time = obs_history[-1][5]
        prev_wait_time = obs_history[-2][5]
        if wait_time > 5.0 and prev_wait_time <= 5.0:
            plot_attention_heatmap(obs_history, weights, step_count, action, f"High Wait Time ({wait_time:.1f}s)",
                                   episode)

        # 条件3: 当信道刚从繁忙变为空闲时 (机会窗口)
        if step_count > 1:
            channel_was_busy = obs_history[-2][1] > 0.5
            channel_is_free = obs_history[-1][1] < 0.5
            if channel_was_busy and channel_is_free:
                plot_attention_heatmap(obs_history, weights, step_count, action, "Channel Just Became Free", episode)

        next_obs_history, _, done, _ = env.step(action)
        obs_history = next_obs_history
        last_action = action

        if done:
            print(f"--- [Finished Attention Evaluation for Episode {episode}] ---\n")
            break


def train():
    env = GoSimulatorEnv(sequence_length=SEQUENCE_LENGTH)
    agent = PPOAttentionAgent(
        state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM,
        sequence_length=SEQUENCE_LENGTH, lr_actor_critic=LR_ACTOR_CRITIC,
        gamma=GAMMA, lambda_gae=LAMBDA_GAE, eps_clip=EPS_CLIP, k_epochs=K_EPOCHS
    )

    total_rewards = []
    avg_rewards_window = collections.deque(maxlen=100)
    time_step = 0

    print("开始训练带注意力机制的 RNN-PPO 智能体...")

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

        # [核心修改] 在特定轮次进行评估和可视化
        if episode == 25 or episode == NUM_EPISODES:
            # 先保存当前模型，确保评估函数加载的是最新的权重
            print(f"--- Reached milestone episode {episode}, saving model before evaluation. ---")
            agent.save_model(MODEL_SAVE_PATH)
            # 调用评估和可视化函数
            evaluate_and_plot_attention(agent, env, episode)

        # 每隔一定 episode 保存模型 (可以保留或与上面的逻辑合并)
        if episode % 20 == 0 and not (episode == 25 or episode == NUM_EPISODES):
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
    plt.title('Attention RNN-PPO Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_SAVE_PATH)
    print(f"奖励曲线已保存到 {PLOT_SAVE_PATH}")


if __name__ == '__main__':
    # 确保您已经安装了 seaborn 和 pandas: pip install seaborn pandas
    train()