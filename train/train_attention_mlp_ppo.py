import sys
import os
# --- 动态添加项目根目录到 sys.path ---
# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 假设项目根目录是脚本所在目录的父目录 (Air-Simulator-Agent/)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
# 将项目根目录添加到 sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

import torch
import numpy as np
import collections
import pickle
import time # 用于生成时间戳
from torch.utils.tensorboard import SummaryWriter # 导入 SummaryWriter
from env.go_simulator_env import GoSimulatorEnv
from agent.ppo_attention_mlp_agent import PPOAttentionMLPAgent
import matplotlib.pyplot as plt

# --- 新增：用于在线计算均值和标准差的辅助类 ---
class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.var)

# --- 超参数设置 ---
NUM_EPISODES = 50
UPDATE_TIMESTEP = 4096
SEQUENCE_LENGTH = 10
ACTION_DIM = 2
HIDDEN_DIM = 64 
LR_ACTOR_CRITIC = 3e-5 
GAMMA = 0.99
LAMBDA_GAE = 0.95
EPS_CLIP = 0.2
K_EPOCHS = 10 
BATCH_SIZE = 64 
MAX_GRAD_NORM = 0.5

continueTrain = False

# --- 修正：使用绝对路径 ---
MODEL_SAVE_PATH = os.path.join(project_root, "../Pytorch/models/attention_mlp_ppo_model.pth")
PLOT_SAVE_PATH = os.path.join(project_root, "../Pytorch/plots/training_rewards_attention_mlp_ppo.png")
RMS_SAVE_PATH = os.path.join(project_root, "../Pytorch/models/attention_mlp_ppo_rms.pkl")

TENSORBOARD_LOG_DIR = os.path.join(project_root, "../TensorBoard/attention-ppo/runs", f"attention_mlp_ppo_{int(time.time())}")

def train():
    env = GoSimulatorEnv(grpc_server_address='localhost:50051', sequence_length=SEQUENCE_LENGTH) # 确保 grpc_server_address 正确
    state_dim = env.state_dim
    print(f"检测到状态维度: {state_dim}")

    obs_rms = RunningMeanStd(shape=(state_dim,))

    agent = PPOAttentionMLPAgent(
        state_dim=state_dim, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM,
        sequence_length=SEQUENCE_LENGTH, lr_actor_critic=LR_ACTOR_CRITIC,
        gamma=GAMMA, lambda_gae=LAMBDA_GAE, eps_clip=EPS_CLIP, k_epochs=K_EPOCHS,
        batch_size=BATCH_SIZE, max_grad_norm=MAX_GRAD_NORM
    )
    scheduler = torch.optim.lr_scheduler.StepLR(agent.optimizer, step_size=100, gamma=0.9)

    # --- 初始化 TensorBoard SummaryWriter ---
    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)
    print(f"TensorBoard 日志将保存到: {TENSORBOARD_LOG_DIR}")

    if os.path.exists(MODEL_SAVE_PATH) and continueTrain:
        print(f"--- 发现已存在的模型 {MODEL_SAVE_PATH}，加载权重继续训练。 ---")
        agent.load_model(MODEL_SAVE_PATH)
        if os.path.exists(RMS_SAVE_PATH):
            with open(RMS_SAVE_PATH, 'rb') as f:
                obs_rms = pickle.load(f)
            print(f"--- 同时加载归一化统计数据 {RMS_SAVE_PATH}。 ---")

    total_rewards = []
    avg_rewards_window = collections.deque(maxlen=100)
    time_step_total = 0 # 用于 TensorBoard 的全局步数

    print(f"开始训练 Attention-MLP-PPO 智能体 (带手动归一化), 共 {NUM_EPISODES} 轮, 更新频率 {UPDATE_TIMESTEP} 步")

    for episode in range(1, NUM_EPISODES + 1):
        current_obs_history_raw = env.reset()
        episode_reward = 0
        done = False
        episode_steps = 0

        while not done:
            time_step_total += 1 # 更新全局步数
            episode_steps += 1

            # --- 核心修改：对观测值进行归一化 ---
            obs_rms.update(current_obs_history_raw[-1:])
            normalized_obs_history = (current_obs_history_raw - obs_rms.mean) / (obs_rms.std + 1e-8)
            normalized_obs_history = np.clip(normalized_obs_history, -10.0, 10.0)

            action, log_prob, state_val = agent.select_action(normalized_obs_history)

            next_obs_history_raw, reward, done, _ = env.step(action)

            agent.store_transition(normalized_obs_history, action, log_prob, reward, done, state_val)

            current_obs_history_raw = next_obs_history_raw
            episode_reward += reward

            if time_step_total % UPDATE_TIMESTEP == 0:
                # 在更新前，对奖励进行z-score归一化
                rewards_np = np.array(agent.buffer.rewards)
                rewards_mean = np.mean(rewards_np)
                rewards_std = np.std(rewards_np) + 1e-8
                agent.buffer.rewards = ((rewards_np - rewards_mean) / rewards_std).tolist()
                
                # 获取损失并记录到 TensorBoard
                loss_info = agent.update()
                writer.add_scalar('train/actor_loss', loss_info['actor_loss'], time_step_total)
                writer.add_scalar('train/critic_loss', loss_info['critic_loss'], time_step_total)
                writer.add_scalar('train/entropy_bonus', loss_info['entropy_bonus'], time_step_total)
                writer.add_scalar('train/total_loss', loss_info['total_loss'], time_step_total)
                writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], time_step_total)
                
                # 重置步数计数器 (这里是针对 UPDATE_TIMESTEP 的计数，不是全局步数)
                # time_step = 0 # 移除此行，因为我们使用 time_step_total 作为全局计数器

        total_rewards.append(episode_reward)
        avg_rewards_window.append(episode_reward)
        avg_reward = np.mean(avg_rewards_window)
        scheduler.step()
        print(f"Episode {episode} 结束, 总奖励: {episode_reward:.2f}, 平均奖励 (最近100轮): {avg_reward:.2f}")

        # --- 记录 Episode 级别的指标到 TensorBoard ---
        writer.add_scalar('rollout/ep_rew_mean', episode_reward, time_step_total)
        writer.add_scalar('rollout/ep_len_mean', episode_steps, time_step_total)
        writer.add_scalar('rollout/avg_ep_rew_100_episodes', avg_reward, time_step_total)


        if episode % 25 == 0:
            print(f"--- Episode {episode}，保存模型和归一化统计数据 ---")
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            agent.save_model(MODEL_SAVE_PATH)
            with open(RMS_SAVE_PATH, 'wb') as f:
                pickle.dump(obs_rms, f)

    env.close()
    print("训练完成！")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    agent.save_model(MODEL_SAVE_PATH)
    with open(RMS_SAVE_PATH, 'wb') as f:
        pickle.dump(obs_rms, f)

    plt.figure(figsize=(12, 6))
    plt.plot(total_rewards, label='Episode Reward')
    if len(total_rewards) >= 100:
        moving_avg = np.convolve(total_rewards, np.ones(100)/100, mode='valid')
        plt.plot(np.arange(99, len(total_rewards)), moving_avg, label='Moving Average (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Attention-MLP-PPO Training Progress (with Normalization)')
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(PLOT_SAVE_PATH), exist_ok=True)
    plt.savefig(PLOT_SAVE_PATH)
    print(f"奖励曲线已保存到 {PLOT_SAVE_PATH}")

    # --- 关闭 TensorBoard SummaryWriter ---
    writer.close()
    print("TensorBoard SummaryWriter 已关闭。")


if __name__ == '__main__':
    train()
