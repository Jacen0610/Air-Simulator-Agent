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
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pickle # 用于加载 RunningMeanStd
import time

# --- 关键组件导入 ---
from env.go_simulator_env import GoSimulatorEnv
from agent.ppo_attention_mlp_agent import ActorCriticAttentionMLP

# --- 从训练脚本复制 RunningMeanStd 类 ---
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

# --- 配置 ---
EVAL_EPISODES = 10
SEQUENCE_LENGTH = 10
# --- 修正: 状态维度应为 8 ---
STATE_DIM = 8
ACTION_DIM = 2
# --- 修正: 隐藏层维度应为 64 ---
HIDDEN_DIM = 64

# 热图保存配置
GENERATE_HEATMAPS = True 

# --- 使用基于项目根目录的绝对路径 ---
MODEL_LOAD_PATH = os.path.join(project_root, "Pytorch/models/attention_mlp_ppo_model.pth")
RMS_LOAD_PATH = os.path.join(project_root, "Pytorch/models/attention_mlp_ppo_rms.pkl")
PLOT_DIR = os.path.join(project_root, "Pytorch/plots/eval")
HEATMAP_DIR = os.path.join(project_root, "Pytorch/plots/eval/attention_heatmaps")

# 确保保存目录存在
os.makedirs(PLOT_DIR, exist_ok=True)
if GENERATE_HEATMAPS:
    os.makedirs(HEATMAP_DIR, exist_ok=True)

# --- 绘图函数 ---

def plot_evaluation_rewards(rewards: list, title: str, filename:str):
    if not rewards:
        print("没有可供绘制的奖励数据。")
        return
    episodes = range(1, len(rewards) + 1)
    plt.figure(figsize=(12, 7))
    plt.plot(episodes, rewards, color='dodgerblue', linestyle='-', linewidth=2, label='Episode Reward')
    plt.scatter(episodes, rewards, color='red', zorder=5)
    for i, reward in enumerate(rewards):
        plt.text(episodes[i], reward, f' {reward:.2f}', va='center', ha='center') # 居中对齐
    plt.title(title, fontsize=16)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Original Reward", fontsize=12) # 修正Y轴标签
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"评估奖励图表已保存至: {filename}")
    plt.close()

def plot_attention_heatmap(weights, step, episode, action, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    y_labels = [f"t-{i}" for i in range(SEQUENCE_LENGTH - 1, -1, -1)]
    action_str = "SEND" if action == 1 else "WAIT"

    plt.figure(figsize=(4, 8))
    sns.heatmap(
        weights,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar=True,
        xticklabels=[],
        yticklabels=y_labels
    )
    
    title = f"Attention for Action '{action_str}'\n(Episode {episode}, Step {step})"
    plt.title(title, fontsize=14)
    plt.ylabel("Relative Timestep in History", fontsize=12)
    
    filename = os.path.join(save_dir, f"heatmap_ep{episode}_step{step}_action_{action_str}.png")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def evaluate():
    """
    加载并评估模型，并在所有 episodes 中选择 "SEND" 动作时生成注意力热图。
    """
    # --- 1. 初始化环境和模型 ---
    print("正在初始化环境...")
    env = GoSimulatorEnv(grpc_server_address='localhost:50051', sequence_length=SEQUENCE_LENGTH) # 确保 grpc_server_address 正确

    # --- 加载归一化统计数据 ---
    print(f"正在加载归一化统计数据: {RMS_LOAD_PATH}")
    if not os.path.exists(RMS_LOAD_PATH):
        print(f"错误：找不到归一化统计数据文件 '{RMS_LOAD_PATH}'。")
        print("请确保 train_attention_mlp_ppo.py 脚本已运行并保存了 RMS 统计数据。")
        return
    with open(RMS_LOAD_PATH, 'rb') as f:
        obs_rms = pickle.load(f)
    print("归一化统计数据加载成功！")

    print(f"正在准备加载模型: {MODEL_LOAD_PATH}")
    if not os.path.exists(MODEL_LOAD_PATH):
        print(f"错误：找不到模型文件 '{MODEL_LOAD_PATH}'。")
        return

    device = torch.device("cpu")
    print(f"评估将使用设备: {device}")

    policy = ActorCriticAttentionMLP(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        sequence_length=SEQUENCE_LENGTH
    ).to(device)

    try:
        policy.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
        print("模型权重加载成功！")
    except Exception as e:
        print(f"加载模型权重时发生错误: {e}")
        env.close()
        return
    
    policy.eval() # 设置模型为评估模式

    # --- 2. 运行评估循环 ---
    print(f"\n开始评估模型，共 {EVAL_EPISODES} 个 episodes...")
    
    eval_rewards = []
    for i in range(1, EVAL_EPISODES + 1):
        current_obs_history_raw = env.reset()
        done = False
        episode_reward = 0
        step_count = 0

        while not done:
            # --- 核心修改：对观测值进行归一化 ---
            # 归一化整个历史序列
            normalized_obs_history = (current_obs_history_raw - obs_rms.mean) / (obs_rms.std + 1e-8)
            # 裁剪归一化后的值
            normalized_obs_history = np.clip(normalized_obs_history, -10.0, 10.0)

            state_tensor = torch.tensor(normalized_obs_history, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action_dist, _, attention_weights = policy(state_tensor, return_weights=True)
                action = action_dist.probs.argmax().item()

            if GENERATE_HEATMAPS and action == 1:
                weights_np = attention_weights.squeeze(0).cpu().numpy()
                plot_attention_heatmap(weights_np, step_count, i, action, HEATMAP_DIR)

            next_obs_history_raw, reward, done, _ = env.step(action)
            
            current_obs_history_raw = next_obs_history_raw
            episode_reward += reward
            step_count += 1
        
        eval_rewards.append(episode_reward)
        print(f"评估 Episode {i}/{EVAL_EPISODES} | Reward: {episode_reward:.2f} | Steps: {step_count}")

    # --- 3. 绘制并保存结果 ---
    print("\n评估完成。正在绘制奖励图表...")
    # 使用时间戳确保文件名唯一
    timestamp = int(time.time())
    PLOT_FILENAME = os.path.join(PLOT_DIR, f"attention_mlp_ppo_evaluation_rewards.png")
    plot_evaluation_rewards(
        eval_rewards,
        f"Attention-MLP PPO Evaluation (Avg: {np.mean(eval_rewards):.2f})",
        PLOT_FILENAME
    )

    # --- 4. 清理 ---
    print("\n流程结束，关闭环境。")
    env.close()


if __name__ == '__main__':
    evaluate()
