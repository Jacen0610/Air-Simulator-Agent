# evaluate_attention_mlp_ppo.py
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns  # 导入 seaborn 用于绘制高质量热图

# --- 关键组件导入 ---
from go_simulator_env import GoSimulatorEnv
from ppo_attention_mlp_agent import ActorCriticAttentionMLP

# --- 配置 ---
EVAL_EPISODES = 10
SEQUENCE_LENGTH = 10
STATE_DIM = 7
ACTION_DIM = 2
HIDDEN_DIM = 128

# 热图保存配置
HEATMAP_DIR = "attention_heatmaps"
# 设置为 True 以在所有 episodes 的 "SEND" 动作时生成热图
GENERATE_HEATMAPS = True 

MODEL_LOAD_PATH = "attention_mlp_ppo_model.pth"
PLOT_SAVE_PATH = "evaluation_rewards_attention_mlp_ppo.png"

# --- 绘图函数 ---

def plot_evaluation_rewards(rewards: list, title: str, filename:str):
    # (此函数保持不变)
    if not rewards:
        print("没有可供绘制的奖励数据。")
        return
    episodes = range(1, len(rewards) + 1)
    plt.figure(figsize=(12, 7))
    plt.plot(episodes, rewards, color='dodgerblue', linestyle='-', linewidth=2, label='Episode Reward')
    plt.scatter(episodes, rewards, color='red', zorder=5)
    for i, reward in enumerate(rewards):
        plt.text(episodes[i], reward, f' {reward:.2f}', va='center')
    plt.title(title, fontsize=16)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"评估奖励图表已保存至: {filename}")
    plt.close()

def plot_attention_heatmap(weights, step, episode, action, save_dir):
    """
    为单步的注意力权重绘制并保存热图。
    """
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
    env = GoSimulatorEnv(sequence_length=SEQUENCE_LENGTH)

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
    
    policy.eval()

    # --- 2. 运行评估循环 ---
    print(f"\n开始评估模型，共 {EVAL_EPISODES} 个 episodes...")
    
    eval_rewards = []
    for i in range(1, EVAL_EPISODES + 1):
        current_obs_history = env.reset()
        done = False
        episode_reward = 0
        step_count = 0

        while not done:
            state_tensor = torch.tensor(current_obs_history, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action_dist, _, attention_weights = policy(state_tensor, return_weights=True)
                action = action_dist.probs.argmax().item()

            # [核心修改] 只要开启热图生成且动作为 "SEND" (1)，就绘制热图
            if GENERATE_HEATMAPS and action == 1:
                weights_np = attention_weights.squeeze(0).cpu().numpy()
                plot_attention_heatmap(weights_np, step_count, i, action, HEATMAP_DIR)

            next_obs_history, reward, done, _ = env.step(action)
            
            current_obs_history = next_obs_history
            episode_reward += reward
            step_count += 1
        
        eval_rewards.append(episode_reward)
        print(f"评估 Episode {i}/{EVAL_EPISODES} | Reward: {episode_reward:.2f} | Steps: {step_count}")

    # --- 3. 绘制并保存结果 ---
    print("\n评估完成。正在绘制奖励图表...")
    plot_evaluation_rewards(
        eval_rewards,
        f"Attention-MLP PPO Evaluation ({len(eval_rewards)} Episodes)",
        PLOT_SAVE_PATH
    )

    # --- 4. 清理 ---
    print("\n流程结束，关闭环境。")
    env.close()


if __name__ == '__main__':
    evaluate()
