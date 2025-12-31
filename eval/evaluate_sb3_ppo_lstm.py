import sys
import os
import argparse # 导入 argparse

# --- 动态添加项目根目录到 sys.path ---
# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 假设项目根目录是脚本所在目录的父目录 (Air-Simulator-Agent/)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
# 将项目根目录添加到 sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import time
import torch as th # 导入 torch

# [核心修改] 导入 RecurrentPPO 和对应的 LSTM 环境
from sb3_contrib import RecurrentPPO
# --- 导入手动包装所需的组件 ---
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure # 导入 logger 配置
# --------------------------------
from env.gym_env_for_lstm import GymEnvForLSTM

def plot_evaluation_rewards(rewards: list, title: str, filename: str):
    """
    为评估过程绘制奖励曲线。
    """
    if not rewards:
        print("没有可供绘制的奖励数据。")
        return

    episodes = range(1, len(rewards) + 1)
    
    plt.figure(figsize=(12, 7))
    
    plt.plot(episodes, rewards, color='dodgerblue', linestyle='-', linewidth=2, label='Episode Reward')
    plt.scatter(episodes, rewards, color='red', zorder=5)

    for i, reward in enumerate(rewards):
        plt.text(episodes[i], reward, f' {reward:.2f}', va='center', ha='center')

    plt.title(title, fontsize=16)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Original Reward", fontsize=12)
    
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(filename)
    print(f"评估奖励图表已保存至: {filename}")
    plt.close()

def main():
    """
    主评估流程 - 专门用于 RecurrentPPO (LSTM) 模型。
    """
    # --- 解析命令行参数 ---
    parser = argparse.ArgumentParser(description='Evaluate SB3 Recurrent PPO Agent')
    parser.add_argument('--grpc_port', type=str, default='50051', help='gRPC server port (default: 50051)')
    args = parser.parse_args()
    grpc_address = f'localhost:{args.grpc_port}'
    print(f"Using gRPC server address: {grpc_address}")

    # --- 配置 ---
    EVAL_EPISODES = 10
    DUMP_FREQUENCY = 600 # 每隔多少步写入一次日志
    
    # --- 使用绝对路径 ---
    MODEL_DIR = os.path.join(project_root, "SB3/models")
    PLOT_DIR = os.path.join(project_root, "SB3/plots/eval")
    
    # TensorBoard 日志目录
    TBL_LOG_DIR = os.path.join(project_root, "sb3_logs/eval_lstm_ppo")
    
    # [核心修改] 指向 LSTM 模型文件
    MODEL_PATH = os.path.join(MODEL_DIR, "recurrent_ppo_lstm.zip")
    STATS_PATH = os.path.join(MODEL_DIR, "recurrent_ppo_lstm_vec_normalize.pkl")
    
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型文件 '{MODEL_PATH}'。")
        print("请先运行 train_sb3_ppo_lstm.py 脚本来训练并保存一个模型。")
        return
    
    if not os.path.exists(STATS_PATH):
        print(f"错误：找不到统计文件 '{STATS_PATH}'。")
        print("请先运行 train_sb3_ppo_lstm.py 脚本来训练并保存统计数据。")
        return

    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(TBL_LOG_DIR, exist_ok=True)

    # --- 1. 创建环境并加载模型 ---
    print("正在初始化为 LSTM 优化的 Gym 环境...")
    
    # --- 核心修正：手动创建和包装环境 ---
    # 1. 创建原始环境
    raw_env = GymEnvForLSTM(grpc_server_address=grpc_address)
    # 2. 使用 Monitor 包装 (在旧版本中，Monitor 默认不自动重置)
    monitored_env = Monitor(raw_env)
    # 3. 转换为 VecEnv
    vec_env = DummyVecEnv([lambda: monitored_env])
    # 4. 使用 VecNormalize 加载统计数据
    env = VecNormalize.load(STATS_PATH, vec_env)
    # ------------------------------------

    # 评估时不要更新统计数据
    env.training = False
    # 评估时也不要归一化奖励，以便看到真实的奖励值
    env.norm_reward = False

    print(f"正在从 {MODEL_PATH} 加载已训练的 RecurrentPPO 模型...")
    try:
        # [核心修改] 使用 RecurrentPPO.load
        model = RecurrentPPO.load(MODEL_PATH, env=env)
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        env.close()
        return

    # --- 配置 TensorBoard Logger ---
    run_name = f"eval_run_{int(time.time())}"
    new_logger = configure(os.path.join(TBL_LOG_DIR, run_name), ["stdout", "tensorboard"])
    model.set_logger(new_logger)
    print(f"TensorBoard 日志将保存至: {os.path.join(TBL_LOG_DIR, run_name)}")

    # --- 2. 运行评估循环 ---
    print(f"\n开始评估模型，共 {EVAL_EPISODES} 个 episodes...")
    
    eval_rewards = []
    episodes_completed = 0
    total_steps = 0 # 引入总步数计数器
    
    # --- 核心修正：适配 VecEnv 的自动重置行为 ---
    # 1. 在循环外只 reset 一次
    obs = env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    
    # 用于收集每个 episode 的统计数据
    episode_values = []
    episode_entropies = []

    # 2. 使用 while 循环，直到完成指定数量的 episodes
    while episodes_completed < EVAL_EPISODES:
        total_steps += 1 # 步数 +1
        
        # 1. 先预测动作 (同时获取下一个 LSTM 状态)
        # 注意：这里我们使用当前的 lstm_states
        action, next_lstm_states = model.predict(
            obs,
            state=lstm_states, 
            episode_start=episode_starts,
            deterministic=True
        )

        # 2. 计算价值和熵 (使用当前的 lstm_states 和 刚刚预测的 action)
        # RecurrentPPO 的 predict_values 不支持 state 参数，所以我们使用 evaluate_actions
        with th.no_grad():
            obs_tensor = th.as_tensor(obs).to(model.device)
            action_tensor = th.as_tensor(action).to(model.device)
            episode_starts_tensor = th.as_tensor(episode_starts).to(model.device)
            
            # 转换 lstm_states 为 tensor (因为 model.predict 返回的是 numpy)
            if lstm_states is None:
                lstm_states_tensor = None
            else:
                lstm_states_tensor = tuple(th.as_tensor(s).to(model.device) for s in lstm_states)

            # evaluate_actions 返回 values, log_prob, entropy
            values, log_prob, entropy = model.policy.evaluate_actions(
                obs_tensor, 
                action_tensor, 
                lstm_states_tensor, 
                episode_starts_tensor
            )
            
            current_value = values.item()
            current_entropy = entropy.mean().item()

            # 记录实时数据
            model.logger.record("trace/step_entropy", current_entropy)
            model.logger.record("trace/step_value", current_value)

            episode_values.append(current_value)
            episode_entropies.append(current_entropy)

        # 3. 更新状态
        lstm_states = next_lstm_states

        # --- 定期写入日志 ---
        if total_steps % DUMP_FREQUENCY == 0:
            model.logger.dump(step=total_steps)

        # 4. 执行环境步进
        obs, reward, done, info = env.step(action)
        
        # 在 episode 的后续步骤中，episode_starts 应为 False
        episode_starts = done
        
        # 5. 检查 info 字典，看 VecEnv 是否自动重置了环境
        if 'episode' in info[0]:
            episodes_completed += 1
            original_episode_reward = info[0]['episode']['r']
            episode_length = info[0]['episode']['l']
            
            eval_rewards.append(original_episode_reward)
            
            # 计算本 episode 的平均价值和平均熵
            avg_value = np.mean(episode_values) if episode_values else 0.0
            avg_entropy = np.mean(episode_entropies) if episode_entropies else 0.0
            
            print(f"评估 Episode {episodes_completed}/{EVAL_EPISODES} | "
                  f"Reward: {original_episode_reward:.2f} | "
                  f"Steps: {episode_length} | "
                  f"Avg Value: {avg_value:.4f} | "
                  f"Avg Entropy: {avg_entropy:.4f}")
            
            # --- 记录到 TensorBoard (使用 total_steps 作为 X 轴) ---
            model.logger.record("eval/reward", original_episode_reward)
            model.logger.record("eval/episode_length", episode_length)
            model.logger.record("eval/mean_value_estimate", avg_value)
            model.logger.record("eval/mean_entropy", avg_entropy)
            model.logger.dump(step=total_steps)
            
            # 重置统计列表
            episode_values = []
            episode_entropies = []

    # --- 3. 绘制并保存奖励图表 ---
    print("\n评估完成。正在绘制奖励图表...")
    timestamp = int(time.time())
    PLOT_FILENAME = os.path.join(PLOT_DIR, f"evaluation_rewards_lstm_{timestamp}.png")
    plot_evaluation_rewards(
        eval_rewards,
        f"LSTM PPO Model Evaluation Rewards (Avg: {np.mean(eval_rewards):.2f})",
        PLOT_FILENAME
    )

    # --- 4. 清理 ---
    print("\n流程结束，关闭环境。")
    env.close()

if __name__ == '__main__':
    main()
