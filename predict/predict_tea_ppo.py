import sys
import os
import argparse
import numpy as np

# --- 动态添加项目根目录到 sys.path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from env.tea_gym_env import TEAGymEnv
from agent.tea_feature_extractor import TEA_Extractor_V2


def main():
    # --- 解析命令行参数 ---
    parser = argparse.ArgumentParser(description='Fast-Evaluate SB3 TEA-PPO Agent')
    parser.add_argument('--grpc_port', type=str, default='50051', help='gRPC server port')
    parser.add_argument('--eval_episodes', type=int, default=10, help='评估的回合数量')

    # 默认指向项目根目录下的 SB3/models/
    default_model_dir = os.path.join(project_root, "SB3/models")

    # 用户只需输入文件名，无需输入路径和后缀
    parser.add_argument('--model_name', type=str, default='tea_ppo_final', help='模型文件名 (不带.zip)')
    parser.add_argument('--stats_name', type=str, default='tea_ppo_vec_norm', help='统计文件名 (不带.pkl)')

    args = parser.parse_args()
    grpc_address = f'localhost:{args.grpc_port}'

    # --- 拼接完整路径 ---
    MODEL_PATH = os.path.join(default_model_dir, f"{args.model_name}.zip")
    STATS_PATH = os.path.join(default_model_dir, f"{args.stats_name}.pkl")

    # 检查文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"Error: 找不到模型文件 {MODEL_PATH}")
        return
    if not os.path.exists(STATS_PATH):
        print(f"Error: 找不到统计文件 {STATS_PATH}")
        return

    # --- 配置 ---
    SEQUENCE_LENGTH = 96

    # --- 1. 创建并加载归一化环境 ---
    print(f"正在从 {default_model_dir} 加载环境统计: {args.stats_name}.pkl")

    env_lambda = lambda: Monitor(TEAGymEnv(grpc_server_address=grpc_address, sequence_length=SEQUENCE_LENGTH))
    base_vec_env = DummyVecEnv([env_lambda])

    env = VecNormalize.load(STATS_PATH, base_vec_env)
    env.training = False
    env.norm_reward = False

    # --- 2. 加载训练好的模型 ---
    print(f"正在加载模型: {args.model_name}.zip")
    try:
        model = PPO.load(MODEL_PATH, env=env)
    except Exception as e:
        print(f"模型加载失败: {e}")
        env.close()
        return

    # --- 3. 运行评估 ---
    print(f"\n开始评估 {args.eval_episodes} 个回合...")

    episode_rewards, episode_lengths = evaluate_policy(
        model,
        env,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        return_episode_rewards=True,
        warn=True
    )

    # --- 4. 打印结果 ---
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)

    print("\n" + "=" * 50)
    print(f"模型评估报告: {args.model_name}")
    print("=" * 50)
    print(f"总计评估回合: {len(episode_rewards)}")
    print(f"平均奖励:     {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"平均回合长度: {mean_length:.2f} steps")
    print("=" * 50)

    env.close()


if __name__ == '__main__':
    main()