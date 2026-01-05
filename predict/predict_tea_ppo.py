import sys
import os
import argparse
import numpy as np

# --- 动态添加项目根目录到 sys.path ---
# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 假设项目根目录是脚本所在目录的父目录 (Air-Simulator-Agent/)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
# 将项目根目录添加到 sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor # [新增] 显式导入 Monitor

# 导入 TEA 相关的环境和特征提取器
# 即使不直接调用，也必须导入，以便 SB3 在加载模型时能够识别自定义类
from env.tea_gym_env import TEAGymEnv
from agent.tea_feature_extractor import TEA_Extractor_V2

def main():
    """
    使用 stable_baselines3.common.evaluation.evaluate_policy 进行高效评估。
    这个方法主要用于快速获取模型的性能基准（平均奖励），SPS 会很高，
    但不会提供自定义的逐步诊断信息（如熵、价值等）。
    """
    # --- 解析命令行参数 ---
    parser = argparse.ArgumentParser(description='Fast-Evaluate SB3 TEA-PPO Agent using evaluate_policy')
    parser.add_argument('--grpc_port', type=str, default='50051', help='gRPC server port (default: 50051)')
    args = parser.parse_args()
    grpc_address = f'localhost:{args.grpc_port}'
    print(f"Using gRPC server address: {grpc_address}")

    # --- 配置 ---
    EVAL_EPISODES = 20  # 建议使用稍多的 episode 数量以获得更稳定的平均值
    SEQUENCE_LENGTH = 32 # 必须与训练脚本 train_sb3_tea_ppo.py 中的设置一致
    
    # --- 路径设置 ---
    # 检查训练脚本中可能存在的 models/models 嵌套路径
    MODEL_PATH = os.path.join(project_root, "SB3/models/models/tea_ppo_final.zip")
    STATS_PATH = os.path.join(project_root, "SB3/models/models/tea_ppo_vec_norm.pkl")
    
    # 如果嵌套路径不存在，则使用标准路径
    if not os.path.exists(MODEL_PATH):
        print(f"Info: Did not find model at '{MODEL_PATH}', trying standard path...")
        MODEL_PATH = os.path.join(project_root, "SB3/models/tea_ppo_final.zip")
        STATS_PATH = os.path.join(project_root, "SB3/models/tea_ppo_vec_norm.pkl")

    # 检查模型文件和统计数据文件是否存在
    if not os.path.exists(MODEL_PATH) or not os.path.exists(STATS_PATH):
        print(f"Error: Model or stats file not found.")
        print(f"Checked paths: '{MODEL_PATH}' and '{STATS_PATH}'")
        print("Please ensure train_sb3_tea_ppo.py has been run and saved the model.")
        return

    # --- 1. 创建并加载归一化环境 (关键步骤) ---
    print("Initializing TEA Gym environment and loading normalization stats...")
    
    # [改进] 1.1. 创建一个基础的、被 Monitor 包装的 lambda 函数
    # 这是 evaluate_policy 的标准做法
    env_lambda = lambda: Monitor(TEAGymEnv(grpc_server_address=grpc_address, sequence_length=SEQUENCE_LENGTH))
    
    # 1.2. 使用该 lambda 函数创建 VecEnv
    base_vec_env = DummyVecEnv([env_lambda])
    
    # 1.3. 使用 .load 方法加载统计数据，这将返回一个配置好的 VecNormalize 环境
    env = VecNormalize.load(STATS_PATH, base_vec_env)
    
    # 1.4. 设置为评估模式
    env.training = False      # 不更新运行中的均值和方差
    env.norm_reward = False   # 获取真实的、未经归一化的奖励
    
    print("Environment loaded and configured for evaluation.")

    # --- 2. 加载训练好的模型 ---
    print(f"Loading trained model from {MODEL_PATH}...")
    try:
        # 无需手动指定 device='cuda'，SB3 会自动加载到保存时所在的设备
        model = PPO.load(MODEL_PATH, env=env)
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        env.close()
        return

    # --- 3. 运行高效评估 ---
    print(f"\nStarting evaluation for {EVAL_EPISODES} episodes...")
    
    # evaluate_policy 是 SB3 提供的标准评估工具，它经过优化，速度很快
    # return_episode_rewards=True 可以让我们方便地计算标准差
    # warn=True 是一个有用的安全检查，确保 Monitor 包装器存在
    episode_rewards, episode_lengths = evaluate_policy(
        model,
        env,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        return_episode_rewards=True,
        warn=True
    )
    
    # --- 4. 打印评估结果 ---
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    print("\n" + "="*50)
    print("        High-Speed Evaluation Results")
    print("="*50)
    print(f"Episodes Evaluated: {len(episode_rewards)}")
    print(f"Mean Reward:        {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.2f} steps")
    print("="*50)

    # --- 5. 清理 ---
    print("\nEvaluation complete. Closing environment.")
    env.close()

if __name__ == '__main__':
    main()
