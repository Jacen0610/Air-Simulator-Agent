# 文件路径: train_sb3_tea_ppo.py
import os, sys
import argparse # 导入 argparse
from typing import Callable

# --- 动态添加项目根目录 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# 导入我们新定义的组件
from agent.tea_feature_extractor import TEA_Extractor_V3
from env.tea_gym_env import TEAGymEnv
# 复用你原来的 Callback 和 Plot 函数
from train.train_sb3_transformer_ppo import AttentionVisualizationCallback, plot_rewards


def main():
    # --- 解析命令行参数 ---
    parser = argparse.ArgumentParser(description='Train SB3 TEA-PPO Agent')
    parser.add_argument('--grpc_port', type=str, default='50051', help='gRPC server port (default: 50051)')
    args = parser.parse_args()
    grpc_address = f'localhost:{args.grpc_port}'
    print(f"Using gRPC server address: {grpc_address}")

    # --- 1. 参数设置 ---
    TOTAL_EPISODES = 30  # TEA 结构较深，建议多跑一些 Episode 观察收敛
    GAMMA = 0.98
    SEQUENCE_LENGTH = 96

    MODEL_DIR = os.path.join(project_root, "SB3/models")
    PLOT_DIR = os.path.join(project_root, "SB3/plots/train")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # --- 2. 环境初始化 ---
    vec_env = make_vec_env(lambda: TEAGymEnv(
        grpc_server_address=grpc_address,
        sequence_length=SEQUENCE_LENGTH
    ), n_envs=1)

    # 同步 Gamma 以适配 VecNormalize
    env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, gamma=GAMMA)

    # --- 3. 定义 TEA-PPO 策略参数 ---
    policy_kwargs = dict(
        features_extractor_class=TEA_Extractor_V2,
        features_extractor_kwargs=dict(
            features_dim=512,
            embed_dim=128
        ),
        # 决策头：TEA 已经集成了 LSTM 特征，后端 MLP 保持轻量
        net_arch=dict(pi=[128, 64], vf=[512,256])
    )

    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        """
        线性学习率调度器。
        :param initial_value: 初始学习率。
        :return: 学习率调度函数。
        """

        def func(progress_remaining: float) -> float:
            """
            progress_remaining 从 1.0 减少到 0.0。
            """
            return progress_remaining * initial_value

        return func

    # 使用你建议的起始学习率
    initial_lr = 1e-4
    # --- 4. 实例化模型 ---
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        learning_rate=linear_schedule(initial_lr),  # 这里应用线性衰减
        gamma=GAMMA,
        n_steps=4096,  # 增加更新频率
        batch_size=512,  # 适配 FPS
        n_epochs=3,  # 充分利用每批数据
        clip_range=0.2,
        gae_lambda=0.95,
        ent_coef=0.05,
        vf_coef=0.2,
        max_grad_norm=0.3,
        device="cuda",
        tensorboard_log="./sb3_logs/tea_ppo/train/"
    )

    callback = AttentionVisualizationCallback(total_episodes=TOTAL_EPISODES, viz_freq=20, verbose=1)

    try:
        print("TEA-PPO 训练启动: Attention 过滤 + LSTM 记忆...")
        model.learn(total_timesteps=int(1e12), callback=callback, tb_log_name="TEA_PPO_v1")
    finally:
        model.save(os.path.join(MODEL_DIR,"tea_ppo_final"))
        env.save(os.path.join(MODEL_DIR,"tea_ppo_vec_norm.pkl"))
        env.close()


if __name__ == '__main__':
    main()