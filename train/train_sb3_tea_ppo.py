import os, sys
import argparse
import numpy as np
from typing import Callable
from datetime import datetime

# --- 动态添加项目根目录 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import constant_fn

from agent.tea_feature_extractor import TEA_Extractor_V2
from env.tea_gym_env import TEAGymEnv


# --- 自定义回合控制 Callback ---
class EpisodeControlCallback(BaseCallback):
    def __init__(self, max_episodes: int, verbose=0):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.episode_count = 0

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            self.episode_count += 1
            if self.verbose > 0:
                print(f"进度: 第 {self.episode_count}/{self.max_episodes} 个回合完成")

            if self.episode_count >= self.max_episodes:
                print(f">>> 已达到目标回合数 {self.max_episodes}，停止训练并保存...")
                return False
        return True


def main():
    # --- 1. 解析命令行参数 ---
    parser = argparse.ArgumentParser(description='Train SB3 TEA-PPO Agent')
    parser.add_argument('--grpc_port', type=str, default='50051', help='gRPC port')
    parser.add_argument('--continue_train', action='store_true', help='是否加载模型继续训练')
    parser.add_argument('--fine_tune', action='store_true', help='是否开启低熵微调模式')
    parser.add_argument('--model_name', type=str, default='tea_ppo_final', help='加载的模型名(不带.zip)')
    parser.add_argument('--stats_name', type=str, default='tea_ppo_vec_norm.pkl', help='加载的Stats文件名')
    parser.add_argument('--total_episodes', type=int, default=50, help='本次训练跑多少个回合')
    args = parser.parse_args()

    grpc_address = f'localhost:{args.grpc_port}'

    # --- 2. 核心路径 ---
    GAMMA = 0.95
    SEQUENCE_LENGTH = 96
    MODEL_DIR = os.path.join(project_root, "SB3/models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 加载路径
    load_model_path = os.path.join(MODEL_DIR, args.model_name)
    load_stats_path = os.path.join(MODEL_DIR, args.stats_name)

    # --- 3. 环境初始化 ---
    vec_env = make_vec_env(lambda: TEAGymEnv(
        grpc_server_address=grpc_address,
        sequence_length=SEQUENCE_LENGTH
    ), n_envs=1)

    # --- 4. 模型加载或新建 ---
    if args.continue_train and os.path.exists(load_model_path + ".zip"):
        print(f"载入模型: {load_model_path} | 载入Stats: {load_stats_path}")

        # 加载归一化统计数据
        env = VecNormalize.load(load_stats_path, vec_env)

        # 加载PPO模型
        model = PPO.load(load_model_path, env=env, device="cuda")

        if args.fine_tune:
            print(">>> 模式: Fine-tune (收网模式)")

            # 【关键修改 1】锁定 Stats，不再更新均值和方差，防止特征漂移导致无效动作激增
            env.training = False
            env.norm_reward = False # 微调阶段通常建议关闭奖励归一化，使用原始惩罚分
            new_lr = 8e-6  # 极其精细的步长

            # 【关键修改 2】注入新的微调参数
            model.ent_coef = 0.0006  # 缓解僵硬，减少连点试探
            model.clip_range = constant_fn(0.1)
            model.lr_schedule = constant_fn(new_lr)

            # 强制同步优化器参数
            for param_group in model.policy.optimizer.param_groups:
                param_group['lr'] = new_lr

            print(
                f">>> 核心参数已重置: LR={new_lr}, Clip={model.clip_range}, Ent={model.ent_coef}, Stats_Locked={not env.training}")
        else:
            print(">>> 模式: Continue (追加训练)")
            model.learning_rate = 1e-4
            env.training = True  # 继续学习环境特征
    else:
        print(f">>> 模式: New (全新训练) | 目标回合: {args.total_episodes}")
        env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, gamma=GAMMA)
        policy_kwargs = dict(
            features_extractor_class=TEA_Extractor_V2,
            features_extractor_kwargs=dict(features_dim=512, embed_dim=128),
            net_arch=dict(pi=[128, 64], vf=[512, 256])
        )
        model = PPO(
            "MlpPolicy", env, policy_kwargs=policy_kwargs,
            verbose=1, learning_rate=1e-4, gamma=GAMMA,
            n_steps=2048, batch_size=512, n_epochs=10,
            ent_coef=0.1, target_kl=0.015, device="cuda",
            tensorboard_log="./sb3_logs/tea_ppo/train/"
        )

    # --- 5. 训练执行 ---
    episode_callback = EpisodeControlCallback(max_episodes=args.total_episodes, verbose=1)

    try:
        model.learn(total_timesteps=int(1e10), callback=episode_callback, tb_log_name="TEA_PPO_v2_FT")
    finally:
        # --- 6. 自动化命名保存 (年月日_时分) ---
        mode_tag = "finetuned" if args.fine_tune else "continued" if args.continue_train else "initial"
        time_suffix = datetime.now().strftime("%Y%m%d_%H%M")

        final_save_name = f"tea_ppo_{mode_tag}_{time_suffix}"

        # 保存模型权重
        model.save(os.path.join(MODEL_DIR, final_save_name))
        # 保存对应的环境统计数据 (每个模型一个专属Stats)
        env.save(os.path.join(MODEL_DIR, f"{final_save_name}_stats.pkl"))

        print("-" * 50)
        print(f"训练任务结束!")
        print(f"模型文件: {final_save_name}.zip")
        print(f"统计文件: {final_save_name}_stats.pkl")
        print("-" * 50)
        env.close()


if __name__ == '__main__':
    main()