import os, sys
import argparse
import numpy as np
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
from stable_baselines3.common.callbacks import BaseCallback

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
    parser.add_argument('--model_name', type=str, default='tea_ppo_final', help='加载的模型名')
    # 新增：通过参数控制回合数
    parser.add_argument('--total_episodes', type=int, default=50, help='本次训练跑多少个回合')
    args = parser.parse_args()

    grpc_address = f'localhost:{args.grpc_port}'

    # --- 2. 核心参数 ---
    GAMMA = 0.95
    SEQUENCE_LENGTH = 96
    MODEL_DIR = os.path.join(project_root, "SB3/models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, args.model_name)
    stats_path = os.path.join(MODEL_DIR, "tea_ppo_vec_norm.pkl")

    # --- 3. 环境初始化 ---
    vec_env = make_vec_env(lambda: TEAGymEnv(
        grpc_server_address=grpc_address,
        sequence_length=SEQUENCE_LENGTH
    ), n_envs=1)

    # --- 4. 模型加载或新建 ---
    if args.continue_train and os.path.exists(model_path + ".zip"):
        print(f"载入模型: {model_path} | 目标回合: {args.total_episodes}")
        env = VecNormalize.load(stats_path, vec_env)
        model = PPO.load(model_path, env=env, device="cuda")

        if args.fine_tune:
            print(">>> 模式: Fine-tune (收网) - 极低熵系数")
            model.ent_coef = 0.001
            model.learning_rate = 3e-5
        else:
            print(">>> 模式: Continue (追加训练)")
            model.learning_rate = 1e-4
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

    # --- 5. 训练控制 ---
    episode_callback = EpisodeControlCallback(max_episodes=args.total_episodes, verbose=0)

    try:
        model.learn(total_timesteps=int(1e10), callback=episode_callback, tb_log_name="TEA_PPO_v1")
    finally:
        # 根据模式自动命名，防止覆盖
        mode_tag = "finetuned" if args.fine_tune else "continued" if args.continue_train else "initial"
        save_name = f"tea_ppo_{mode_tag}"
        model.save(os.path.join(MODEL_DIR, save_name))
        env.save(stats_path)
        print(f"训练结束。模型保存为: {save_name}, 环境统计已更新。")
        env.close()


if __name__ == '__main__':
    main()