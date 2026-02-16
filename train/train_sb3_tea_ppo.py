import warnings

warnings.filterwarnings("ignore", message="You are trying to run PPO on the GPU")

import os, sys
import argparse
import numpy as np
from datetime import datetime
import torch as th

# --- 动态添加项目根目录 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_linear_fn

from agent.tea_feature_extractor import TEA_Extractor_V3
from env.tea_gym_env import TEAGymEnv


# --- 修改后的检查点保存 Callback ---
class SaveCheckpointCallback(BaseCallback):
    def __init__(self, n_episodes: int, total_episodes: int, save_path: str, verbose=1):
        super().__init__(verbose)
        self.n_episodes = n_episodes
        self.total_episodes = total_episodes  # 新增：总目标轮数
        self.save_path = save_path
        self.episode_count = 0
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            self.episode_count += 1

            # 1. 每 N 轮保存一次快照
            if self.episode_count % self.n_episodes == 0:
                timestamp = datetime.now().strftime("%H%M")
                model_path = os.path.join(self.save_path, f"tea_ep{self.episode_count}_{timestamp}.zip")
                stats_path = os.path.join(self.save_path, f"stats_ep{self.episode_count}_{timestamp}.pkl")

                self.model.save(model_path)
                self.training_env.save(stats_path)
                print(f"\n>>> 第 {self.episode_count} 轮存盘成功。")

            # 2. 核心逻辑：达到总轮数后停止训练
            if self.episode_count >= self.total_episodes:
                print(f"\n" + "!" * 30)
                print(f">>> 已达到目标训练轮数 ({self.total_episodes})，正在触发停止...")
                print("!" * 30 + "\n")
                return False  # 返回 False 会让 model.learn() 立即安全退出

        return True  # 继续训练


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_port', type=int, default=50051)
    parser.add_argument('--total_episodes', type=int, default=30)
    parser.add_argument('--n_steps', type=int, default=8192)
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--model_name', type=str, default='tea_ppo_final')
    args = parser.parse_args()

    # --- 物理步数与学习率校准 ---
    SPS = 250 if args.n_steps >= 8192 else 300
    STEPS_PER_EPISODE = SPS * 60 * 40
    TOTAL_STEPS = STEPS_PER_EPISODE * args.total_episodes

    GAMMA = 0.98
    # 更改保存目录名以区分版本
    MODEL_DIR = os.path.join(project_root, "SB3/models/checkpoints_v3")
    LOG_DIR = "./sb3_logs/tea_ppo/train/"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- 环境初始化 ---
    vec_env = make_vec_env(lambda: TEAGymEnv(
        grpc_server_address=f"localhost:{args.train_port}",
        sequence_length=96
    ), n_envs=1)

    # --- 加载或新建逻辑 ---
    if args.fine_tune and os.path.exists(os.path.join(MODEL_DIR, args.model_name + ".zip")):
        load_path = os.path.join(MODEL_DIR, args.model_name)
        stats_path = os.path.join(MODEL_DIR, f"{args.model_name}_stats.pkl")
        env = VecNormalize.load(stats_path, vec_env)
        model = PPO.load(load_path, env=env, device="cuda" if th.cuda.is_available() else "cpu")
        print(f">>> 已从 {load_path} 加载现有模型进行微调")
    else:
        env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, gamma=GAMMA)
        lr_schedule = get_linear_fn(1e-4, 1e-6, TOTAL_STEPS)

        policy_kwargs = dict(
            features_extractor_class=TEA_Extractor_V3,
            features_extractor_kwargs=dict(features_dim=512, embed_dim=128),
            net_arch=dict(pi=[256, 128], vf=[512, 256])
        )

        model = PPO(
            "MlpPolicy", env, policy_kwargs=policy_kwargs,
            learning_rate=lr_schedule,
            gamma=GAMMA,
            n_steps=args.n_steps,
            batch_size=1024,
            n_epochs=10,
            ent_coef=0.02,
            clip_range=0.2,
            gae_lambda=0.95,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.015,
            verbose=0,  # 4060Ti 建议开启 1 方便观察进度
            tensorboard_log=LOG_DIR,
            device="cuda" if th.cuda.is_available() else "cpu"
        )

    # --- 实例化检查点保存回调 ---
    checkpoint_cb = SaveCheckpointCallback(
        n_episodes=5,
        total_episodes=args.total_episodes,  # 传入这个参数
        save_path=MODEL_DIR
    )

    print(f"\n>>> 4060Ti 训练启动 | 总计回合: {args.total_episodes} | 预估总步数: {TOTAL_STEPS}")

    try:
        model.learn(total_timesteps=TOTAL_STEPS + 5000, callback=checkpoint_cb)
    finally:
        suffix = datetime.now().strftime("%Y%m%d_%H%M")
        model.save(os.path.join(MODEL_DIR, f"tea_final_{suffix}"))
        env.save(os.path.join(MODEL_DIR, f"tea_final_{suffix}_stats.pkl"))
        env.close()


if __name__ == '__main__':
    main()