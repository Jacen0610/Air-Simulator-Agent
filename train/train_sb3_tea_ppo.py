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
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import get_linear_fn, ConstantSchedule

from agent.tea_feature_extractor import TEA_Extractor_V3
from env.tea_gym_env import TEAGymEnv


# --- 自定义评估：每 N 个 Episode 考试一次 ---
class EveryNEpisodesEvalCallback(BaseCallback):
    def __init__(self, eval_env, n_episodes: int, save_path: str, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.n_episodes = n_episodes
        self.save_path = save_path
        self.episode_count = 0
        self.best_mean_reward = -np.inf
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            self.episode_count += 1
            if self.episode_count % self.n_episodes == 0:
                print(f"\n>>> 触发第 {self.episode_count} 回合确定性评估...")

                episode_reward = 0
                obs = self.eval_env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.eval_env.step(action)
                    episode_reward += reward[0]

                print(f">>> 评估得分: {episode_reward:.2f}")

                if episode_reward > self.best_mean_reward:
                    self.best_mean_reward = episode_reward
                    self.model.save(os.path.join(self.save_path, "best_model.zip"))
                    self.eval_env.save(os.path.join(self.save_path, "best_model_stats.pkl"))
                    print(f"★ 发现历史最高得分，已更新 best_model.zip")
        return True


# --- 主训练流程 ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grpc_port', type=str, default='50051')
    parser.add_argument('--continue_train', action='store_true')
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--model_name', type=str, default='tea_ppo_final')
    parser.add_argument('--total_episodes', type=int, default=30)
    args = parser.parse_args()

    # 配置
    GAMMA = 0.97
    MODEL_DIR = os.path.join(project_root, "SB3/models")
    LOG_DIR = "./sb3_logs/tea_ppo/train/"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 环境
    vec_env = make_vec_env(lambda: TEAGymEnv(
        grpc_server_address=f'localhost:{args.grpc_port}',
        sequence_length=96
    ), n_envs=1)

    # 逻辑判断
    if (args.continue_train or args.fine_tune) and os.path.exists(os.path.join(MODEL_DIR, args.model_name + ".zip")):
        # 加载逻辑
        load_path = os.path.join(MODEL_DIR, args.model_name)
        stats_path = os.path.join(MODEL_DIR, f"{args.model_name}_stats.pkl")

        env = VecNormalize.load(stats_path, vec_env)
        model = PPO.load(load_path, env=env, device="cuda")

        if args.fine_tune:
            print(">>> 执行改进版微调策略...")
            model.ent_coef = 0.05  # 注入探索熵
            model.learning_rate = 5e-5  # 脉冲学习率
            model.gamma = 0.99  # 强化长线时延意识
            model.clip_range = ConstantSchedule(0.2)
            env.norm_reward = False  # 确保惩罚真实
        else:
            model.learning_rate = 1e-4
    else:
        # 全新训练
        env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, gamma=GAMMA)
        policy_kwargs = dict(
            features_extractor_class=TEA_Extractor_V3,
            features_extractor_kwargs=dict(features_dim=512, embed_dim=128),
            net_arch=dict(pi=[256, 128], vf=[512, 256])
        )
        SPS = 300
        TOTAL_STEPS = args.total_episodes * 40 * 60 * SPS
        lr_schedule = get_linear_fn(1e-4, 1e-6, TOTAL_STEPS)
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=0,
                    learning_rate=lr_schedule,
                    gamma=GAMMA,
                    n_steps=8192,
                    batch_size=1024,
                    n_epochs=10,
                    ent_coef=0.02,
                    clip_range=0.2,
                    gae_lambda=0.95,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    target_kl=0.015,
                    device="cuda",
                    tensorboard_log=LOG_DIR)

    # Callback
    eval_cb = EveryNEpisodesEvalCallback(env, n_episodes=5, save_path=os.path.join(MODEL_DIR, "tea_best_model_v3"))

    # 启动
    try:
        # 这里用 step 数占位，实际由模拟器的 done 信号控制
        model.learn(total_timesteps=int(1e10), callback=eval_cb)
    finally:
        suffix = datetime.now().strftime("%Y%m%d_%H%M")
        model.save(os.path.join(MODEL_DIR, f"tea_final_{suffix}"))
        env.save(os.path.join(MODEL_DIR, f"tea_final_{suffix}_stats.pkl"))
        env.close()


if __name__ == '__main__':
    main()