import warnings

warnings.filterwarnings("ignore")
import os, sys, argparse
from datetime import datetime
import torch as th
import wandb
from wandb.integration.sb3 import WandbCallback

# --- 动态添加项目根目录 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import ConstantSchedule

from agent.tea_feature_extractor import TEA_Extractor_V3
from env.tea_gym_env import TEAGymEnv


class FineTuneCallback(BaseCallback):
    def __init__(self, total_episodes, save_path):
        super().__init__()
        self.total_episodes = total_episodes
        self.save_path = save_path
        self.episode_count = 0
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            self.episode_count += 1
            timestamp = datetime.now().strftime("%H%M")
            # 保持一致的文件名规则
            base_name = f"ft_ep{self.episode_count}_{timestamp}"
            self.model.save(os.path.join(self.save_path, f"{base_name}.zip"))
            self.training_env.save(os.path.join(self.save_path, f"stats_{base_name}.pkl"))

            print(f">>> [FT] 轮次 {self.episode_count}/{self.total_episodes} 保存成功")

            if self.episode_count >= self.total_episodes:
                return False
        return True


def main():
    parser = argparse.ArgumentParser()
    # 只需要传入模型路径
    parser.add_argument('--model_name', type=str, required=True, help="模型路径 (.zip)")
    parser.add_argument('--port', type=int, default=50051)
    parser.add_argument('--episodes', type=int, default=5)
    args = parser.parse_args()

    # --- 自动推导 Stats 路径 ---
    # 假设你的 stats 命名规则是 stats_epXX_XXXX.pkl 或 tea_epXX_XXXX_stats.pkl
    # 这里我们优先寻找与模型文件名对应的 .pkl 文件
    model_path = args.model_name
    model_dir = os.path.dirname(model_path)
    model_filename = os.path.basename(model_path)

    # 尝试常见的命名映射逻辑：
    # 1. 直接替换后缀 2. 处理 stats_ 前缀
    stats_path = model_path.replace(".zip", ".pkl")
    if not os.path.exists(stats_path):
        # 针对你提到的 stats 命名可能略有不同的逻辑尝试
        potential_name = model_filename.replace("tea_", "stats_").replace(".zip", ".pkl")
        stats_path = os.path.join(model_dir, potential_name)

    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"未找到配套的 Stats 文件: {stats_path}\n请确保 .pkl 文件与 .zip 在同一目录下且名字匹配。")

    print(f"📦 匹配成功！\n模型: {model_filename}\n参数: {os.path.basename(stats_path)}")

    # --- 环境与模型配置 ---
    # 修改保存路径到 SB3/models/finetune_precision_results
    SAVE_DIR = os.path.join(project_root, "SB3", "models", "finetune_precision_results")

    raw_env = make_vec_env(lambda: TEAGymEnv(f'localhost:{args.port}', 96), n_envs=1)
    env = VecNormalize.load(stats_path, raw_env)

    # 微调核心：加载模型并覆盖高精度参数
    # 注意：为了让 WandbCallback 能够记录 Tensorboard 数据，我们需要指定 tensorboard_log
    # 虽然我们主要看 W&B，但 sync_tensorboard=True 依赖于此
    model = PPO.load(model_path, env=env, device="cuda" if th.cuda.is_available() else "cpu", tensorboard_log=f"./runs/{model_filename}_ft")

    # 手术级去噪参数
    new_lr = 5e-6
    model.learning_rate = new_lr
    model.lr_schedule = ConstantSchedule(new_lr)  # 锁定学习率
    model.ent_coef = 0.001
    model.gae_lambda = 0.98
    model.clip_range = 0.1

    # 强制更新优化器参数组
    for param_group in model.policy.optimizer.param_groups:
        param_group['lr'] = new_lr

    # --- W&B 初始化 ---
    run_name = f"FT_{model_filename}_{datetime.now().strftime('%m%d_%H%M')}"
    wandb.init(
        project="Air-Simulator-FineTune",
        name=run_name,
        config={
            "learning_rate": model.learning_rate,
            "ent_coef": model.ent_coef,
            "clip_range": model.clip_range,
            "target_kl": model.target_kl,
            "base_model": model_filename,
            "episodes": args.episodes,
            "port": args.port
        },
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    cb = FineTuneCallback(total_episodes=args.episodes, save_path=SAVE_DIR)
    wandb_cb = WandbCallback(
        gradient_save_freq=0,  # 不保存梯度直方图以节省空间，需要可改为 100
        model_save_path=None,  # 模型保存由 FineTuneCallback 处理
        verbose=2
    )

    print(f"\n🚀 开始精准微调 (5e-6 LR + 0.001 Ent)...")
    print(f"📂 结果将保存至: {SAVE_DIR}")
    print(f"📊 W&B Run: {run_name}")

    try:
        # 组合回调函数
        model.learn(total_timesteps=int(1e9), callback=[cb, wandb_cb])
    finally:
        env.close()
        wandb.finish()


if __name__ == '__main__':
    main()