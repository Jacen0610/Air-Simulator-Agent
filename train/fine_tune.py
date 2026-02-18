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
    parser.add_argument('--model_name', type=str, required=True, help="模型文件名 (例如: tea_ppo_final)")
    parser.add_argument('--port', type=int, default=50051)
    parser.add_argument('--episodes', type=int, default=5)
    args = parser.parse_args()

    # --- [改进] 智能模型路径解析 ---
    # 定义默认的模型存储目录
    DEFAULT_MODELS_DIR = os.path.join(project_root, "SB3", "models")
    
    # [新增] 自动补全 .zip 后缀
    input_name = args.model_name
    if not input_name.endswith(".zip"):
        input_name += ".zip"
    
    # 1. 尝试直接使用用户提供的路径 (处理用户可能输入了带路径的文件名)
    if os.path.exists(input_name):
        model_path = input_name
    # 2. 尝试在 SB3/models 下查找
    elif os.path.exists(os.path.join(DEFAULT_MODELS_DIR, input_name)):
        model_path = os.path.join(DEFAULT_MODELS_DIR, input_name)
    # 3. 尝试在 SB3/models/models 下查找 (处理可能的嵌套)
    elif os.path.exists(os.path.join(DEFAULT_MODELS_DIR, "models", input_name)):
        model_path = os.path.join(DEFAULT_MODELS_DIR, "models", input_name)
    else:
        raise FileNotFoundError(f"无法找到模型文件: {input_name}\n已搜索路径:\n - {os.path.abspath(input_name)}\n - {DEFAULT_MODELS_DIR}")

    # --- [改进] 自动推导 Stats 路径的稳健逻辑 ---
    model_dir = os.path.dirname(model_path)
    model_filename = os.path.basename(model_path)
    model_base_name = model_filename.replace(".zip", "")

    stats_path = None
    pkl_files_found = []

    # 策略 1: 寻找文件名包含模型基础名的 .pkl 文件
    try:
        # 确保目录存在且可读
        if os.path.isdir(model_dir):
            files_in_dir = os.listdir(model_dir)
            pkl_files_found = [f for f in files_in_dir if f.endswith(".pkl")]

            # 寻找最直接的匹配
            for pkl_file in pkl_files_found:
                if model_base_name in pkl_file:
                    stats_path = os.path.join(model_dir, pkl_file)
                    break
    except Exception as e:
        print(f"Warning: 无法扫描目录 '{model_dir}'。错误: {e}")

    # 策略 2: 如果没找到，但目录里只有一个 .pkl 文件，就用它
    if stats_path is None and len(pkl_files_found) == 1:
        stats_path = os.path.join(model_dir, pkl_files_found[0])

    # 策略 3: 如果新策略失败，尝试旧的、直接的替换逻辑
    if stats_path is None or not os.path.exists(stats_path):
        stats_path_alt = model_path.replace(".zip", ".pkl")
        if os.path.exists(stats_path_alt):
            stats_path = stats_path_alt

    # 最终检查
    if stats_path is None or not os.path.exists(stats_path):
        error_msg = f"错误: 未找到与 '{model_filename}' 配套的 Stats (.pkl) 文件。\n"
        error_msg += f"  - 搜索目录: '{model_dir}'\n"
        if pkl_files_found:
            error_msg += f"  - 在目录中找到了以下 .pkl 文件: {pkl_files_found}\n"
        else:
            error_msg += f"  - 在目录中未找到任何 .pkl 文件。\n"
        error_msg += "请确保 VecNormalize 的 .pkl 文件与模型 .zip 文件在同一目录下。"
        raise FileNotFoundError(error_msg)

    print(f"📦 匹配成功！\n模型: {model_path}\n参数: {stats_path}")

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
    
    # [修复] clip_range 必须是一个函数 (schedule)
    model.clip_range = ConstantSchedule(0.1)
    
    # [确认] 设置 target_kl
    model.target_kl = 0.003

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
            "clip_range": 0.1, # 记录数值即可
            "target_kl": model.target_kl, # 记录 target_kl
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