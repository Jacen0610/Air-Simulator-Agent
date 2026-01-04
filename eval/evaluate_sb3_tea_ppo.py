import sys
import os
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import numpy as np
import time

# 导入 TEA 相关的环境和特征提取器
from env.tea_gym_env import TEAGymEnv
from agent.tea_feature_extractor import TEA_Extractor


# --- 新增：缝隙捕捉分析器 ---
class GapAnalyzer:
    def __init__(self, fps=320):
        self.ms_per_frame = 1000.0 / fps  # 约 3.125ms
        self.reset_stats()

    def reset_stats(self):
        self.delays = []  # 记录每次成功的反应延迟(帧数)
        self.gap_start_frame = -1
        self.total_gaps_found = 0
        self.captured_gaps = 0

    def update(self, last_busy, current_busy, action, current_frame):
        # 检测下降沿：从忙碌变为闲置
        if last_busy == 1 and current_busy == 0:
            self.gap_start_frame = current_frame
            self.total_gaps_found += 1

        # 如果在缝隙期间模型发包 (Action 1)
        if action == 1 and self.gap_start_frame != -1:
            delay = current_frame - self.gap_start_frame
            self.delays.append(delay)
            self.captured_gaps += 1
            self.gap_start_frame = -1  # 消耗掉这个缝隙信号
            return delay

        # 如果背景流量重新变忙，而模型还没发包，说明错过了这个缝隙
        if current_busy == 1 and self.gap_start_frame != -1:
            self.gap_start_frame = -1

        return None

    def get_report(self):
        avg_delay_frames = np.mean(self.delays) if self.delays else 0
        return {
            "avg_delay_ms": avg_delay_frames * self.ms_per_frame,
            "capture_rate": (self.captured_gaps / self.total_gaps_found * 100) if self.total_gaps_found > 0 else 0,
            "total_gaps": self.total_gaps_found
        }


# --- 原有的绘图函数保持不变 ---
def plot_evaluation_rewards(rewards: list, title: str, filename: str):
    # ... (保持你原来的代码不变) ...
    pass


def main():
    # --- 基础配置保持不变 ---
    parser = argparse.ArgumentParser(description='Evaluate SB3 TEA-PPO Agent')
    parser.add_argument('--grpc_port', type=str, default='50051', help='gRPC server port (default: 50051)')
    args = parser.parse_args()
    grpc_address = f'localhost:{args.grpc_port}'

    EVAL_EPISODES = 5
    SEQUENCE_LENGTH = 96
    DUMP_FREQUENCY = 800

    # 路径设置 (保持你原来的逻辑)
    MODEL_PATH = os.path.join(project_root, "SB3/models/models/tea_ppo_final.zip")
    STATS_PATH = os.path.join(project_root, "SB3/models/models/tea_ppo_vec_norm.pkl")
    # ... (路径检查逻辑略) ...

    # --- 1. 环境初始化 ---
    raw_env = TEAGymEnv(grpc_server_address=grpc_address, sequence_length=SEQUENCE_LENGTH)
    monitored_env = Monitor(raw_env)
    vec_env = DummyVecEnv([lambda: monitored_env])
    env = VecNormalize.load(STATS_PATH, vec_env)
    env.training = False
    env.norm_reward = False

    # --- 2. 加载模型 ---
    model = PPO.load(MODEL_PATH, env=env)

    # --- 3. 初始化分析器 ---
    analyzer = GapAnalyzer(fps=320)

    # --- 运行评估循环 ---
    print(f"\n开始深度评估...")
    obs = env.reset()
    episodes_completed = 0
    total_steps = 0

    last_is_busy = 0

    while episodes_completed < EVAL_EPISODES:
        total_steps += 1

        # 提取当前帧的 is_busy 状态 (假设索引1是is_busy)
        # obs 的形状是 (1, 96, 12)，取最后一帧 [-1] 的第二个特征 [1]
        current_is_busy = obs[0, -1, 1]

        # 预测动作
        action, _ = model.predict(obs, deterministic=True)

        # 分析缝隙捕捉情况
        delay_detected = analyzer.update(last_is_busy, current_is_busy, action[0], total_steps)
        last_is_busy = current_is_busy

        # 执行环境步
        obs, reward, done, info = env.step(action)

        if 'episode' in info[0]:
            episodes_completed += 1
            report = analyzer.get_report()

            print(f"Episode {episodes_completed} | Reward: {info[0]['episode']['r']:.2f} | "
                  f"缝隙捕捉率: {report['capture_rate']:.1f}% | "
                  f"平均延迟: {report['avg_delay_ms']:.2f} ms")

            # 记录到 TensorBoard
            model.logger.record("eval/gap_capture_rate", report['capture_rate'])
            model.logger.record("eval/reaction_delay_ms", report['avg_delay_ms'])
            model.logger.dump(step=total_steps)

            # 每个 episode 重置分析器，看单次表现
            # 如果想看全局平均，可以不在这里 reset
            analyzer.reset_stats()

    env.close()


if __name__ == '__main__':
    main()