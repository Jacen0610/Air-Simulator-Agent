import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env.go_simulator_env import GoSimulatorEnv


class GymEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, grpc_server_address='localhost:50051', sequence_length=32):  # 建议对齐 Transformer 的 32
        super().__init__()

        # 1. 明确 sequence_length
        self.sequence_length = sequence_length

        self.go_env = GoSimulatorEnv(
            grpc_server_address=grpc_server_address,
            sequence_length=sequence_length
        )

        self.action_space = spaces.Discrete(self.go_env.action_dim)

        # 2. 观测空间声明（确保 dtype 严格为 float32）
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.sequence_length, self.go_env.state_dim),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        # 传递 seed 给底层随机数生成器（如果有的话）
        super().reset(seed=seed)

        # 如果 GoSimulatorEnv 支持 seed，建议传入
        # initial_observation = self.go_env.reset(seed=seed)
        initial_observation = self.go_env.reset()

        return np.array(initial_observation, dtype=np.float32), {}

    def step(self, action):
        observation, reward, done, info = self.go_env.step(action)

        # 3. 奖励缩放（在这里做最后的保险）
        # 如果你在 Go 端改了 Log，这里可以除以 10 或者不除
        # reward = reward / 10.0

        # 4. 区分 Terminated 和 Truncated
        # 假设 info 中包含了步数信息，或者由 Python 端计数
        terminated = done
        truncated = info.get("time_out", False)  # 确保 Go 端能识别 40 分钟到期的情形

        return np.array(observation, dtype=np.float32), float(reward), terminated, truncated, info

if __name__ == '__main__':
    # 提供一个简单的示例，展示如何使用这个 Gym 环境
    print("创建一个 Gym 环境实例...")
    # 注意：运行此示例前，请确保 Go 模拟器正在运行
    env = GymEnv()

    print("重置环境...")
    obs, info = env.reset()

    print(f"初始观测的形状: {obs.shape}")
    print(f"观测空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")

    print("\n执行一个随机动作...")
    random_action = env.action_space.sample()
    print(f"选择的随机动作: {random_action}")

    # 执行一步
    obs, reward, terminated, truncated, info = env.step(random_action)

    print(f"下一步的观测形状: {obs.shape}")
    print(f"获得的奖励: {reward}")
    print(f"是否终止: {terminated}")

    # 关闭环境
    env.close()
    print("\n环境已关闭。")
