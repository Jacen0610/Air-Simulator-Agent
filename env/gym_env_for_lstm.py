import gymnasium as gym
from gymnasium import spaces
import numpy as np

# 导入您现有的、返回观测序列的底层 Go 环境
from env.go_simulator_env import GoSimulatorEnv

class GymEnvForLSTM(gym.Env):
    """
    一个新的 Gymnasium 环境封装器，专门为 SB3 的内置循环策略 (如 MlpLstmPolicy) 设计。

    它将 GoSimulatorEnv 返回的二维观测序列 (sequence_length, features) 
    转换为 SB3 循环策略期望的一维观测 (features,)。
    """
    metadata = {'render_modes': []}

    def __init__(self, grpc_server_address='localhost:50051', sequence_length=96):
        """
        初始化环境。
        """
        super().__init__()
        self.sequence_length = sequence_length

        # 内部实例化底层的 Go 环境。
        # 注意：我们仍然需要它内部维护一个序列，但我们只取最新的观测。
        self.go_env = GoSimulatorEnv(
            grpc_server_address=grpc_server_address,
            sequence_length=sequence_length  # 这个值需要与底层环境的默认值匹配
        )

        # 定义动作空间 (保持不变)
        self.action_space = spaces.Discrete(self.go_env.action_dim)

        # [核心修改] 定义观测空间为一维向量
        # SB3 的 MlpLstmPolicy 会自己处理时间序列，它期望每一步只接收当前的观测
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.go_env.state_dim,),  # 形状是 (12,) 而不是 (10, 12)
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        重置环境并返回单个初始观测。
        """
        super().reset(seed=seed)

        # 底层环境的 reset 返回一个完整的历史序列
        initial_observation_sequence = self.go_env.reset()

        # [核心修改] 我们只返回这个序列中的最后一个（即最新的）观测
        latest_observation = initial_observation_sequence[-1]

        info = {}
        return latest_observation, info

    def step(self, action):
        """
        在环境中执行一步，并返回单个最新观测。
        """
        # 底层环境的 step 返回下一个完整的历史序列
        next_observation_sequence, reward, done, info = self.go_env.step(action)

        # [核心修改] 我们只返回新序列中的最后一个（即最新的）观测
        latest_observation = next_observation_sequence[-1]

        terminated = done
        truncated = False # 根据 SB3 的建议，通常只在因为时间限制等外部因素结束时才为 True

        return latest_observation, reward, terminated, truncated, info

    def close(self):
        """
        关闭环境并清理资源。
        """
        self.go_env.close()

    def render(self):
        """
        空操作。
        """
        pass

if __name__ == '__main__':
    # 提供一个简单的示例，展示如何使用这个新的 Gym 环境
    print("创建一个为 LSTM 策略优化的 Gym 环境实例...")
    # 注意：运行此示例前，请确保 Go 模拟器正在运行
    env = GymEnvForLSTM()

    print("重置环境...")
    obs, info = env.reset()

    print(f"初始观测的形状 (一维): {obs.shape}")
    print(f"观测空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")

    print("\n执行一个随机动作...")
    random_action = env.action_space.sample()
    print(f"选择的随机动作: {random_action}")

    obs, reward, terminated, truncated, info = env.step(random_action)

    print(f"下一步的观测形状 (一维): {obs.shape}")
    print(f"获得的奖励: {reward}")
    print(f"是否终止: {terminated}")

    env.close()
    print("\n环境已关闭。")
