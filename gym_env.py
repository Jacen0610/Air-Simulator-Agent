import gymnasium as gym
from gymnasium import spaces
import numpy as np
from go_simulator_env import GoSimulatorEnv

class GymEnv(gym.Env):
    """
    一个封装了 GoSimulatorEnv 的 Gymnasium 环境，使其符合标准的 RL 接口。
    """
    metadata = {'render_modes': []}

    def __init__(self, grpc_server_address='localhost:50051', sequence_length=10):
        """
        初始化 Gymnasium 环境。

        :param grpc_server_address: gRPC 服务器的地址。
        :param sequence_length: 观测历史的长度。
        """
        super().__init__()

        # 内部实例化底层环境
        self.go_env = GoSimulatorEnv(
            grpc_server_address=grpc_server_address,
            sequence_length=sequence_length
        )

        # 定义动作空间
        # 根据 .proto 文件，我们有两个离散动作: ACTION_WAIT (0) 和 ACTION_SEND (1)
        self.action_space = spaces.Discrete(self.go_env.action_dim)

        # 定义观测空间
        # GoSimulatorEnv 返回一个形状为 (sequence_length, 7) 的 numpy 数组
        # 值是浮点数，理论上没有严格的上下界，所以使用 -inf 到 inf
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(sequence_length, self.go_env.state_dim),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        重置环境并返回初始观测。
        """
        # Gymnasium 的 reset 支持可选的 seed 和 options，我们暂时用不到
        super().reset(seed=seed)

        # 调用底层环境的 reset
        initial_observation = self.go_env.reset()

        # Gymnasium 的 reset 返回 (observation, info)
        info = {}
        return initial_observation, info

    def step(self, action):
        """
        在环境中执行一步。
        """
        # 调用底层环境的 step
        observation, reward, done, info = self.go_env.step(action)

        # Gymnasium 的 step 返回五元组: obs, reward, terminated, truncated, info
        # 在我们的场景中，'done' 同时代表 terminated 和 truncated
        terminated = done
        truncated = done

        return observation, reward, terminated, truncated, info

    def close(self):
        """
        关闭环境并清理资源。
        """
        self.go_env.close()

    def render(self):
        """
        由于这是一个模拟器环境，没有可视化界面，所以 render 是一个空操作。
        """
        pass

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