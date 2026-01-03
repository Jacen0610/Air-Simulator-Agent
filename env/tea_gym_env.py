# 文件路径: env/tea_gym_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env.go_simulator_env import GoSimulatorEnv


class TEAGymEnv(gym.Env):
    def __init__(self, grpc_server_address='localhost:50051', sequence_length=96):
        super().__init__()
        self.sequence_length = sequence_length
        self.go_env = GoSimulatorEnv(
            grpc_server_address=grpc_server_address,
            sequence_length=sequence_length
        )
        self.action_space = spaces.Discrete(self.go_env.action_dim)

        # 显式声明 (32, 12) 形状
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.sequence_length, self.go_env.state_dim),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.go_env.reset()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        obs, reward, done, info = self.go_env.step(action)
        terminated = done
        truncated = info.get("time_out", False)
        return np.array(obs, dtype=np.float32), float(reward), terminated, truncated, info