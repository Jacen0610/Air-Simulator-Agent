import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from env.go_simulator_env import GoSimulatorEnv


class TEAGymEnv(gym.Env):
    def __init__(self, grpc_server_address='localhost:50051', sequence_length=96, history_size=16):
        super().__init__()
        self.sequence_length = sequence_length
        self.history_size = history_size
        self.go_env = GoSimulatorEnv(
            grpc_server_address=grpc_server_address,
            sequence_length=sequence_length
        )

        # 原始维度 12 + 16位原始0/1序列 = 28
        self.total_state_dim = self.go_env.state_dim + self.history_size

        self.action_space = spaces.Discrete(self.go_env.action_dim)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.sequence_length, self.total_state_dim),
            dtype=np.float32
        )

        # 用于记录每一帧的 busy 历史
        self.busy_queue = deque([0.0] * self.history_size, maxlen=self.history_size)

    def _inject_history(self, raw_obs):
        # raw_obs shape: (96, 12)
        # 我们需要为 96 帧里的每一帧都拼上它的“过去16帧历史”
        # 注意：这里为了简化，我们假设 96 帧内的时间是连续的
        refined_obs = []
        temp_queue = deque(list(self.busy_queue), maxlen=self.history_size)

        for i in range(len(raw_obs)):
            current_busy = raw_obs[i, 1]  # 假设 index 1 是 is_busy
            temp_queue.append(current_busy)
            # 拼接原始 12 维 + 16 维历史
            refined_frame = np.concatenate([raw_obs[i], list(temp_queue)])
            refined_obs.append(refined_frame)

        # 更新全局队列，供下一 step 使用
        self.busy_queue.append(raw_obs[-1, 1])
        return np.array(refined_obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.busy_queue = deque([0.0] * self.history_size, maxlen=self.history_size)
        obs = self.go_env.reset()
        refined_obs = self._inject_history(obs)
        return refined_obs, {}

    def step(self, action):
        obs, reward, done, info = self.go_env.step(action)

        # 1. 注入原始 0/1 序列特征
        refined_obs = self._inject_history(obs)

        terminated = done
        truncated = info.get("time_out", False)

        if np.isnan(refined_obs).any() or np.isinf(refined_obs).any():
            print("Found NaN in refined_obs from Go!")
            refined_obs = np.nan_to_num(refined_obs, 0.0)

        if np.isnan(reward) or np.isinf(reward):
            print("Found NaN in Reward!")
            reward = 0.0

        return refined_obs, float(reward), terminated, truncated, info