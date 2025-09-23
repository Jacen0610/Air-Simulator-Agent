import time

import grpc
import numpy as np

# 导入由 protoc 生成的代码
import simulator_pb2
import simulator_pb2_grpc


class MultiAgentSimEnv:
    """
    一个封装了 gRPC 模拟器的多智能体环境，提供了类似 Gym 的接口。
    """

    def __init__(self, grpc_address="localhost:50051"):
        """
        初始化环境并连接到 gRPC 服务器。

        Args:
            grpc_address (str): gRPC 服务器的地址。
        """
        channel = grpc.insecure_channel(grpc_address)
        self.stub = simulator_pb2_grpc.SimulatorStub(channel)
        self.agent_ids = []
        print(f"连接到 gRPC 模拟器于 {grpc_address}...")

    def _parse_states(self, states_proto):
        """
        将从 gRPC 返回的 Protobuf 状态解析为 Python 字典。
        """
        observations = {}
        rewards = {}
        dones = {}

        for agent_id, state in states_proto.items():
            # 将观测数据转换为一个 numpy 向量
            obs_vec = np.array([
                float(state.observation.has_message),
                state.observation.top_message_priority,
                float(state.observation.primary_channel_busy),
                float(state.observation.backup_channel_busy),
                state.observation.pending_acks_count,
                state.observation.outbound_queue_length,
            ], dtype=np.float32)

            observations[agent_id] = obs_vec
            rewards[agent_id] = state.reward
            dones[agent_id] = state.done

        return observations, rewards, dones

    def reset(self):
        """
        重置环境并返回初始观测。
        """
        print("正在重置环境...")
        request = simulator_pb2.ResetRequest()
        response = self.stub.Reset(request)

        # 存储 agent_ids，因为它们在整个 episode 中是固定的
        self.agent_ids = list(response.states.keys())
        print(f"环境已重置。发现智能体: {self.agent_ids}")

        observations, _, _ = self._parse_states(response.states)
        return observations

    def step(self, actions):
        """
        在环境中执行一步。

        Args:
            actions (dict): 一个从 agent_id 到 action_id (int) 的字典。

        Returns:
            tuple: (observations, rewards, dones, infos)
        """
        # 将 Python 字典转换为 Protobuf 的 map
        step_request = simulator_pb2.StepRequest(
            actions={
                agent_id: simulator_pb2.Action.Value(f'ACTION_WAIT') if action_id == 1
                else simulator_pb2.Action.Value(f'ACTION_SEND_PRIMARY') if action_id == 2
                else simulator_pb2.Action.Value(f'ACTION_SEND_BACKUP')
                for agent_id, action_id in actions.items()
            }
        )

        response = self.stub.Step(step_request)
        observations, rewards, dones = self._parse_states(response.states)

        # 在多智能体设置中，通常有一个全局的 done 标志
        # 这里我们假设如果任何一个 agent 完成，整个 episode 就完成了
        all_done = any(dones.values())

        # infos 字典可以用来传递调试信息，这里我们留空
        infos = {}

        return observations, rewards, dones, all_done, infos