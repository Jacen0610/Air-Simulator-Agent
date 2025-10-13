import time
import grpc
import numpy as np

# 导入由 protoc 生成的代码
import simulator_pb2
import simulator_pb2_grpc

class SingleAgentSimEnv:
    """
    一个封装了 gRPC 模拟器的单智能体环境，提供了类似 Gym 的接口。
    """

    def __init__(self, grpc_address="localhost:50051"):
        """
        初始化环境并连接到 gRPC 服务器。
        """
        channel = grpc.insecure_channel(grpc_address)
        self.stub = simulator_pb2_grpc.SimulatorStub(channel)
        self.agent_id = None
        print(f"连接到 gRPC 模拟器于 {grpc_address}...")

    def _parse_state(self, state_proto):
        """
        将从 gRPC 返回的单个 Protobuf 状态解析为 Python 对象。
        """
        obs_vec = np.array([
            float(state_proto.observation.has_message),
            float(state_proto.observation.primary_channel_busy),
            float(state_proto.observation.backup_channel_busy),
            state_proto.observation.pending_acks_count,
            state_proto.observation.outbound_queue_length,
            state_proto.observation.top_message_wait_time_seconds,
            float(state_proto.observation.is_retransmission),
        ], dtype=np.float32)

        reward = state_proto.reward
        done = state_proto.done

        return obs_vec, reward, done

    def reset(self):
        """
        重置环境并返回初始观测。
        """
        print("正在重置环境...")
        request = simulator_pb2.ResetRequest()
        response = self.stub.Reset(request)

        if not response.states:
            raise ValueError("环境重置后没有返回任何智能体状态。")
        
        self.agent_id = list(response.states.keys())[0]
        print(f"环境已重置。发现智能体: {self.agent_id}")

        initial_state = response.states[self.agent_id]
        observation, _, _ = self._parse_state(initial_state)
        
        return observation

    def step(self, action_type):
        """
        在环境中为单个智能体执行一步。
        [核心修改] 根据新的 proto 文件构建 Action 消息。

        Args:
            action_type (int): 要执行的动作类型 ID (0 for WAIT, 1 for SEND)。

        Returns:
            tuple: (observation, reward, done, info)
        """
        if self.agent_id is None:
            raise RuntimeError("必须在调用 step 之前先调用 reset()。")

        # [核心修改] 构建新的 Action 消息
        # 如果动作是 SEND，我们暂时硬编码 p_value=1.0
        p_value = 1.0 if action_type == 1 else 0.0
        action_msg = simulator_pb2.Action(type=action_type, p_value=p_value)

        step_request = simulator_pb2.StepRequest(
            actions={self.agent_id: action_msg}
        )

        response = self.stub.Step(step_request)
        
        agent_state = response.states[self.agent_id]
        observation, reward, done = self._parse_state(agent_state)

        info = {}

        return observation, reward, done, info
