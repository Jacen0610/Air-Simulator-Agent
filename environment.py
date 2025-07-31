# C:/workspace/python/Air-Simulator-Agent/environment.py
import grpc
import torch
from proto import simulator_pb2
from proto import simulator_pb2_grpc


class GoSimulatorEnv:
    """
    与 Go 模拟器进行 gRPC 通信的 Python 环境。
    """

    def __init__(self, host='localhost:50051'):
        """
        构造函数现在只负责建立连接，不进行 reset。
        """
        print("🔌 正在连接到 Go gRPC 服务器...")
        self.channel = grpc.insecure_channel(host)
        self.stub = simulator_pb2_grpc.SimulatorStub(self.channel)
        self.agent_ids = []  # agent_ids 将在第一次 reset 时被设置
        print("✅ 连接成功！")

    def reset(self):
        """
        重置环境，开始一个新的 episode。
        这是获取智能体列表和初始观测的唯一入口。
        """
        try:
            request = simulator_pb2.ResetRequest()
            response = self.stub.Reset(request)

            # 在第一次 reset 时，设置 agent_ids
            if not self.agent_ids:
                self.agent_ids = list(response.states.keys())

            observations = {
                agent_id: self._convert_observation(state.observation)
                for agent_id, state in response.states.items()
            }
            return observations
        except grpc.RpcError as e:
            print(f"❌ Reset gRPC 请求失败: {e.status()}")
            return None

    def step(self, actions):
        """
        在环境中执行一步。
        """
        # **[核心修正]** 直接将整数动作+1赋值。
        # gRPC库会自动处理整数到枚举的映射。
        proto_actions = {
            agent_id: action + 1
            for agent_id, action in actions.items()
        }

        try:
            request = simulator_pb2.StepRequest(actions=proto_actions)
            response = self.stub.Step(request)

            observations = {
                agent_id: self._convert_observation(state.observation)
                for agent_id, state in response.states.items()
            }
            rewards = {
                agent_id: state.reward
                for agent_id, state in response.states.items()
            }
            dones = {
                agent_id: state.done
                for agent_id, state in response.states.items()
            }

            return observations, rewards, dones, {}  # 返回一个空的 info 字典以匹配 Gym API
        except grpc.RpcError as e:
            print(f"❌ Step gRPC 请求失败: {e.status()}")
            # 在出错时返回一个合理的值，避免程序崩溃
            return None, None, None, None

    def _convert_observation(self, obs_proto):
        """
        将 protobuf 的 Observation 转换为 PyTorch Tensor。
        """
        # **[核心修改]** 按照 .proto 文件定义的顺序构建 6 维张量
        return torch.tensor([
            float(obs_proto.has_message),
            float(obs_proto.top_message_priority),
            float(obs_proto.primary_channel_busy),
            float(obs_proto.backup_channel_busy),
            float(obs_proto.pending_acks_count),  # <--- 新增
            float(obs_proto.outbound_queue_length)  # <--- 新增
        ], dtype=torch.float32)

    def close(self):
        """
        关闭 gRPC 连接。
        """
        if self.channel:
            self.channel.close()
            print("🔌 gRPC 连接已关闭。")