# go_simulator_env.py
import grpc
import numpy as np
import collections
import time

# 导入由 proto 文件生成的 gRPC 客户端模块
# 确保 simulator_pb2.py 和 simulator_pb2_grpc.py 在 Python 路径中
import simulator_pb2
import simulator_pb2_grpc

class GoSimulatorEnv:
    """
    封装 Go 模拟器 gRPC 服务的强化学习环境。
    提供类似 OpenAI Gym 的 reset() 和 step() 接口。
    """
    def __init__(self, grpc_server_address='localhost:50051', sequence_length=10):
        self.grpc_server_address = grpc_server_address
        self.channel = None
        self.stub = None
        self.state_dim = 7
        self.action_dim = 2
        self.sequence_length = sequence_length
        self.observation_history = collections.deque(maxlen=sequence_length)
        # 连接将在第一次调用 reset() 时建立。

    def _connect_grpc(self):
        """
        [修正后] 建立或重新建立 gRPC 连接。
        这是一个对连接的“硬重置”。
        """
        # 如果存在旧的 channel，先关闭它。
        if self.channel:
            self.channel.close()

        print(f"正在尝试连接 gRPC 服务器: {self.grpc_server_address}...")
        # 创建一个新的 channel 和 stub。
        self.channel = grpc.insecure_channel(self.grpc_server_address)
        self.stub = simulator_pb2_grpc.SimulatorStub(self.channel)

        # 等待 channel 准备就绪，并设置一个超时。
        try:
            grpc.channel_ready_future(self.channel).result(timeout=10)
            print("gRPC 连接成功！")
        except grpc.FutureTimeoutError:
            # 如果超时，清理新创建的 channel 并抛出错误。
            self.channel.close()
            self.channel = None
            self.stub = None
            raise RuntimeError(
                f"无法连接到 gRPC 服务器 {self.grpc_server_address}。"
                "请确保 Go 模拟器正在运行。"
            )

    def _parse_observation(self, proto_obs: simulator_pb2.AgentObservation) -> np.ndarray:
        """将 protobuf 观测数据转换为 NumPy 数组。"""
        obs_vector = np.array([
            float(proto_obs.has_message),
            float(proto_obs.primary_channel_busy),
            float(proto_obs.backup_channel_busy),
            float(proto_obs.pending_acks_count),
            float(proto_obs.outbound_queue_length),
            proto_obs.top_message_wait_time_seconds,
            float(proto_obs.is_retransmission)
        ], dtype=np.float32)
        return obs_vector

    def _get_action_enum(self, action_int: int) -> simulator_pb2.Action:
        """将整数动作转换为 protobuf 枚举。"""
        if action_int == 0:
            return simulator_pb2.ACTION_WAIT
        elif action_int == 1:
            return simulator_pb2.ACTION_SEND
        else:
            raise ValueError(f"无效的动作整数: {action_int}")

    def reset(self) -> np.ndarray:
        """
        [修正后] 重置环境。现在会在每个 episode 开始时确保一个全新的连接。
        """
        # 总是在一个 episode 开始时尝试连接/重连。
        # 这使得它对 Go 服务器在 episode 之间重启具有鲁棒性。
        self._connect_grpc()

        print("正在重置模拟器环境...")
        try:
            response = self.stub.Reset(simulator_pb2.ResetRequest())
            initial_state = response.state
            initial_obs_vector = self._parse_observation(initial_state.observation)

            # 初始化观测历史
            self.observation_history.clear()
            for _ in range(self.sequence_length):
                self.observation_history.append(initial_obs_vector)

            print("环境重置成功，等待 Go 模拟器启动飞行计划...")
            time.sleep(2)

            return np.array(list(self.observation_history))
        except grpc.RpcError as e:
            print(f"gRPC Reset 调用失败: {e.code()} - {e.details()}")
            # 如果 Reset 失败，说明有严重问题，直接抛出异常。
            raise

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        [修正后] 执行一步。它依赖于 reset() 建立的连接，
        如果连接在 episode 中途断开，将会失败。
        """
        # 我们假设连接是正常的。如果不是，RpcError 会被捕获。
        if not self.stub:
             raise ConnectionError("gRPC 连接未建立。请先调用 reset()。")

        proto_action = self._get_action_enum(action)
        request = simulator_pb2.StepRequest(action=proto_action)

        try:
            response = self.stub.Step(request)
            agent_state = response.state

            obs_vector = self._parse_observation(agent_state.observation)
            reward = agent_state.reward
            done = agent_state.done

            self.observation_history.append(obs_vector)

            info = {}
            return np.array(list(self.observation_history)), reward, done, info
        except grpc.RpcError as e:
            print(f"gRPC Step 调用失败: {e.code()} - {e.details()}")
            # 如果 step 失败，通常意味着服务器在 episode 中途崩溃。
            # 最好的处理方式是让训练循环崩溃并由用户重启。
            raise

    def close(self):
        """关闭 gRPC 连接。"""
        if self.channel:
            self.channel.close()
            print("gRPC 连接已关闭。")
