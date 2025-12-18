# go_simulator_env.py
import grpc
import numpy as np
from collections import deque
import time

# 导入由 proto 文件生成的 gRPC 客户端模块
# 确保 simulator_pb2.py 和 simulator_pb2_grpc.py 在 Python 路径中
import proto.simulator_pb2 as simulator_pb2
import proto.simulator_pb2_grpc as simulator_pb2_grpc

class GoSimulatorEnv:
    """
    [2024-05-24 更新] 封装 Go 模拟器 gRPC 服务的强化学习环境。
    此版本适配了包含8个特征的 protobuf 定义。
    """
    def __init__(self, grpc_server_address='localhost:50051', sequence_length=10):
        self.grpc_server_address = grpc_server_address
        self.channel = None
        self.stub = None
        
        # [核心修改] 状态维度现在与 proto 文件中的 AgentObservation 字段数量完全对应
        self.state_dim = 8 
        self.action_dim = 2
        
        self.sequence_length = sequence_length
        # 这个队列现在存储的是从 Go 直接收到的、包含所有特征的观测向量
        self.observation_history = deque(maxlen=self.sequence_length)
        
        # 连接将在第一次调用 reset() 时建立

    def _connect_grpc(self):
        """建立或重新建立 gRPC 连接。"""
        if self.channel:
            self.channel.close()

        print(f"正在尝试连接 gRPC 服务器: {self.grpc_server_address}...")
        self.channel = grpc.insecure_channel(self.grpc_server_address)
        self.stub = simulator_pb2_grpc.SimulatorStub(self.channel)

        try:
            grpc.channel_ready_future(self.channel).result(timeout=10)
            print("gRPC 连接成功！")
        except grpc.FutureTimeoutError:
            self.channel.close()
            self.channel = None
            self.stub = None
            raise RuntimeError(
                f"无法连接到 gRPC 服务器 {self.grpc_server_address}。"
                "请确保 Go 模拟器正在运行。"
            )

    def _parse_observation(self, proto_obs: simulator_pb2.AgentObservation) -> np.ndarray:
        """
        [核心修改] 将 protobuf 观测数据转换为 NumPy 数组。
        顺序与 simulator.proto 文件中的定义严格一致。
        """
        obs_vector = np.array([
            proto_obs.is_channel_busy,
            proto_obs.has_data_to_send,
            proto_obs.outbound_queue_length,
            proto_obs.top_message_wait_time_seconds,
            proto_obs.consecutive_idle_steps,
            proto_obs.last_send_caused_collision,
            proto_obs.steps_since_last_collision,
            proto_obs.channel_busy_ratio,
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
        """重置环境，并在每个 episode 开始时确保一个全新的连接。"""
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
            time.sleep(2) # 保留这个等待，可能有助于模拟器完全初始化

            return np.array(list(self.observation_history))
        except grpc.RpcError as e:
            print(f"gRPC Reset 调用失败: {e.code()} - {e.details()}")
            raise

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """执行一步，并返回从 Go 模拟器收到的新状态。"""
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
            raise

    def close(self):
        """关闭 gRPC 连接。"""
        if self.channel:
            self.channel.close()
            print("gRPC 连接已关闭。")
