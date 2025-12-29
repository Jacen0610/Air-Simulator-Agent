# static_agent.py
import numpy as np
import random


class CSMAAgent:
    """
    一个遵循固定 P-坚持 CSMA 策略的静态智能体。
    它不进行任何学习，仅用于作为基准对比。
    """

    def __init__(self, p_value=0.05):
        """
        初始化 CSMA 智能体。

        :param p_value: P-坚持概率值。
        """
        self.p_value = p_value
        print(f"--- Initialized Static CSMA Agent with p = {self.p_value} ---")

    def select_action(self, observation_history):
        """
        [2024-05-22 更新] 根据 CSMA 规则选择动作。
        此版本适配了新的 protobuf 状态定义。

        :param observation_history: 包含历史状态的序列。
        :return: 动作 (0 for WAIT, 1 for SEND)
        """
        # 对于这个简单的策略，我们只关心当前时刻的状态
        current_state = observation_history[-1]

        # [核心修改] 从状态向量中提取所需信息，使用新的索引约定
        # 根据 simulator.proto:
        is_channel_busy = current_state[1] > 0.5
        has_data_to_send = current_state[0] > 0.5

        # 如果没有消息要发送，必须等待
        if not has_data_to_send:
            return 0  # WAIT

        # 如果有消息要发送，则执行 P-坚持 CSMA 逻辑
        if is_channel_busy:
            # 信道忙，必须等待
            return 0  # WAIT
        else:
            # 信道空闲，以概率 p 决定发送
            if random.random() < self.p_value:
                return 1  # SEND
            else:
                return 0  # WAIT

    # 以下方法是为了让它能被评估脚本调用，但它们不做任何事
    def load_model(self, path):
        """这个伪方法让静态智能体能被使用相同接口的评估脚本调用。"""
        print(f"Static agent does not load models. Using fixed policy with p={self.p_value}.")
        pass
