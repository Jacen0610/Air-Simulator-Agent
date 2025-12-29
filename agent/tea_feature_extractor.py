import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TEA_Extractor(BaseFeaturesExtractor):
    """
    TEA 架构：瓶颈压缩 (Attention) + 特征演进 (LSTM)
    直觉逻辑：
    - Attention 在 32 步 (70ms) 窗口内寻找高密度背景下的“可发送空隙”。
    - LSTM 记住 90s 维度的背景模式，提供长程决策惯性，压制无效动作。
    """
    def __init__(self, observation_space, features_dim=256, embed_dim=128):
        super().__init__(observation_space, features_dim)
        seq_len, state_dim = observation_space.shape  # (32, 12)

        # 1. 瓶颈压缩前置：特征投影
        self.projection = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # 2. 瞬时扫描仪 (Attention)：专注 70ms 内的空隙识别
        # 采用 2 个 Head 以兼顾“干扰强度”和“信道占用”两个关键维度的对比
        self.instant_scanner = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=2,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(embed_dim)

        # 3. 特征演进器 (LSTM)：记忆 90s 背景模式演变
        # 负责处理经 Attention 提纯后的信号，确保长程稳定性
        self.long_term_memory = nn.LSTM(
            input_size=embed_dim,
            hidden_size=features_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, observations):
        # Step 1: 投影到嵌入空间 (Batch, 32, 128)
        x = self.projection(observations)

        # Step 2: 瓶颈压缩 - Attention 聚焦 (找出空隙)
        # 这里的 scanner_out 实际上是 32 步中每一步根据上下文加权后的结果
        scanner_out, _ = self.instant_scanner(x, x, x)
        refined_signals = self.attn_norm(x + scanner_out)

        # Step 3: 特征演进 - LSTM 记忆
        # 将提纯后的 32 步信号喂给 LSTM
        # LSTM 的 Hidden State 会跨越步数累积，感知 90s 背景模式
        _, (h_n, _) = self.long_term_memory(refined_signals)

        # 输出最后一步的隐状态 (Batch, 256)
        return h_n[-1]