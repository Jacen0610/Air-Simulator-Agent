import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class AviationAttentionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        # observation_space 应该是 (sequence_length, state_dim)
        super().__init__(observation_space, features_dim)
        
        # 从 observation_space 获取序列长度和状态维度
        self.seq_len = observation_space.shape[0]
        self.state_dim = observation_space.shape[1]

        # 1. 可学习的位置编码 (Positional Encoding)
        # 形状: (1, seq_len, state_dim)
        self.pos_embedding = nn.Parameter(th.randn(1, self.seq_len, self.state_dim))

        # 2. 多头注意力层 (Multi-Head Attention)
        # embed_dim 必须能被 num_heads 整除
        num_heads = 2
        if self.state_dim % num_heads != 0:
            raise ValueError(f"state_dim ({self.state_dim}) must be divisible by num_heads ({num_heads})")
            
        self.attention = nn.MultiheadAttention(
            embed_dim=self.state_dim, 
            num_heads=num_heads, 
            batch_first=True
        )

        # 3. 规范化层
        self.layernorm = nn.LayerNorm(self.state_dim)

        # 4. 最终特征输出层 (MLP)
        # 输入维度是展平后的 (seq_len * state_dim)
        self.flat_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.state_dim * self.seq_len, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # observations shape: (batch_size, seq_len, state_dim)
        
        # 注入位置信息
        x = observations + self.pos_embedding
        
        # 注意力计算: Self-Attention
        # attn_output 捕捉了序列内部的依赖关系
        # x, x, x 分别作为 Query, Key, Value
        attn_output, _ = self.attention(x, x, x)
        
        # 残差连接与规范化
        x = self.layernorm(x + attn_output)
        
        # 展平并通过最终的 MLP
        return self.flat_out(x)
