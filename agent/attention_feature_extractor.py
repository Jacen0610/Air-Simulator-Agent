import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AviationTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128, embed_dim: int = 128):
        """
        针对12维异步状态空间优化的特征提取器
        :param observation_space: 输入形状应为 (seq_len, 12)
        :param features_dim: 输出给 PPO Actor/Critic 的维度 (推荐 128 或 256)
        :param embed_dim: Transformer 内部投影维度 (推荐 128，对齐 4 头注意力)
        """
        super().__init__(observation_space, features_dim)

        # 获取输入维度
        # 此时输入 shape 应为 (Batch, Seq_Len, 12)
        self.seq_len = observation_space.shape[0]
        self.state_dim = observation_space.shape[1]
        self.embed_dim = embed_dim
        num_heads = 4  # 128 维度平分给 4 个头，每个头处理 32 维，计算效率最高

        # 1. 初始投影层 (Input Embedding)
        # 将 12 维物理量投影到 128 维语义空间
        self.input_projection = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU()
        )

        # 2. 位置嵌入 (Positional Embedding)
        # 学习序列中每个时刻的时间顺序关系
        self.pos_embedding = nn.Parameter(th.randn(1, self.seq_len, self.embed_dim))

        # 3. Transformer Encoder 层 (两层结构)
        # 相比单层 Attention，标准 Encoder Block 包含 FFN 层，能更好模拟非线性信道规律
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=self.embed_dim * 2,  # 隐含层扩大
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 4. 最终输出层 (Flatten -> Output)
        # 将 (Batch, Seq_Len, Embed_Dim) 转换为 (Batch, Features_Dim)
        self.output_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.embed_dim * self.seq_len, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # 1. 特征投影 (B, L, 12) -> (B, L, 128)
        x = self.input_projection(observations)

        # 2. 注入位置信息
        x = x + self.pos_embedding

        # 3. Transformer 特征提取 (自注意力计算)
        # 自动学习 1s/0.1s ratio 与 wait_time、dt_step 之间的因果关系
        x = self.transformer_encoder(x)

        # 4. 输出给 PPO
        return self.output_head(x)