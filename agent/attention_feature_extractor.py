import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AviationAttentionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128, embed_dim: int = 64):
        """
        修改点说明：
        1. 增加 embed_dim：将原始 8 维状态投影到更高维，利于 Multi-head Attention 捕捉复杂关系。
        2. 结构优化：先进行特征嵌入，再进行位置编码。
        """
        super().__init__(observation_space, features_dim)

        self.seq_len = observation_space.shape[0]
        self.state_dim = observation_space.shape[1]
        self.embed_dim = embed_dim  # 推荐设为 32 或 64

        # [修改 1] 输入特征投影层：将 8 维特征映射到 embed_dim
        # 理由：原始特征如“等待秒数”和“忙碌标志”量级差异大，投影后更利于注意力机制收敛
        self.input_embedding = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU()
        )

        # [修改 2] 位置编码维度适配 embed_dim
        self.pos_embedding = nn.Parameter(th.randn(1, self.seq_len, self.embed_dim))

        # [修改 3] 多头注意力：使用 embed_dim 进行计算
        num_heads = 4  # 增加头数以同时监控微观和宏观规律
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.layernorm = nn.LayerNorm(self.embed_dim)

        # [修改 4] 最终 MLP 的输入维度适配 embed_dim
        self.flat_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.embed_dim * self.seq_len, features_dim),
            nn.ReLU()
        )

        self.last_attn_weights = None

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # observations shape: (batch_size, seq_len, state_dim)

        # [修改 5] 先投影再加位置编码
        x = self.input_embedding(observations)  # (B, L, embed_dim)
        x = x + self.pos_embedding

        # [修改 6] 获取权重用于科研分析
        # 在航空场景中，我们最关心序列最后一个 step 对历史的关注度
        attn_output, attn_weights = self.attention(x, x, x, need_weights=True)

        if not self.training:
            # 提取最后一行权重：即“当前决策时刻”对“历史所有时刻”的关注度
            # 形状通常为 (batch, seq_len, seq_len)，我们取最后一个 query
            self.last_attn_weights = attn_weights.detach().cpu().numpy()[0, -1, :]

        x = self.layernorm(x + attn_output)
        return self.flat_out(x)