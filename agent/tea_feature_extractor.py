# 文件路径: agent/tea_feature_extractor.py
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TEA_Extractor(BaseFeaturesExtractor):
    """
    TEA (Temporal-Enhanced Attention) 特征提取器
    创新点：Spatial-Attention (过滤瞬时干扰) + Recurrent-Memory (平滑长程决策)
    """

    def __init__(self, observation_space, features_dim=256, embed_dim=128):
        # observation_space shape: (32, 12)
        super().__init__(observation_space, features_dim)
        seq_len, state_dim = observation_space.shape

        # 1. 特征投影层：将原始 12 维物理特征映射到高维嵌入空间
        self.projection = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )

        # 2. 空间注意力层：在 32 步窗口内扫描最重要的干扰特征
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(embed_dim)

        # 3. 递归记忆层：处理 40 分钟的长周期逻辑，减少无效抖动
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=features_dim, num_layers=1, batch_first=True)

    def forward(self, observations):
        # 1. 投影: (Batch, 32, 12) -> (Batch, 32, 128)
        x = self.projection(observations)

        # 2. 注意力扫描: 识别 32 步内的关键脉冲
        attn_out, _ = self.attention(x, x, x)
        x = self.attn_norm(x + attn_out)  # 残差连接

        # 3. 递归处理: 只有经过注意力过滤后的信号才会进入长程记忆
        # 我们取 LSTM 最后一个时间步的输出作为最终特征
        lstm_out, (h_n, c_n) = self.lstm(x)
        return h_n[-1]  # 输出形状: (Batch, 256)