import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TEA_Extractor_V2(BaseFeaturesExtractor):
    """
    改进版 TEA 架构：增加“实时特征直连”通道
    - 解决 Attention 全局平滑导致的决策迟滞问题。
    - 保持低无效动作的同时，找回那 500ms 的反应速度。
    """
    def __init__(self, observation_space, features_dim=256, embed_dim=128):
        # 注意：为了容纳直连特征，我们调整内部隐向量维度
        super().__init__(observation_space, features_dim)
        seq_len, state_dim = observation_space.shape  # (96, 12) 或 (32, 12)

        # 1. 基础投影层
        self.projection = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # 2. 瞬时扫描仪 (Attention) - 保持不变，负责模式识别
        self.instant_scanner = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=2,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(embed_dim)

        # 3. 特征演进器 (LSTM) - 负责长程稳定性
        # 将 hidden_size 设为 features_dim 的一部分，为直连特征留出空间
        self.lstm_hidden_dim = features_dim - 64  # 留出 64 维给实时特征
        self.long_term_memory = nn.LSTM(
            input_size=embed_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # 4. 【新增】实时特征投影头
        # 专门处理当前最后一帧的原始观察值
        self.latest_frame_projector = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU()
        )

        # 5. 【新增】特征融合层
        # 将 LSTM 的长程判断和最新帧的瞬间判断进行最终融合
        self.fusion_layer = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU()
        )

    def forward(self, observations):
        # Step 1: 提取原始输入中最新的一帧 (Batch, 12)
        # 这是物理世界最实时的信号，不经过任何时序平滑
        latest_raw_obs = observations[:, -1, :]
        latest_feature = self.latest_frame_projector(latest_raw_obs) # (Batch, 64)

        # Step 2: 投影到嵌入空间 (用于时序处理)
        x = self.projection(observations)

        # Step 3: Attention 聚焦 (找出模式，过滤干扰)
        scanner_out, _ = self.instant_scanner(x, x, x)
        refined_signals = self.attn_norm(x + scanner_out)

        # Step 4: LSTM 记忆 (累积长程决策惯性)
        _, (h_n, _) = self.long_term_memory(refined_signals)
        lstm_out = h_n[-1] # (Batch, lstm_hidden_dim)

        # Step 5: 【关键核心】特征拼接 (Concatenation)
        # 将 LSTM 提供的“大局观”和最新帧提供的“瞬间变绿灯信号”强行拼接
        combined = torch.cat([lstm_out, latest_feature], dim=-1) # (Batch, 256)

        # Step 6: 融合输出
        return self.fusion_layer(combined)