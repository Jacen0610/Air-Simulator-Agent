import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TEA_Extractor_V3(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, embed_dim=128):
        super().__init__(observation_space, features_dim)
        seq_len, state_dim = observation_space.shape

        # 1. 位置编码 (加入可学习的位置向量)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim))

        # 2. 基础投影层
        self.projection = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # 3. 瞬时扫描仪 (增强注意力机制)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.instant_scanner = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,  # 增加头数，更好地捕捉复杂的周期背景
            batch_first=True,
            dropout=0.05  # 略微加入噪声，增加泛化力
        )

        # 4. 特征演进器 (更精简的隐层，针对短周期优化)
        # 建议调小维度，强迫模型只记关键特征，减少无效动作
        self.lstm_hidden_dim = 128
        self.long_term_memory = nn.LSTM(
            input_size=embed_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # 5. 实时特征直连
        self.latest_frame_projector = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )

        # 6. 最终融合层 (加入残差思想)
        fusion_input_dim = self.lstm_hidden_dim + 64 + embed_dim  # 增加了Attention池化特征
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU(),
            nn.Linear(features_dim, features_dim)
        )

        self._apply_orthogonal_init()

    def _apply_orthogonal_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu') * 0.5)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param.data, gain=0.1)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)

    def forward(self, observations):
        # 数据清理
        observations = torch.nan_to_num(observations, nan=0.0, posinf=1.0, neginf=0.0)

        # A. 投影 + 位置编码
        x = self.projection(observations)
        x = x + self.pos_embedding  # 关键：给 Attention 加上时间感

        # B. Attention 扫描
        x_norm = self.attn_norm(x)
        attn_out, _ = self.instant_scanner(x_norm, x_norm, x_norm)
        x = x + attn_out  # 残差连接

        # C. 提取 Attention 的池化特征 (捕捉 96 帧内最强的信号)
        pooled_attn, _ = torch.max(x, dim=1)

        # D. LSTM 长期趋势
        self.long_term_memory.flatten_parameters()
        _, (h_n, _) = self.long_term_memory(x)
        lstm_out = h_n[-1]

        # E. 实时特征
        latest_feature = self.latest_frame_projector(observations[:, -1, :])

        # F. 三路融合
        combined = torch.cat([lstm_out, latest_feature, pooled_attn], dim=-1)
        return self.fusion_layer(combined)