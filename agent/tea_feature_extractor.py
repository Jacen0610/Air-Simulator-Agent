import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TEA_Extractor_V2(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, embed_dim=128):
        super().__init__(observation_space, features_dim)
        seq_len, state_dim = observation_space.shape

        # 1. 基础投影层
        self.projection = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # 2. 瞬时扫描仪 (Attention) - 保留 Pre-LN，这是 Transformer 的最佳实践
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.instant_scanner = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            batch_first=True
        )

        # 3. 特征演进器 (LSTM)
        self.lstm_hidden_dim = features_dim - 64
        self.long_term_memory = nn.LSTM(
            input_size=embed_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.lstm_post_norm = nn.LayerNorm(self.lstm_hidden_dim)

        # 4. 实时特征投影 (精简为一层 Norm)
        self.latest_frame_projector = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )

        # 5. 融合与输出 (精简内部结构)
        self.final_norm = nn.LayerNorm(features_dim)
        self.fusion_layer = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.GELU() # 去掉了内部的 LayerNorm
        )

        self._apply_orthogonal_init()

    def _apply_orthogonal_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name: nn.init.orthogonal_(param.data)
                    elif 'bias' in name: nn.init.constant_(param.data, 0)

    def forward(self, observations):
        # 1. 实时路径
        latest_feature = self.latest_frame_projector(observations[:, -1, :])

        # 2. Attention 路径 (Pre-LN)
        x = self.projection(observations)
        scanner_out, _ = self.instant_scanner(self.attn_norm(x), self.attn_norm(x), self.attn_norm(x))
        refined_signals = x + scanner_out

        # 3. LSTM 路径
        self.long_term_memory.flatten_parameters()
        _, (h_n, _) = self.long_term_memory(refined_signals)
        lstm_out = self.lstm_post_norm(h_n[-1]) # 移除了 tanh，靠 LayerNorm 约束

        # 4. 融合
        combined = torch.cat([lstm_out, latest_feature], dim=-1)
        return self.fusion_layer(self.final_norm(combined))