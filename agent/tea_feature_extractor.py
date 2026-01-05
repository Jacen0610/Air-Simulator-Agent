import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TEA_Extractor_V3(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, embed_dim=128):
        super().__init__(observation_space, features_dim)
        seq_len, state_dim = observation_space.shape

        # 1. 输入投影：使用更强的 LayerNorm 约束
        self.projection = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # 2. Attention：强制增加 Residual Dropout 防止过拟合引起的梯度集中
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.instant_scanner = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            dropout=0.1, # 增加随机性，分散梯度
            batch_first=True
        )

        # 3. LSTM：这是最容易爆的地方
        self.lstm_hidden_dim = features_dim - 64
        self.long_term_memory = nn.LSTM(
            input_size=embed_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=1,
            batch_first=True
        )
        # 核心：对 LSTM 输出进行 LayerNorm，并重新启用 Tanh (物理隔离)
        self.lstm_post_norm = nn.LayerNorm(self.lstm_hidden_dim)

        # 4. 实时路径
        self.latest_frame_projector = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )

        # 5. 融合层
        self.final_norm = nn.LayerNorm(features_dim)
        self.fusion_layer = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.LayerNorm(features_dim), # 最后一层加回 Norm，确保护送到 policy 的 logits 不溢出
            nn.GELU()
        )

        self._apply_orthogonal_init()

    def _apply_orthogonal_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 关键：减小 gain，让初始权重更小更安全
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name: nn.init.orthogonal_(param.data, gain=0.01)
                    elif 'bias' in name: nn.init.constant_(param.data, 0)

    def forward(self, observations):
        # 1. 极致的数据清洗 (预防任何来自 Go 的异常)
        observations = torch.nan_to_num(observations, nan=0.0, posinf=1.0, neginf=0.0)
        observations = torch.clamp(observations, 0.0, 1.0) # 因为你说后端已经归一化

        # 2. 实时路径
        latest_feature = self.latest_frame_projector(observations[:, -1, :])

        # 3. Attention (Pre-LN + Residual)
        x = self.projection(observations)
        x_norm = self.attn_norm(x)
        scanner_out, _ = self.instant_scanner(x_norm, x_norm, x_norm)
        # 关键：残差缩放，防止数值翻倍
        refined_signals = x * 0.5 + scanner_out * 0.5

        # 4. LSTM
        self.long_term_memory.flatten_parameters()
        _, (h_n, _) = self.long_term_memory(refined_signals)
        # 终极防御：Norm + Tanh
        lstm_out = self.lstm_post_norm(h_n[-1])
        lstm_out = torch.tanh(lstm_out)

        # 5. 融合
        combined = torch.cat([lstm_out, latest_feature], dim=-1)
        combined = self.final_norm(combined)
        return self.fusion_layer(combined)