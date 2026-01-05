import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TEA_Extractor_V2(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, embed_dim=128):
        super().__init__(observation_space, features_dim)
        seq_len, state_dim = observation_space.shape

        # 1. 基础投影层 (Pre-LN 风格)
        self.projection = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # 2. 瞬时扫描仪 (Attention) - 捕捉 96 帧中的关键时刻
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.instant_scanner = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            batch_first=True
        )

        # 3. 特征演进器 (LSTM) - 处理长短期记忆
        self.lstm_hidden_dim = features_dim - 64
        self.long_term_memory = nn.LSTM(
            input_size=embed_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.lstm_post_norm = nn.LayerNorm(self.lstm_hidden_dim)

        # 4. 实时特征扫描 (直连通道：保证模型对当前帧最敏感)
        self.latest_frame_projector = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )

        # 5. 融合与输出层
        self.final_norm = nn.LayerNorm(features_dim)
        self.fusion_layer = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.GELU()
        )

        # 核心加固：使用较小的 gain 进行初始化，防止训练初期震荡
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
        # --- 数据防线：虽然 Go 修好了，但这里留着也不影响性能 ---
        observations = torch.nan_to_num(observations, nan=0.0, posinf=1.0, neginf=0.0)

        # Step 1: 实时路径
        latest_feature = self.latest_frame_projector(observations[:, -1, :])

        # Step 2: 投影
        x = self.projection(observations)

        # Step 3: Attention (使用残差连接)
        x_norm = self.attn_norm(x)
        scanner_out, _ = self.instant_scanner(x_norm, x_norm, x_norm)
        refined_signals = x + scanner_out

        # Step 4: LSTM
        self.long_term_memory.flatten_parameters()
        _, (h_n, _) = self.long_term_memory(refined_signals)
        lstm_out = self.lstm_post_norm(h_n[-1])

        # Step 5: 融合
        combined = torch.cat([lstm_out, latest_feature], dim=-1)
        combined = self.final_norm(combined)
        return self.fusion_layer(combined)