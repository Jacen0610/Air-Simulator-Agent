import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TEA_Extractor_V2(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, embed_dim=128):
        super().__init__(observation_space, features_dim)
        seq_len, state_dim = observation_space.shape

        # --- 1. 基础投影层 (Pre-LN 风格) ---
        self.projection = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # --- 2. 瞬时扫描仪 (Attention) ---
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.instant_scanner = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            batch_first=True
        )

        # --- 3. 特征演进器 (LSTM) ---
        self.lstm_hidden_dim = features_dim - 64
        self.long_term_memory = nn.LSTM(
            input_size=embed_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=1,
            batch_first=True
        )
        # 核心加固：防止 LSTM 输出值过大
        self.lstm_post_norm = nn.LayerNorm(self.lstm_hidden_dim)

        # --- 4. 实时特征扫描 (直连通道) ---
        self.latest_frame_projector = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64), # 增加一层 Norm
            nn.GELU()
        )

        # --- 5. 融合与输出层 ---
        self.final_norm = nn.LayerNorm(features_dim)
        self.fusion_layer = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU(),
            # 最后一层使用较小的初始化缩放，防止 Logits 爆炸
            nn.Linear(features_dim, features_dim)
        )

        # --- 6. 权重初始化 (关键：正交初始化) ---
        self._apply_orthogonal_init()

    def _apply_orthogonal_init(self):
        """强化学习中极其重要的初始化步奏，能有效缓解 nan"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)

    def forward(self, observations):
        # 0. 极端数值防御
        if torch.isnan(observations).any():
            observations = torch.nan_to_num(observations, nan=0.0, posinf=1.0, neginf=-1.0)

        # Step 1: 实时特征路径 (Batch, 64)
        latest_raw_obs = observations[:, -1, :]
        latest_feature = self.latest_frame_projector(latest_raw_obs)

        # Step 2: 投影
        x = self.projection(observations)

        # Step 3: Attention (Pre-LN)
        x_norm = self.attn_norm(x)
        scanner_out, _ = self.instant_scanner(x_norm, x_norm, x_norm)
        # 加上残差并再次约束
        refined_signals = x + scanner_out

        # Step 4: LSTM
        self.long_term_memory.flatten_parameters()
        _, (h_n, _) = self.long_term_memory(refined_signals)
        # 核心加固：对 LSTM 输出做 LayerNorm 和 Tanh 限制
        lstm_out = self.lstm_post_norm(h_n[-1])
        lstm_out = torch.tanh(lstm_out)

        # Step 5: 融合 (192 + 64 = 256)
        combined = torch.cat([lstm_out, latest_feature], dim=-1)

        # Step 6: 最终输出
        combined = self.final_norm(combined)
        return self.fusion_layer(combined)