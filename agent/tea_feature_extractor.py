import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TEA_Extractor_V2(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, embed_dim=128):
        super().__init__(observation_space, features_dim)
        seq_len, state_dim = observation_space.shape  # (96, 28)

        # 1. 基础投影层 (处理 96 帧全量 28 维特征)
        self.projection = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # 2. 瞬时扫描仪 (Attention) - 识别 220ms 周期
        self.instant_scanner = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,  # 增加 head 数以增强并行感知
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(embed_dim)

        # 3. 特征演进器 (LSTM)
        self.lstm_hidden_dim = features_dim - 64
        self.long_term_memory = nn.LSTM(
            input_size=embed_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # 4. 【直连通道】实时特征扫描
        # 重点：latest_raw_obs 包含了当前的 16 帧原始 0/1 序列
        self.latest_frame_projector = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # 5. 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU()
        )

    def forward(self, observations):
        # observations shape: (Batch, 96, 28)

        # Step 1: 实时特征路径 (只取最后一帧的 28 维，含 16 位历史)
        latest_raw_obs = observations[:, -1, :]
        latest_feature = self.latest_frame_projector(latest_raw_obs)  # (Batch, 64)

        # Step 2: 序列投影
        x = self.projection(observations)

        # Step 3: Attention (多头关注)
        scanner_out, _ = self.instant_scanner(x, x, x)
        refined_signals = self.attn_norm(x + scanner_out)

        # Step 4: LSTM (长程大局观)
        _, (h_n, _) = self.long_term_memory(refined_signals)
        lstm_out = h_n[-1]  # (Batch, 192)

        # Step 5: 融合。此时 lstm_out 知道“现在是密集区”，
        # 而 latest_feature 看到“刚刚变绿 2ms”，两者竞争后产生决定性 Action。
        combined = torch.cat([lstm_out, latest_feature], dim=-1)  # (Batch, 256)

        return self.fusion_layer(combined)