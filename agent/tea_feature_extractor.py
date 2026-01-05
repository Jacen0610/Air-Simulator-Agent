import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TEA_Extractor_V2(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, embed_dim=128):
        super().__init__(observation_space, features_dim)
        # 获取输入维度 (96, 28)
        seq_len, state_dim = observation_space.shape

        # 1. 基础投影层
        self.projection = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # 2. 瞬时扫描仪 (Attention) - Pre-LN 结构加固
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.instant_scanner = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            batch_first=True
        )

        # 3. 特征演进器 (LSTM)
        # 192 = 256 - 64
        self.lstm_hidden_dim = features_dim - 64
        self.long_term_memory = nn.LSTM(
            input_size=embed_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # 4. 【直连通道】实时特征扫描
        # 将 BatchNorm1d 换成 LayerNorm，防止 RL 训练中的数值漂移
        self.latest_frame_projector = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # 5. 最终融合前的归一化层 (防止 nan 的最后一道防线)
        self.final_norm = nn.LayerNorm(features_dim)

        # 6. 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU()
        )

    def forward(self, observations):
        # 0. 数值安全预处理
        if torch.isnan(observations).any():
            observations = torch.nan_to_num(observations, nan=0.0, posinf=1.0, neginf=-1.0)

        # Step 1: 实时特征路径 (Batch, 64)
        latest_raw_obs = observations[:, -1, :]
        latest_feature = self.latest_frame_projector(latest_raw_obs)

        # Step 2: 全局序列投影 (Batch, 96, embed_dim)
        x = self.projection(observations)

        # Step 3: Attention (使用 Pre-LN 结构)
        x_norm = self.attn_norm(x)
        scanner_out, _ = self.instant_scanner(x_norm, x_norm, x_norm)
        # 残差连接
        refined_signals = x + scanner_out

        # Step 4: LSTM 处理
        self.long_term_memory.flatten_parameters()
        _, (h_n, _) = self.long_term_memory(refined_signals)
        lstm_out = h_n[-1] # (Batch, 192)

        # Step 5: 拼接融合 (192 + 64 = 256)
        combined = torch.cat([lstm_out, latest_feature], dim=-1)

        # Step 6: 归一化与最终映射
        combined = self.final_norm(combined)
        return self.fusion_layer(combined)