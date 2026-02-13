import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TEA_Extractor_V3(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, embed_dim=128):
        super().__init__(observation_space, features_dim)
        # 获取观测空间的形状: [seq_len, state_dim]
        seq_len, state_dim = observation_space.shape

        # 1. 位置编码 (加入可学习的位置向量)
        # 赋予 Transformer 识别“哪一帧是最新、哪一帧是旧”的能力
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim))

        # 2. 基础投影层 (将原始特征映射到嵌入空间)
        self.projection = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # 3. 瞬时扫描仪 (Multi-Head Attention)
        # 捕捉 96 帧内背景流量的节律特征
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.instant_scanner = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.05
        )

        # 4. 特征演进器 (LSTM)
        # 负责处理 40 分钟 Episode 内的宏观信道趋势
        self.lstm_hidden_dim = 128
        self.long_term_memory = nn.LSTM(
            input_size=embed_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # 5. 实时特征直连 (Raw Feature Skip-connection)
        # 确保决策层对“当前这一帧”的 wait_time 和 q_size 高度敏感
        self.latest_frame_projector = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )

        # 6. 最终融合层 (修正维度对齐)
        # 维度计算逻辑:
        # lstm_out (128) + latest_feature (64) + pooled_attn (128) + curr_is_busy (1)
        # 总和: 128 + 64 + 128 + 1 = 321
        raw_signal_dim = 1  # 对应 forward 中的 curr_is_busy
        fusion_input_dim = self.lstm_hidden_dim + 64 + embed_dim + raw_signal_dim

        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU(),
            nn.Linear(features_dim, features_dim)
        )

        # 核心加固：使用正交初始化防止 40 分钟长序列下的梯度消失/爆炸
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
        # --- 1. 数据防线与关键特征提取 ---
        observations = torch.nan_to_num(observations, nan=0.0, posinf=1.0, neginf=0.0)

        # 关键物理状态提取: 观测向量索引 1 是 is_busy
        # 我们在这里强行拎出这一帧的忙碌状态，作为决策的“紧急刹车”信号
        curr_is_busy = observations[:, -1, 1:2]

        # --- 2. 投影与时序建模 ---
        x = self.projection(observations)
        x = x + self.pos_embedding  # 注入时间位置信息

        # Attention 扫描 (残差连接)
        x_norm = self.attn_norm(x)
        attn_out, _ = self.instant_scanner(x_norm, x_norm, x_norm)
        x = x + attn_out

        # 捕捉 96 帧中最显著的空隙特征
        pooled_attn, _ = torch.max(x, dim=1)

        # LSTM 长期趋势提取
        self.long_term_memory.flatten_parameters()
        _, (h_n, _) = self.long_term_memory(x)
        lstm_out = h_n[-1]

        # --- 3. 实时特征增强 ---
        # 处理当前帧的全量信息 (包含 wait_time, q_size 等)
        latest_feature = self.latest_frame_projector(observations[:, -1, :])

        # --- 4. 最终融合 (重点解决无效动作) ---
        # 我们把原始的 is_busy 信号乘以 5 倍权重后强行拼接
        # 理由：在融合层中，如果不加权，is_busy 的信号强度会被 wait_time 带来的焦虑淹没
        # 增加这 1 维物理信号能让模型更直观地感知到“忙碌 = 扣8分”
        combined = torch.cat([
            lstm_out,  # 过去背景流量的统计特征 (128维)
            latest_feature,  # 当前帧的综合感知 (64维)
            pooled_attn,  # 96帧内的瞬时节奏感 (128维)
            curr_is_busy * 5.0  # 物理警报信号 (1维)
        ], dim=-1)

        return self.fusion_layer(combined)