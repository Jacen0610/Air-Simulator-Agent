import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TEA_Extractor_V4(BaseFeaturesExtractor):
    # 增加 embed_dim=128 用于兼容之前的 policy_kwargs 传参
    def __init__(self, observation_space, features_dim=256, embed_dim=128):
        super().__init__(observation_space, features_dim)
        seq_len, state_dim = observation_space.shape

        # 1. 空间特征投影 (加上层归一化)
        # 注意：这里我们使用 state_dim (28)
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )

        # 2. 时序处理：GRU 比 LSTM 更抗梯度爆炸
        self.rnn = nn.GRU(
            input_size=256,
            hidden_size=features_dim,
            num_layers=1,
            batch_first=True
        )

        # 3. 最终映射
        self.final_layer = nn.Linear(features_dim, features_dim)

        # 4. 执行极其保守的正交初始化
        self._apply_init()

    def _apply_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 极小的 gain (0.01) 是为了让模型在开始 3 分钟内处于极其安静的状态
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param.data, gain=0.01)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)

    def forward(self, observations):
        # --- 暴力防御 A: 数值截断 ---
        # 确保 observations 没有任何 nan/inf
        observations = torch.nan_to_num(observations, nan=0.0, posinf=1.0, neginf=0.0)
        # 强制限制在 Go 后端给出的 0-1 范围内
        observations = torch.clamp(observations, 0.0, 1.0)

        # 1. 空间映射: (Batch, 96, 28) -> (Batch, 96, 256)
        x = self.network(observations)

        # 2. 时序压缩: (Batch, 96, 256) -> (Batch, 256)
        self.rnn.flatten_parameters()
        _, h_n = self.rnn(x)
        # 取 GRU 最后一层的输出
        rnn_out = h_n[-1]

        # 3. 最终输出
        logits = self.final_layer(rnn_out)

        # --- 暴力防御 B: 梯度防爆墙 ---
        # 强制截断 logits 范围。Categorical 报错的核心原因就是 logits 差值过大
        # 限制在 [-5, 5] 能保证 softmax 后的概率最大也就是 0.99 左右，不会出现概率为 0 的情况
        return torch.clamp(logits, -5.0, 5.0)