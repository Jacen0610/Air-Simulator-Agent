import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TEA_Extractor_V4(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        seq_len, state_dim = observation_space.shape

        # 直接处理 28 维输入，不搞复杂的投影
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )

        # 针对时序特征，只保留最简单的 GRU (比 LSTM 稳定，参数少)
        self.rnn = nn.GRU(
            input_size=256,
            hidden_size=features_dim,
            num_layers=1,
            batch_first=True
        )

        self.final_layer = nn.Linear(features_dim, features_dim)
        self._apply_init()

    def _apply_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)  # 极小增益
                nn.init.constant_(m.bias, 0)

    def forward(self, observations):
        # --- 暴力防御：只要不是有限数字，全变 0 ---
        observations = torch.where(torch.isfinite(observations), observations, torch.zeros_like(observations))

        # 1. 空间映射
        # observations: (Batch, 96, 28) -> x: (Batch, 96, 256)
        x = self.network(observations)

        # 2. 时序处理
        # 只要最后一帧的状态
        self.rnn.flatten_parameters()
        _, h_n = self.rnn(x)
        out = h_n[0]

        # 3. 最终映射并强制截断
        logits = self.final_layer(out)

        # --- 绝对防御：Categorical 报错的终结者 ---
        # 限制 Logits 范围在 [-5, 5] 之间，防止产生 nan
        return torch.clamp(logits, -5.0, 5.0)