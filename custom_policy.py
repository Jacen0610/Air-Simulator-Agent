import torch
import torch.nn as nn
import gymnasium as gym
import math
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        
        pe[0, :, 0::2] = torch.sin(position * div_term)
        
        num_odd_indices = d_model // 2
        if num_odd_indices > 0:
            pe[0, :, 1::2] = torch.cos(position * div_term[:num_odd_indices])
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class AttentionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        seq_len, features_in = observation_space.shape
        self.positional_encoding = PositionalEncoding(d_model=features_in, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=features_in, nhead=1, dim_feedforward=256,
            batch_first=True, activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.linear = nn.Sequential(
            nn.Linear(seq_len * features_in, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations_with_pe = self.positional_encoding(observations)
        encoded_features = self.transformer_encoder(observations_with_pe)
        flattened_features = torch.flatten(encoded_features, start_dim=1)
        return self.linear(flattened_features)
