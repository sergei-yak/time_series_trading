from __future__ import annotations

import torch
from torch import nn


class BiLSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, horizon: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class TransformerModel(nn.Module):
    def __init__(self, input_dim: int, d_model: int, horizon: int, nhead: int = 4, layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z = self.encoder(z)
        return self.head(z[:, -1, :])


class LSTNetModel(nn.Module):
    """Simplified LSTNet: CNN + GRU branch."""

    def __init__(self, input_dim: int, cnn_channels: int, gru_hidden: int, horizon: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, cnn_channels, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(cnn_channels, gru_hidden, batch_first=True)
        self.head = nn.Linear(gru_hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.relu(self.conv(x))
        x = x.transpose(1, 2)
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class ResCNNGRUModel(nn.Module):
    def __init__(self, input_dim: int, channels: int, gru_hidden: int, horizon: int):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.skip = nn.Conv1d(input_dim, channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(channels, gru_hidden, batch_first=True)
        self.head = nn.Linear(gru_hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        residual = self.skip(z)
        z = self.relu(self.conv1(z))
        z = self.conv2(z) + residual
        z = self.relu(z).transpose(1, 2)
        out, _ = self.gru(z)
        return self.head(out[:, -1, :])
