import torch

from trading_forecast.models.architectures import BiLSTMModel, LSTNetModel, ResCNNGRUModel, TransformerModel


def test_model_output_shapes():
    batch, lookback, features, horizon = 4, 30, 13, 5
    x = torch.randn(batch, lookback, features)

    models = [
        BiLSTMModel(features, hidden_dim=16, horizon=horizon),
        TransformerModel(features, d_model=32, horizon=horizon, nhead=4),
        LSTNetModel(features, cnn_channels=16, gru_hidden=16, horizon=horizon),
        ResCNNGRUModel(features, channels=16, gru_hidden=16, horizon=horizon),
    ]

    for model in models:
        y = model(x)
        assert y.shape == (batch, horizon)
)
