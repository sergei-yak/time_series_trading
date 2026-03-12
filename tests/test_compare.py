import numpy as np

from trading_forecast.training.compare import TrainConfig, compare_models


def test_compare_models_returns_train_and_test_diagnostics():
    n_samples, lookback, features, horizon = 48, 12, 13, 3
    x = np.random.randn(n_samples, lookback, features).astype(np.float32)
    y = np.random.randn(n_samples, horizon).astype(np.float32)

    cfg = TrainConfig(epochs=1, batch_size=16, device="cpu")
    out = compare_models(x, y, horizon=horizon, cfg=cfg)

    assert set(out.keys()) == {"BiLSTM", "Transformer", "LSTNet", "ResCNN+GRU"}
    for model_out in out.values():
        assert "history" in model_out
        assert "train" in model_out
        assert "test" in model_out
        assert len(model_out["history"]["train_loss"]) == 1
        assert len(model_out["history"]["test_loss"]) == 1
        assert model_out["train"]["pred"].shape[1] == horizon
        assert model_out["test"]["pred"].shape[1] == horizon
