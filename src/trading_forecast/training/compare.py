from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trading_forecast.models.architectures import BiLSTMModel, LSTNetModel, ResCNNGRUModel, TransformerModel


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 64
    lr: float = 1e-3
    device: str = "cpu"


def split_data(x: np.ndarray, y: np.ndarray, ratio: float = 0.8):
    cutoff = int(len(x) * ratio)
    return (x[:cutoff], y[:cutoff]), (x[cutoff:], y[cutoff:])


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _metrics(pred: np.ndarray, tgt: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(pred - tgt)))
    rmse = float(np.sqrt(np.mean((pred - tgt) ** 2)))
    mape = float(np.mean(np.abs((tgt - pred) / (np.abs(tgt) + 1e-6))) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def evaluate_loader(model: nn.Module, dl: DataLoader, device: str, loss_fn: nn.Module):
    model.eval()
    preds, targets = [], []
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = loss_fn(out, yb)
            total_loss += float(loss.item())
            batches += 1
            preds.append(out.cpu().numpy())
            targets.append(yb.cpu().numpy())

    pred = np.concatenate(preds)
    tgt = np.concatenate(targets)
    avg_loss = total_loss / max(batches, 1)
    return {"loss": avg_loss, "pred": pred, "target": tgt, "metrics": _metrics(pred, tgt)}


def train_one(model: nn.Module, train_dl: DataLoader, test_dl: DataLoader, cfg: TrainConfig):
def train_one(model: nn.Module, train_dl: DataLoader, val_dl: DataLoader, cfg: TrainConfig):
    model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "test_loss": []}
    for _ in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        batches = 0
    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += float(loss.item())
            batches += 1

        history["train_loss"].append(running_loss / max(batches, 1))
        test_eval = evaluate_loader(model, test_dl, cfg.device, loss_fn)
        history["test_loss"].append(test_eval["loss"])

    train_eval = evaluate_loader(model, train_dl, cfg.device, loss_fn)
    test_eval = evaluate_loader(model, test_dl, cfg.device, loss_fn)
    return {
        "history": history,
        "train": train_eval,
        "test": test_eval,
    }

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(cfg.device)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            targets.append(yb.numpy())

    pred = np.concatenate(preds)
    tgt = np.concatenate(targets)
    mae = float(np.mean(np.abs(pred - tgt)))
    rmse = float(np.sqrt(np.mean((pred - tgt) ** 2)))
    mape = float(np.mean(np.abs((tgt - pred) / (np.abs(tgt) + 1e-6))) * 100)
    return {"metrics": {"MAE": mae, "RMSE": rmse, "MAPE": mape}, "pred": pred, "target": tgt}


def compare_models(x: np.ndarray, y: np.ndarray, horizon: int, cfg: TrainConfig):
    (x_train, y_train), (x_test, y_test) = split_data(x, y)
    train_dl = _loader(x_train, y_train, cfg.batch_size, shuffle=True)
    test_dl = _loader(x_test, y_test, cfg.batch_size, shuffle=False)

    input_dim = x.shape[-1]
    models = {
        "BiLSTM": BiLSTMModel(input_dim=input_dim, hidden_dim=64, horizon=horizon),
        "Transformer": TransformerModel(input_dim=input_dim, d_model=64, horizon=horizon),
        "LSTNet": LSTNetModel(input_dim=input_dim, cnn_channels=64, gru_hidden=64, horizon=horizon),
        "ResCNN+GRU": ResCNNGRUModel(input_dim=input_dim, channels=64, gru_hidden=64, horizon=horizon),
    }

    outputs = {}
    for name, model in models.items():
        outputs[name] = train_one(model, train_dl, test_dl, cfg)

    return outputs
