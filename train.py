from __future__ import annotations

"""Training entrypoint.

This file is intentionally designed as the main place an optimization agent can edit:
- model architecture definitions
- optimizer/scheduler choices
- training hyperparameters
- training loop behavior
"""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATASET_PATH = Path("artifacts/dataset.npz")
RESULTS_PATH = Path("results.tsv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 7e-4
    weight_decay: float = 1e-5
    device: str = DEVICE


class BiLSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, horizon: int, num_layers: int = 2):
        super().__init__()
        self.net = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.net(x)
        return self.head(out[:, -1, :])


class TransformerModel(nn.Module):
    def __init__(self, input_dim: int, d_model: int, horizon: int, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z = self.encoder(z)
        return self.head(z[:, -1, :])


class LSTNetModel(nn.Module):
    def __init__(self, input_dim: int, cnn_channels: int, gru_hidden: int, horizon: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, cnn_channels, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(cnn_channels, gru_hidden, batch_first=True)
        self.head = nn.Linear(gru_hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        z = self.relu(self.conv(z))
        z = z.transpose(1, 2)
        out, _ = self.gru(z)
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


def load_dataset(path: Path = DATASET_PATH):
    data = np.load(path)
    return {
        "x_train": data["x_train"],
        "y_train": data["y_train"],
        "x_val": data["x_val"],
        "y_val": data["y_val"],
        "x_test": data["x_test"],
        "y_test": data["y_test"],
    }


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate(model: nn.Module, dl: DataLoader, loss_fn: nn.Module, device: str):
    model.eval()
    losses = []
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            losses.append(float(loss.item()))
            preds.append(pred.cpu().numpy())
            targets.append(yb.cpu().numpy())
    pred_arr = np.concatenate(preds)
    tgt_arr = np.concatenate(targets)
    mae = float(np.mean(np.abs(pred_arr - tgt_arr)))
    rmse = float(np.sqrt(np.mean((pred_arr - tgt_arr) ** 2)))
    return {"loss": float(np.mean(losses)), "mae": mae, "rmse": rmse}


def train_one_model(model: nn.Module, ds: dict[str, np.ndarray], cfg: TrainConfig):
    model = model.to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    train_dl = make_loader(ds["x_train"], ds["y_train"], cfg.batch_size, shuffle=True)
    val_dl = make_loader(ds["x_val"], ds["y_val"], cfg.batch_size, shuffle=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    history = {"train_loss": [], "val_loss": []}
    best_state = None
    best_val = float("inf")

    for _ in range(cfg.epochs):
        model.train()
        epoch_losses = []
        for xb, yb in train_dl:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        scheduler.step()
        train_loss = float(np.mean(epoch_losses))
        val_eval = evaluate(model, val_dl, loss_fn, cfg.device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_eval["loss"])

        if val_eval["loss"] < best_val:
            best_val = val_eval["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def run_experiment():
    ds = load_dataset()
    cfg = TrainConfig()

    input_dim = ds["x_train"].shape[-1]
    horizon = ds["y_train"].shape[-1]
    model_zoo = {
        "bilstm": BiLSTMModel(input_dim=input_dim, hidden_dim=96, horizon=horizon),
        "transformer": TransformerModel(input_dim=input_dim, d_model=64, horizon=horizon),
        "lstnet": LSTNetModel(input_dim=input_dim, cnn_channels=64, gru_hidden=64, horizon=horizon),
        "rescnn_gru": ResCNNGRUModel(input_dim=input_dim, channels=64, gru_hidden=64, horizon=horizon),
    }

    test_dl = make_loader(ds["x_test"], ds["y_test"], cfg.batch_size, shuffle=False)
    loss_fn = nn.MSELoss()

    output = {}
    for name, model in model_zoo.items():
        trained, history = train_one_model(model, ds, cfg)
        metrics = evaluate(trained, test_dl, loss_fn, cfg.device)
        output[name] = {"history": history, "test": metrics}
        print(f"{name}: test_loss={metrics['loss']:.4f}, mae={metrics['mae']:.4f}, rmse={metrics['rmse']:.4f}")

    return output


def ensure_results_header(path: Path = RESULTS_PATH) -> None:
    if not path.exists() or path.stat().st_size == 0:
        path.write_text("commit\tRMSE\tstatus\tmodel\tdescription\n", encoding="utf-8")


def load_best_rmse_by_model(path: Path = RESULTS_PATH) -> dict[str, float]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        best: dict[str, float] = {}
        for row in reader:
            model_name = (row.get("model") or "").strip()
            rmse_str = (row.get("RMSE") or "").strip()
            status = (row.get("status") or "").strip()
            if not model_name or not rmse_str or status == "crash":
                continue
            try:
                rmse_value = float(rmse_str)
            except ValueError:
                continue
            if model_name not in best or rmse_value < best[model_name]:
                best[model_name] = rmse_value
    return best


def append_results_tsv(
    output: dict[str, dict[str, dict[str, list[float]] | dict[str, float]]],
    status: str,
    description: str,
    commit: str | None = None,
    path: Path = RESULTS_PATH,
) -> str:
    allowed_statuses = {"auto", "keep", "discard", "crash"}
    if status not in allowed_statuses:
        raise ValueError(f"Unsupported status '{status}'. Allowed: auto, keep, discard, crash")
    ensure_results_header(path)
    best_rmse_by_model = load_best_rmse_by_model(path)
    run_id = commit or datetime.now().strftime("run_%Y-%m-%d_%H%M%S")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for model_name, payload in output.items():
            test = payload["test"]
            rmse = float(test["rmse"])
            if status == "auto":
                prev_best = best_rmse_by_model.get(model_name, float("inf"))
                row_status = "keep" if rmse <= prev_best else "discard"
            else:
                row_status = status
            row_description = f"{description}; test_loss={test['loss']:.6f} mae={test['mae']:.6f}"
            writer.writerow([run_id, f"{rmse:.6f}", row_status, model_name, row_description])
    return run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training experiment and optionally log all model results.")
    parser.add_argument(
        "--status",
        default="auto",
        help="Status mode for results.tsv (auto, keep, discard, crash).",
    )
    parser.add_argument(
        "--description",
        default="auto run",
        help="Short run description saved in results.tsv.",
    )
    parser.add_argument(
        "--commit",
        default=None,
        help="Commit hash or run id for results.tsv. Defaults to a timestamp run id.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Run training without appending rows to results.tsv.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = run_experiment()
    if not args.no_log:
        run_id = append_results_tsv(
            output=output,
            status=args.status,
            description=args.description,
            commit=args.commit,
        )
        print(f"Logged {len(output)} model rows to {RESULTS_PATH} under commit={run_id}")
