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
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

DATASET_PATH = Path("artifacts/dataset.npz")
RESULTS_PATH = Path("results.tsv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAMES = ("bilstm", "transformer", "lstnet", "rescnn_gru", "nbeats", "nhits")


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 0.00101812
    weight_decay: float = 5e-05
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
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        horizon: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        dim_feedforward = dim_feedforward or d_model * 4
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z = self.dropout(z)
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


class NBeatsBlock(nn.Module):
    def __init__(
        self,
        backcast_size: int,
        horizon: int,
        hidden_dim: int,
        theta_dim: int,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = backcast_size
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.theta = nn.Linear(hidden_dim, theta_dim)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        return self.theta(self.backbone(x_flat))


class TrendBasis(nn.Module):
    def __init__(self, backcast_size: int, horizon: int, degree: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.horizon = horizon
        self.degree = degree

        backcast_grid = torch.linspace(-1.0, 0.0, backcast_size)
        forecast_grid = torch.linspace(0.0, 1.0, horizon)
        self.register_buffer(
            'backcast_basis',
            torch.stack([backcast_grid ** i for i in range(degree)], dim=0),
        )
        self.register_buffer(
            'forecast_basis',
            torch.stack([forecast_grid ** i for i in range(degree)], dim=0),
        )

    def forward(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.degree
        theta_back = theta[:, :p]
        theta_fore = theta[:, p: 2 * p]
        backcast = theta_back @ self.backcast_basis
        forecast = theta_fore @ self.forecast_basis
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    def __init__(self, backcast_size: int, horizon: int, harmonics: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.horizon = horizon
        self.harmonics = harmonics

        def make_basis(length: int) -> torch.Tensor:
            t = torch.linspace(0.0, 1.0, length)
            basis = [torch.ones_like(t)]
            for k in range(1, harmonics + 1):
                basis.append(torch.cos(2.0 * math.pi * k * t))
                basis.append(torch.sin(2.0 * math.pi * k * t))
            return torch.stack(basis, dim=0)

        self.register_buffer('backcast_basis', make_basis(backcast_size))
        self.register_buffer('forecast_basis', make_basis(horizon))
        self.theta_dim = self.backcast_basis.size(0) * 2

    def forward(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.backcast_basis.size(0)
        theta_back = theta[:, :p]
        theta_fore = theta[:, p: 2 * p]
        backcast = theta_back @ self.backcast_basis
        forecast = theta_fore @ self.forecast_basis
        return backcast, forecast


class GenericBasis(nn.Module):
    def __init__(self, backcast_size: int, horizon: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.horizon = horizon

    def forward(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        backcast = theta[:, : self.backcast_size]
        forecast = theta[:, self.backcast_size : self.backcast_size + self.horizon]
        return backcast, forecast


class NBeatsStack(nn.Module):
    def __init__(
        self,
        backcast_size: int,
        horizon: int,
        hidden_dim: int,
        num_blocks: int,
        basis: nn.Module,
        theta_dim: int,
        share_weights: bool = False,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.basis = basis
        if share_weights:
            shared = NBeatsBlock(
                backcast_size=backcast_size,
                horizon=horizon,
                hidden_dim=hidden_dim,
                theta_dim=theta_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
            self.blocks = nn.ModuleList([shared for _ in range(num_blocks)])
        else:
            self.blocks = nn.ModuleList(
                [
                    NBeatsBlock(
                        backcast_size=backcast_size,
                        horizon=horizon,
                        hidden_dim=hidden_dim,
                        theta_dim=theta_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                    )
                    for _ in range(num_blocks)
                ]
            )

    def forward(self, residual: torch.Tensor, forecast: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            theta = block(residual)
            backcast, block_forecast = self.basis(theta)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return residual, forecast


class NBeatsModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        lookback: int,
        horizon: int,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        num_layers: int = 4,
        trend_degree: int = 3,
        seasonality_harmonics: int = 6,
        dropout: float = 0.0,
        share_weights: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.lookback = lookback
        self.horizon = horizon
        self.backcast_size = input_dim * lookback

        trend_basis = TrendBasis(self.backcast_size, horizon, degree=trend_degree)
        seasonality_basis = SeasonalityBasis(self.backcast_size, horizon, harmonics=seasonality_harmonics)
        generic_basis = GenericBasis(self.backcast_size, horizon)

        self.stacks = nn.ModuleList(
            [
                NBeatsStack(
                    backcast_size=self.backcast_size,
                    horizon=horizon,
                    hidden_dim=hidden_dim,
                    num_blocks=num_blocks,
                    basis=trend_basis,
                    theta_dim=2 * trend_degree,
                    share_weights=share_weights,
                    num_layers=num_layers,
                    dropout=dropout,
                ),
                NBeatsStack(
                    backcast_size=self.backcast_size,
                    horizon=horizon,
                    hidden_dim=hidden_dim,
                    num_blocks=num_blocks,
                    basis=seasonality_basis,
                    theta_dim=seasonality_basis.theta_dim,
                    share_weights=share_weights,
                    num_layers=num_layers,
                    dropout=dropout,
                ),
                NBeatsStack(
                    backcast_size=self.backcast_size,
                    horizon=horizon,
                    hidden_dim=hidden_dim,
                    num_blocks=max(1, num_blocks // 2),
                    basis=generic_basis,
                    theta_dim=self.backcast_size + horizon,
                    share_weights=share_weights,
                    num_layers=num_layers,
                    dropout=dropout,
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x.reshape(x.size(0), -1)
        forecast = x.new_zeros((x.size(0), self.horizon))
        for stack in self.stacks:
            residual, forecast = stack(residual, forecast)
        return forecast


class NHitsBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        lookback: int,
        horizon: int,
        hidden_dim: int,
        pool_size: int,
        n_freq_downsample: int,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pool_size = max(1, pool_size)
        self.horizon = horizon
        self.n_freq_downsample = max(1, n_freq_downsample)
        pooled_steps = math.ceil(lookback / self.pool_size)
        mlp_in_dim = input_dim * pooled_steps
        coarse_horizon = math.ceil(horizon / self.n_freq_downsample)
        layers: list[nn.Module] = []
        in_dim = mlp_in_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.backcast_head = nn.Linear(hidden_dim, input_dim * lookback)
        self.forecast_head = nn.Linear(hidden_dim, coarse_horizon)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        x_tf = x.transpose(1, 2)
        pooled = F.avg_pool1d(x_tf, kernel_size=self.pool_size, stride=self.pool_size, ceil_mode=True)
        flat = pooled.reshape(batch_size, -1)
        h = self.backbone(flat)
        backcast = self.backcast_head(h).reshape(batch_size, x.size(1), x.size(2))
        coarse_forecast = self.forecast_head(h).unsqueeze(1)
        forecast = F.interpolate(coarse_forecast, size=self.horizon, mode='linear', align_corners=False).squeeze(1)
        return backcast, forecast


class NHitsModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        lookback: int,
        horizon: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        num_layers: int = 3,
        dropout: float = 0.0,
        pool_sizes: list[int] | None = None,
        downsample_frequencies: list[int] | None = None,
    ):
        super().__init__()
        if pool_sizes is None:
            base_pool_sizes = [8, 4, 2, 1]
            pool_sizes = base_pool_sizes[: max(1, num_blocks)]
        if downsample_frequencies is None:
            base_freqs = [8, 4, 2, 1]
            downsample_frequencies = base_freqs[: len(pool_sizes)]
        if len(pool_sizes) != len(downsample_frequencies):
            raise ValueError('pool_sizes and downsample_frequencies must have the same length')

        self.blocks = nn.ModuleList(
            [
                NHitsBlock(
                    input_dim=input_dim,
                    lookback=lookback,
                    horizon=horizon,
                    hidden_dim=hidden_dim,
                    pool_size=pool_size,
                    n_freq_downsample=freq,
                    num_layers=num_layers,
                    dropout=dropout,
                )
                for pool_size, freq in zip(pool_sizes, downsample_frequencies)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        forecast = x.new_zeros((x.size(0), self.blocks[0].horizon))
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return forecast


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
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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


def run_experiment(args: argparse.Namespace):
    ds = load_dataset()
    cfg = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    input_dim = ds["x_train"].shape[-1]
    lookback = ds["x_train"].shape[1]
    horizon = ds["y_train"].shape[-1]
    transformer_ff = args.transformer_dim_feedforward
    if args.transformer_d_model % args.transformer_nhead != 0:
        raise ValueError("transformer-d-model must be divisible by transformer-nhead")
    if lookback < args.lstnet_kernel_size:
        raise ValueError("lookback must be >= lstnet kernel size")
    selected_models = set(args.models)
    unknown_models = selected_models.difference(MODEL_NAMES)
    if unknown_models:
        raise ValueError(f"Unknown models requested: {sorted(unknown_models)}")

    if transformer_ff is not None and transformer_ff <= 0:
        transformer_ff = None
    model_zoo = {
        "bilstm": BiLSTMModel(
            input_dim=input_dim,
            hidden_dim=args.bilstm_hidden_dim,
            horizon=horizon,
            num_layers=args.bilstm_num_layers,
        ),
        "transformer": TransformerModel(
            input_dim=input_dim,
            d_model=args.transformer_d_model,
            horizon=horizon,
            nhead=args.transformer_nhead,
            num_layers=args.transformer_num_layers,
            dim_feedforward=transformer_ff,
            dropout=args.transformer_dropout,
        ),
        "lstnet": LSTNetModel(
            input_dim=input_dim,
            cnn_channels=args.lstnet_cnn_channels,
            gru_hidden=args.lstnet_gru_hidden,
            horizon=horizon,
            kernel_size=args.lstnet_kernel_size,
        ),
        "rescnn_gru": ResCNNGRUModel(
            input_dim=input_dim,
            channels=args.rescnn_channels,
            gru_hidden=args.rescnn_gru_hidden,
            horizon=horizon,
        ),
        "nbeats": NBeatsModel(
            input_dim=input_dim,
            lookback=lookback,
            horizon=horizon,
            hidden_dim=args.nbeats_hidden_dim,
            num_blocks=args.nbeats_num_blocks,
            num_layers=args.nbeats_num_layers,
            trend_degree=args.nbeats_trend_degree,
            seasonality_harmonics=args.nbeats_seasonality_harmonics,
            dropout=args.nbeats_dropout,
            share_weights=args.nbeats_share_weights,
        ),
        "nhits": NHitsModel(
            input_dim=input_dim,
            lookback=lookback,
            horizon=horizon,
            hidden_dim=args.nhits_hidden_dim,
            num_blocks=args.nhits_num_blocks,
            num_layers=args.nhits_num_layers,
            dropout=args.nhits_dropout,
            pool_sizes=args.nhits_pool_sizes,
            downsample_frequencies=args.nhits_downsample_frequencies,
        ),
    }

    test_dl = make_loader(ds["x_test"], ds["y_test"], cfg.batch_size, shuffle=False)
    loss_fn = nn.MSELoss()

    output = {}
    for name, model in model_zoo.items():
        if name not in selected_models:
            continue
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


def _parse_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return [int(part.strip()) for part in cleaned.split(",") if part.strip()]


def _parse_model_list(value: str | None) -> list[str]:
    if value is None:
        return list(MODEL_NAMES)
    cleaned = value.strip()
    if not cleaned:
        return list(MODEL_NAMES)
    parts = [part.strip().lower() for part in cleaned.split(",") if part.strip()]
    if not parts:
        return list(MODEL_NAMES)
    return parts


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
    parser.add_argument(
        "--models",
        type=_parse_model_list,
        default=list(MODEL_NAMES),
        help="Comma-separated model names to run. Default runs all models.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.00101812, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=5e-5, help="AdamW weight decay.")
    parser.add_argument("--bilstm-hidden-dim", type=int, default=160, help="BiLSTM hidden dimension.")
    parser.add_argument("--bilstm-num-layers", type=int, default=3, help="BiLSTM layer count.")
    parser.add_argument("--transformer-d-model", type=int, default=96, help="Transformer d_model.")
    parser.add_argument("--transformer-nhead", type=int, default=4, help="Transformer attention heads.")
    parser.add_argument("--transformer-num-layers", type=int, default=3, help="Transformer encoder layers.")
    parser.add_argument(
        "--transformer-dim-feedforward",
        type=int,
        default=256,
        help="Transformer feedforward dim. Omit or <=0 for default 4*d_model.",
    )
    parser.add_argument("--transformer-dropout", type=float, default=0.0, help="Transformer dropout.")
    parser.add_argument("--lstnet-cnn-channels", type=int, default=64, help="LSTNet conv channels.")
    parser.add_argument("--lstnet-gru-hidden", type=int, default=128, help="LSTNet GRU hidden size.")
    parser.add_argument("--lstnet-kernel-size", type=int, default=5, help="LSTNet conv kernel size.")
    parser.add_argument("--rescnn-channels", type=int, default=96, help="ResCNN-GRU channels.")
    parser.add_argument("--rescnn-gru-hidden", type=int, default=128, help="ResCNN-GRU hidden size.")
    parser.add_argument("--nbeats-hidden-dim", type=int, default=256, help="N-BEATS hidden dimension.")
    parser.add_argument("--nbeats-num-blocks", type=int, default=5, help="Blocks per N-BEATS stack.")
    parser.add_argument("--nbeats-num-layers", type=int, default=5, help="MLP layers per N-BEATS block.")
    parser.add_argument("--nbeats-trend-degree", type=int, default=3, help="Polynomial degree basis for N-BEATS trend stack.")
    parser.add_argument("--nbeats-seasonality-harmonics", type=int, default=6, help="Number of Fourier harmonics for N-BEATS seasonality stack.")
    parser.add_argument("--nbeats-dropout", type=float, default=0.1, help="Dropout inside N-BEATS blocks.")
    parser.add_argument("--nbeats-share-weights", action="store_true", help="Share weights within each N-BEATS stack.")
    parser.add_argument("--nhits-hidden-dim", type=int, default=448, help="N-HiTS hidden dimension.")
    parser.add_argument("--nhits-num-blocks", type=int, default=3, help="N-HiTS block count.")
    parser.add_argument("--nhits-num-layers", type=int, default=2, help="MLP layers per N-HiTS block.")
    parser.add_argument("--nhits-dropout", type=float, default=0.03, help="Dropout inside N-HiTS blocks.")
    parser.add_argument(
        "--nhits-pool-sizes",
        type=_parse_int_list,
        default=[6, 3, 1],
        help="Comma-separated N-HiTS pool sizes, e.g. 8,4,2,1.",
    )
    parser.add_argument(
        "--nhits-downsample-frequencies",
        type=_parse_int_list,
        default=[6, 3, 1],
        help="Comma-separated N-HiTS forecast downsampling factors, e.g. 8,4,2,1.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = run_experiment(args)
    if not args.no_log:
        run_id = append_results_tsv(
            output=output,
            status=args.status,
            description=args.description,
            commit=args.commit,
        )
        print(f"Logged {len(output)} model rows to {RESULTS_PATH} under commit={run_id}")
