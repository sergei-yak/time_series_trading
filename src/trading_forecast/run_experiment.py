from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from trading_forecast import __version__
from trading_forecast.data.coinbase_client import CoinbaseConfig, CoinbasePublicClient
from trading_forecast.data.pipeline import DataPipeline, DatasetConfig, utc_now
from trading_forecast.training.compare import TrainConfig, compare_models

COINBASE_ALLOWED_GRANULARITIES = {60, 300, 900, 3600, 21600, 86400}


def parse_args():
    p = argparse.ArgumentParser(description="BTC forecasting model comparison")
    p.add_argument("--version", action="version", version=f"trading_forecast {__version__}")
    p.add_argument("--product-id", default="BTC-USD", help="Coinbase product id, e.g. BTC-USD")
    p.add_argument("--granularity", type=int, default=60, help="Candlestick size in seconds (Coinbase min=60)")
    p.add_argument("--hours", type=int, default=24 * 14, help="Lookback history size in hours")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--output-csv", default="artifacts/coinbase_dataset.csv")
    p.add_argument("--plots-dir", default="artifacts/plots")
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.granularity not in COINBASE_ALLOWED_GRANULARITIES:
        supported = ", ".join(map(str, sorted(COINBASE_ALLOWED_GRANULARITIES)))
        raise ValueError(f"Unsupported Coinbase granularity: {args.granularity}. Supported values: {supported}")
    if args.horizon < 1:
        raise ValueError("--horizon must be >= 1")
    if args.lookback < 1:
        raise ValueError("--lookback must be >= 1")


def resolve_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available in this PyTorch installation")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_output_parent(path_like: str) -> Path:
    out_path = Path(path_like)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def ensure_dir(path_like: str) -> Path:
    path = Path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_predictions_split(model_name: str, split: str, y_true, y_pred, out_dir: Path) -> str:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true.reshape(-1), mode="lines", name=f"Real Close Price ({split})"))
    fig.add_trace(go.Scatter(y=y_pred.reshape(-1), mode="lines", name=f"Predicted Close Price ({split})"))
    fig.update_layout(
        title=f"{model_name}: Real vs Predicted Close Price ({split.title()} Set)",
        xaxis_title="Timeline (flattened horizon steps)",
        yaxis_title="Price",
        template="plotly_white",
    )

    safe_name = model_name.lower().replace('+', 'plus').replace(' ', '_')
    out_path = out_dir / f"{safe_name}_{split}_prediction.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    return str(out_path)


def plot_learning_curve(model_name: str, history: dict[str, list[float]], out_dir: Path) -> str:
    import plotly.graph_objects as go

    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], mode="lines+markers", name="Train Loss"))
    fig.add_trace(go.Scatter(x=epochs, y=history["test_loss"], mode="lines+markers", name="Test Loss"))
    fig.update_layout(
        title=f"{model_name}: Train vs Test Loss by Epoch",
        xaxis_title="Epoch",
        yaxis_title="MSE Loss",
        template="plotly_white",
    )

    safe_name = model_name.lower().replace('+', 'plus').replace(' ', '_')
    out_path = out_dir / f"{safe_name}_learning_curve.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
def plot_model_predictions(model_name: str, y_true, y_pred, out_dir: Path) -> str:
    import plotly.graph_objects as go

    true_flat = y_true.reshape(-1)
    pred_flat = y_pred.reshape(-1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=true_flat, mode="lines", name="Real Close Price"))
    fig.add_trace(go.Scatter(y=pred_flat, mode="lines", name=f"{model_name} Predicted Close Price"))
    fig.update_layout(
        title=f"{model_name}: Real vs Predicted Close Price (Test Set)",
        xaxis_title="Test timeline (flattened horizon steps)",
        yaxis_title="Price",
        template="plotly_white",
    )

    out_path = out_dir / f"{model_name.lower().replace('+', 'plus').replace(' ', '_')}_test_prediction.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    true_flat = y_true.reshape(-1)
    pred_flat = y_pred.reshape(-1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(true_flat, label="Real Close Price", linewidth=2)
    ax.plot(pred_flat, label=f"{model_name} Predicted Close Price", linewidth=1.5)
    ax.set_title(f"{model_name}: Real vs Predicted Close Price (Test Set)")
    ax.set_xlabel("Test timeline (flattened horizon steps)")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(alpha=0.3)

    out_path = out_dir / f"{model_name.lower().replace('+', 'plus').replace(' ', '_')}_test_prediction.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


def main():
    args = parse_args()
    validate_args(args)

    end = utc_now()
    start = end - timedelta(hours=args.hours)

    cb_cfg = CoinbaseConfig(product_id=args.product_id, granularity_seconds=args.granularity)
    pipeline_cfg = DatasetConfig(lookback=args.lookback, horizon=args.horizon, granularity_seconds=args.granularity)

    client = CoinbasePublicClient(cb_cfg)
    pipeline = DataPipeline(client, pipeline_cfg)

    df = pipeline.fetch_dataset(start=start, end=end)
    out_csv = ensure_output_parent(args.output_csv)
    pd.DataFrame(df).to_csv(out_csv, index=False)

    x, y = pipeline.make_supervised(df)

    device = resolve_device(args.device)
    train_cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, device=device)
    outputs = compare_models(x, y, horizon=args.horizon, cfg=train_cfg)

    plots_dir = ensure_dir(args.plots_dir)
    results = {}
    plot_paths = {}
    for model_name, output in outputs.items():
        train_true = pipeline.inverse_close_scale(output["train"]["target"])
        train_pred = pipeline.inverse_close_scale(output["train"]["pred"])
        test_true = pipeline.inverse_close_scale(output["test"]["target"])
        test_pred = pipeline.inverse_close_scale(output["test"]["pred"])

        plot_paths[model_name] = {
            "train_prediction": plot_predictions_split(model_name, "train", train_true, train_pred, plots_dir),
            "test_prediction": plot_predictions_split(model_name, "test", test_true, test_pred, plots_dir),
            "learning_curve": plot_learning_curve(model_name, output["history"], plots_dir),
        }

        results[model_name] = {
            "train": output["train"]["metrics"],
            "test": output["test"]["metrics"],
            "final_train_loss": output["history"]["train_loss"][-1],
            "final_test_loss": output["history"]["test_loss"][-1],
        }
        y_pred_real = pipeline.inverse_close_scale(output["pred"])
        y_true_real = pipeline.inverse_close_scale(output["target"])
        plot_paths[model_name] = plot_model_predictions(model_name, y_true_real, y_pred_real, plots_dir)
        results[model_name] = output["metrics"]

    print(
        json.dumps(
            {
                "rows": len(df),
                "output_csv": str(out_csv),
                "plots_dir": str(plots_dir),
                "plot_files": plot_paths,
                "device": device,
                "results": results,
                "version": __version__,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
