from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import pandas as pd
import torch

from trading_forecast.data.coinbase_client import CoinbaseConfig, CoinbasePublicClient
from trading_forecast.data.pipeline import DataPipeline, DatasetConfig, utc_now
from trading_forecast.training.compare import TrainConfig, compare_models

COINBASE_ALLOWED_GRANULARITIES = {60, 300, 900, 3600, 21600, 86400}


def parse_args():
    p = argparse.ArgumentParser(description="BTC forecasting model comparison")
    p.add_argument("--product-id", default="BTC-USD", help="Coinbase product id, e.g. BTC-USD")
    p.add_argument("--granularity", type=int, default=60, help="Candlestick size in seconds (Coinbase min=60)")
    p.add_argument("--hours", type=int, default=24 * 14, help="Lookback history size in hours")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--output-csv", default="artifacts/coinbase_dataset.csv")
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.granularity not in COINBASE_ALLOWED_GRANULARITIES:
        supported = ", ".join(map(str, sorted(COINBASE_ALLOWED_GRANULARITIES)))
        raise ValueError(f"Unsupported Coinbase granularity: {args.granularity}. Supported values: {supported}")
    if args.horizon < 1:
        raise ValueError("--horizon must be >= 1")
    if args.lookback < 1:
        raise ValueError("--lookback must be >= 1")


def ensure_output_parent(output_csv: str) -> Path:
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


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
    out_path = ensure_output_parent(args.output_csv)
    pd.DataFrame(df).to_csv(out_path, index=False)

    x, y = pipeline.make_supervised(df)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, device=device)
    results = compare_models(x, y, horizon=args.horizon, cfg=train_cfg)

    print(json.dumps({"rows": len(df), "output_csv": str(out_path), "device": device, "results": results}, indent=2))


if __name__ == "__main__":
    main()
