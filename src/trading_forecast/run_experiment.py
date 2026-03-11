from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import torch

from trading_forecast.data.coinbase_client import CoinbaseConfig, CoinbasePublicClient
from trading_forecast.data.pipeline import DataPipeline, DatasetConfig, utc_now
from trading_forecast.training.compare import TrainConfig, compare_models


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


def main():
    args = parse_args()
    end = utc_now()
    start = end - timedelta(hours=args.hours)

    cb_cfg = CoinbaseConfig(product_id=args.product_id, granularity_seconds=args.granularity)
    pipeline_cfg = DatasetConfig(lookback=args.lookback, horizon=args.horizon, granularity_seconds=args.granularity)

    client = CoinbasePublicClient(cb_cfg)
    pipeline = DataPipeline(client, pipeline_cfg)

    df = pipeline.fetch_dataset(start=start, end=end)
    pd.DataFrame(df).to_csv(args.output_csv, index=False)

    x, y = pipeline.make_supervised(df)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, device=device)
    results = compare_models(x, y, horizon=args.horizon, cfg=train_cfg)

    print(json.dumps({"rows": len(df), "device": device, "results": results}, indent=2))


if __name__ == "__main__":
    main()
