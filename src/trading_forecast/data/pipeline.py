from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from trading_forecast.data.coinbase_client import CoinbasePublicClient, iter_windows

REQUIRED_COLUMNS = [
    "open",
    "close",
    "high",
    "low",
    "base_asset_volume",
    "number_of_trades",
    "quote_asset_volume",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "bid_price",
    "bid_quantity",
    "ask_price",
    "ask_quantity",
]


@dataclass
class DatasetConfig:
    lookback: int = 60
    horizon: int = 5
    granularity_seconds: int = 60


class DataPipeline:
    def __init__(self, client: CoinbasePublicClient, cfg: DatasetConfig):
        self.client = client
        self.cfg = cfg
        self.scaler = StandardScaler()

    def fetch_dataset(self, start: datetime, end: datetime) -> pd.DataFrame:
        candles = []
        for w_start, w_end in iter_windows(start, end, granularity_seconds=self.cfg.granularity_seconds):
            chunk = self.client.candles(w_start, w_end)
            if not chunk.empty:
                candles.append(chunk)
        if not candles:
            raise ValueError("No candle data returned. Check symbol/time range.")

        frame = pd.concat(candles, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time")
        frame = frame.set_index("time")

        trades = self.client.trades()
        if trades.empty:
            trade_agg = pd.DataFrame(index=frame.index)
        else:
            trades = trades.set_index("time").sort_index()
            bucket = f"{self.cfg.granularity_seconds}s"
            trade_agg = pd.DataFrame(index=frame.index)
            grouped = trades.groupby(pd.Grouper(freq=bucket))
            trade_agg["number_of_trades"] = grouped["trade_id"].count()
            trade_agg["quote_asset_volume"] = grouped.apply(lambda x: (x["price"] * x["size"]).sum())
            buys = trades[trades["side"].str.lower() == "buy"]
            buy_grouped = buys.groupby(pd.Grouper(freq=bucket))
            trade_agg["taker_buy_base_asset_volume"] = buy_grouped["size"].sum()
            trade_agg["taker_buy_quote_asset_volume"] = buy_grouped.apply(lambda x: (x["price"] * x["size"]).sum())

        frame = frame.join(trade_agg, how="left")

        # Coinbase public REST does not provide historical L1 snapshots. Use latest snapshot as fallback.
        book = self.client.level1_book()
        for key, val in book.items():
            frame[key] = val

        for col in REQUIRED_COLUMNS:
            if col not in frame.columns:
                frame[col] = np.nan

        frame = frame[REQUIRED_COLUMNS].replace([np.inf, -np.inf], np.nan)
        frame = frame.ffill().bfill()
        return frame.reset_index(names="timestamp")

    def make_supervised(self, df: pd.DataFrame):
        features = df[REQUIRED_COLUMNS].astype(float).values
        scaled = self.scaler.fit_transform(features)

        xs, ys = [], []
        target_idx = REQUIRED_COLUMNS.index("close")
        for i in range(self.cfg.lookback, len(df) - self.cfg.horizon + 1):
            xs.append(scaled[i - self.cfg.lookback : i])
            ys.append(scaled[i : i + self.cfg.horizon, target_idx])

        x_arr = np.asarray(xs, dtype=np.float32)
        y_arr = np.asarray(ys, dtype=np.float32)
        if x_arr.size == 0:
            raise ValueError("Not enough rows for lookback+horizon configuration")
        return x_arr, y_arr


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)
