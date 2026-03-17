from __future__ import annotations

"""Data preparation entrypoint for the trading project.

Responsibilities:
- hold fixed constants and default configuration
- download market data from Coinbase public REST API
- build a feature table
- create train/validation/test windows for sequence modeling
- persist artifacts used by ``train.py``
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ===== Fixed project constants (edit rarely) =====
PRODUCT_ID = "BTC-USD"
GRANULARITY_SECONDS = 60
LOOKBACK = 60
HORIZON = 5
HOURS_OF_HISTORY = 24 * 28
VAL_RATIO = 0.1
TEST_RATIO = 0.2
RANDOM_SEED = 42
COINBASE_BASE_URL = "https://api.exchange.coinbase.com"
MAX_CANDLES_PER_REQUEST = 300

FEATURE_COLUMNS = [
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
class PrepareConfig:
    product_id: str = PRODUCT_ID
    granularity_seconds: int = GRANULARITY_SECONDS
    lookback: int = LOOKBACK
    horizon: int = HORIZON
    hours_of_history: int = HOURS_OF_HISTORY
    val_ratio: float = VAL_RATIO
    test_ratio: float = TEST_RATIO
    random_seed: int = RANDOM_SEED
    output_dir: Path = Path("artifacts")


class CoinbasePublicClient:
    def __init__(self, product_id: str, base_url: str = COINBASE_BASE_URL):
        self.product_id = product_id
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _get(self, path: str, params: dict | None = None):
        response = self.session.get(f"{self.base_url}{path}", params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def candles(self, start: datetime, end: datetime, granularity_seconds: int) -> pd.DataFrame:
        data = self._get(
            f"/products/{self.product_id}/candles",
            {
                "start": start.replace(tzinfo=UTC).isoformat(),
                "end": end.replace(tzinfo=UTC).isoformat(),
                "granularity": granularity_seconds,
            },
        )
        df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "base_asset_volume"])
        if df.empty:
            return df
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df.sort_values("time")

    def trades(self) -> pd.DataFrame:
        data = self._get(f"/products/{self.product_id}/trades")
        df = pd.DataFrame(data)
        if df.empty:
            return df
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["size"] = pd.to_numeric(df["size"], errors="coerce")
        return df.sort_values("time")

    def level1_book(self) -> dict[str, float]:
        data = self._get(f"/products/{self.product_id}/book", params={"level": 1})
        bid_price, bid_qty, _ = data["bids"][0]
        ask_price, ask_qty, _ = data["asks"][0]
        return {
            "bid_price": float(bid_price),
            "bid_quantity": float(bid_qty),
            "ask_price": float(ask_price),
            "ask_quantity": float(ask_qty),
        }


def iter_candle_windows(start: datetime, end: datetime, granularity_seconds: int):
    step = timedelta(seconds=MAX_CANDLES_PER_REQUEST * granularity_seconds)
    cursor = start
    while cursor < end:
        nxt = min(cursor + step, end)
        yield cursor, nxt
        cursor = nxt


def fetch_feature_frame(cfg: PrepareConfig) -> pd.DataFrame:
    now = datetime.now(tz=UTC)
    start = now - timedelta(hours=cfg.hours_of_history)
    client = CoinbasePublicClient(product_id=cfg.product_id)

    candle_chunks: list[pd.DataFrame] = []
    for w_start, w_end in iter_candle_windows(start, now, cfg.granularity_seconds):
        chunk = client.candles(w_start, w_end, cfg.granularity_seconds)
        if not chunk.empty:
            candle_chunks.append(chunk)

    if not candle_chunks:
        raise ValueError("No candle data returned from Coinbase")

    candles = pd.concat(candle_chunks, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time")
    candles = candles.set_index("time")

    trades = client.trades()
    if trades.empty:
        trade_features = pd.DataFrame(index=candles.index)
    else:
        trades = trades.set_index("time")
        grouped = trades.groupby(pd.Grouper(freq=f"{cfg.granularity_seconds}s"))
        trade_features = pd.DataFrame(index=candles.index)
        trade_features["number_of_trades"] = grouped["trade_id"].count()
        trade_features["quote_asset_volume"] = grouped.apply(lambda x: (x["price"] * x["size"]).sum())
        buys = trades[trades["side"].str.lower() == "buy"]
        buy_grouped = buys.groupby(pd.Grouper(freq=f"{cfg.granularity_seconds}s"))
        trade_features["taker_buy_base_asset_volume"] = buy_grouped["size"].sum()
        trade_features["taker_buy_quote_asset_volume"] = buy_grouped.apply(lambda x: (x["price"] * x["size"]).sum())

    frame = candles.join(trade_features, how="left")
    for key, value in client.level1_book().items():
        frame[key] = value

    for column in FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan

    frame = frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return frame.reset_index(names="timestamp")


def build_supervised(frame: pd.DataFrame, lookback: int, horizon: int):
    scaler = StandardScaler()
    arr = scaler.fit_transform(frame[FEATURE_COLUMNS].astype(float).values)

    close_idx = FEATURE_COLUMNS.index("close")
    xs, ys = [], []
    for i in range(lookback, len(arr) - horizon + 1):
        xs.append(arr[i - lookback : i])
        ys.append(arr[i : i + horizon, close_idx])

    if not xs:
        raise ValueError("Not enough rows for the chosen lookback/horizon")

    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    return x, y, scaler


def save_dataset_splits(x: np.ndarray, y: np.ndarray, cfg: PrepareConfig, output_dir: Path):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=cfg.test_ratio,
        random_state=cfg.random_seed,
        shuffle=False,
    )
    val_size_within_train = cfg.val_ratio / (1.0 - cfg.test_ratio)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=val_size_within_train,
        random_state=cfg.random_seed,
        shuffle=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "dataset.npz",
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )


def main() -> None:
    cfg = PrepareConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    frame = fetch_feature_frame(cfg)
    frame.to_csv(cfg.output_dir / "raw_features.csv", index=False)

    x, y, _ = build_supervised(frame, lookback=cfg.lookback, horizon=cfg.horizon)
    save_dataset_splits(x, y, cfg=cfg, output_dir=cfg.output_dir)

    print(f"Prepared dataset saved to: {cfg.output_dir / 'dataset.npz'}")
    print(f"Rows: {len(frame)} | samples: {len(x)} | lookback={cfg.lookback} | horizon={cfg.horizon}")


if __name__ == "__main__":
    main()
