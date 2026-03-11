from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd
import requests


@dataclass
class CoinbaseConfig:
    product_id: str = "BTC-USD"
    granularity_seconds: int = 60
    base_url: str = "https://api.exchange.coinbase.com"


class CoinbasePublicClient:
    """Minimal Coinbase Exchange public REST client for candles/trades/book."""

    def __init__(self, config: CoinbaseConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _get(self, path: str, params: dict | None = None) -> list | dict:
        response = self.session.get(f"{self.config.base_url}{path}", params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def candles(self, start: datetime, end: datetime) -> pd.DataFrame:
        payload = self._get(
            f"/products/{self.config.product_id}/candles",
            {
                "start": start.replace(tzinfo=timezone.utc).isoformat(),
                "end": end.replace(tzinfo=timezone.utc).isoformat(),
                "granularity": self.config.granularity_seconds,
            },
        )
        df = pd.DataFrame(payload, columns=["time", "low", "high", "open", "close", "base_asset_volume"])
        if df.empty:
            return df
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df.sort_values("time").reset_index(drop=True)

    def trades(self) -> pd.DataFrame:
        payload = self._get(f"/products/{self.config.product_id}/trades")
        df = pd.DataFrame(payload)
        if df.empty:
            return df
        df["time"] = pd.to_datetime(df["time"], utc=True)
        numeric = ["price", "size"]
        for col in numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def level1_book(self) -> dict:
        payload = self._get(f"/products/{self.config.product_id}/book", params={"level": 1})
        bid_price, bid_quantity, _ = payload["bids"][0]
        ask_price, ask_quantity, _ = payload["asks"][0]
        return {
            "bid_price": float(bid_price),
            "bid_quantity": float(bid_quantity),
            "ask_price": float(ask_price),
            "ask_quantity": float(ask_quantity),
        }


def iter_windows(start: datetime, end: datetime, max_points: int = 300, granularity_seconds: int = 60) -> Iterable[tuple[datetime, datetime]]:
    """Coinbase candle endpoint has max records per call. Split range safely."""
    step = timedelta(seconds=max_points * granularity_seconds)
    cursor = start
    while cursor < end:
        upper = min(cursor + step, end)
        yield cursor, upper
        cursor = upper
