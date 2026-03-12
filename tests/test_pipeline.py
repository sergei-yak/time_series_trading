from datetime import datetime, timedelta, timezone

import pandas as pd

from trading_forecast.data.pipeline import DataPipeline, DatasetConfig


class DummyClient:
    def candles(self, start, end):
        idx = pd.date_range(start=start, end=end - timedelta(minutes=1), freq="1min", tz=timezone.utc)
        return pd.DataFrame(
            {
                "time": idx,
                "low": 1.0,
                "high": 2.0,
                "open": 1.2,
                "close": 1.8,
                "base_asset_volume": 100.0,
            }
        )

    def trades(self):
        return pd.DataFrame(
            {
                "time": pd.date_range(datetime.now(tz=timezone.utc) - timedelta(minutes=10), periods=10, freq="1min"),
                "trade_id": range(10),
                "price": 2.0,
                "size": 1.0,
                "side": ["buy", "sell"] * 5,
            }
        )

    def level1_book(self):
        return {"bid_price": 1.7, "bid_quantity": 3.0, "ask_price": 1.8, "ask_quantity": 2.5}


def test_pipeline_outputs_required_columns():
    cfg = DatasetConfig(lookback=5, horizon=2, granularity_seconds=60)
    pipeline = DataPipeline(DummyClient(), cfg)
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(hours=1)
    df = pipeline.fetch_dataset(start, end)
    x, y = pipeline.make_supervised(df)

    assert len(df.columns) == 14
    assert x.shape[-1] == 13
    assert y.shape[-1] == 2
