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
TRADE_LADDER_BPS_BOUNDS = [2.0, 5.0, 10.0, 20.0, 50.0]
MAX_TRADE_PAGES = 200
MIN_MICROSTRUCT_NONZERO_RATIO = 0.02
ORDERBOOK_LEVELS = 10

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
    "taker_sell_base_asset_volume",
    "taker_sell_quote_asset_volume",
    "buy_volume_share_base",
    "buy_volume_share_quote",
    "trade_imbalance_base",
    "trade_imbalance_quote",
    "trade_vwap",
    "return_1m",
    "return_5m",
    "return_15m",
    "return_30m",
    "return_60m",
    "log_return_1m",
    "realized_vol_5m",
    "realized_vol_15m",
    "realized_vol_30m",
    "realized_vol_60m",
    "momentum_sum_5m",
    "momentum_sum_15m",
    "momentum_sum_30m",
    "momentum_sum_60m",
    "momentum_mean_5m",
    "momentum_mean_15m",
    "momentum_mean_30m",
    "momentum_mean_60m",
    "hl_range_frac",
    "oc_body_frac",
    "upper_wick_frac",
    "lower_wick_frac",
    "log_base_volume",
    "log_quote_volume",
    "base_volume_zscore_30m",
    "quote_volume_zscore_30m",
    "trade_count_roll5",
    "trade_count_roll15",
    "illiquidity_amihud",
    "ema_9_ratio",
    "ema_21_ratio",
    "ema_55_ratio",
    "sma_20_ratio",
    "sma_60_ratio",
    "ema_cross_9_21",
    "ema_cross_21_55",
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "atr_14_frac",
    "bb_pos_20",
    "bb_width_20",
    "trend_strength_20_60",
    "vol_regime_code",
    "trend_regime_code",
    "liquidity_regime_code",
    "midprice",
    "spread_bps",
    "microprice",
    "obi_5",
    "obi_10",
    "bid_depth_5bps",
    "ask_depth_5bps",
    "depth_imb_5bps",
    "book_slope_bid",
    "book_slope_ask",
    "d_midprice_1m",
    "d_spread_bps_1m",
    "d_obi_5_1m",
    "minute_sin",
    "minute_cos",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]
for i in range(1, len(TRADE_LADDER_BPS_BOUNDS) + 1):
    FEATURE_COLUMNS.extend(
        [
            f"trade_buy_notional_l{i}",
            f"trade_sell_notional_l{i}",
            f"trade_notional_imbalance_l{i}",
        ]
    )


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
    max_trade_pages: int = MAX_TRADE_PAGES
    min_microstruct_nonzero_ratio: float = MIN_MICROSTRUCT_NONZERO_RATIO
    orderbook_levels: int = ORDERBOOK_LEVELS
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

    def trades_history(self, start: datetime, end: datetime, max_pages: int = MAX_TRADE_PAGES) -> pd.DataFrame:
        all_rows: list[dict] = []
        cursor_after: str | None = None
        seen_cursors: set[str] = set()
        path = f"/products/{self.product_id}/trades"

        for _ in range(max_pages):
            params: dict[str, str | int] = {"limit": 100}
            if cursor_after is not None:
                params["after"] = cursor_after

            response = self.session.get(f"{self.base_url}{path}", params=params, timeout=30)
            response.raise_for_status()
            page = response.json()
            if not page:
                break
            all_rows.extend(page)

            page_df = pd.DataFrame(page)
            if "time" in page_df.columns and not page_df.empty:
                page_df["time"] = pd.to_datetime(page_df["time"], utc=True, errors="coerce")
                oldest = page_df["time"].min()
                if pd.notna(oldest) and oldest <= start:
                    break

            next_cursor = response.headers.get("CB-AFTER")
            if not next_cursor or next_cursor in seen_cursors:
                break
            seen_cursors.add(next_cursor)
            cursor_after = next_cursor

        df = pd.DataFrame(all_rows)
        if df.empty:
            return df
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["size"] = pd.to_numeric(df["size"], errors="coerce")
        df = df.dropna(subset=["time", "price", "size"])
        df = df[(df["time"] >= start) & (df["time"] <= end)]
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


def _book_slope_from_sizes(size_arr: np.ndarray) -> float:
    n = len(size_arr)
    if n < 2:
        return float("nan")
    x = np.arange(1, n + 1, dtype=float)
    return float(np.polyfit(x, size_arr, 1)[0])


def load_orderbook_feature_frame(output_dir: Path, levels: int) -> pd.DataFrame:
    snapshot_path = output_dir / "orderbook_snapshots.csv"
    if not snapshot_path.exists():
        return pd.DataFrame()

    ob = pd.read_csv(snapshot_path)
    if ob.empty or "timestamp" not in ob.columns:
        return pd.DataFrame()

    ob["timestamp"] = pd.to_datetime(ob["timestamp"], utc=True, errors="coerce")
    ob = ob.dropna(subset=["timestamp"])
    if ob.empty:
        return pd.DataFrame()

    def get_levels(prefix: str, row: pd.Series) -> np.ndarray:
        vals: list[float] = []
        for i in range(1, levels + 1):
            col = f"{prefix}_{i}"
            if col in row and pd.notna(row[col]):
                vals.append(float(row[col]))
        return np.asarray(vals, dtype=float)

    def row_to_features(row: pd.Series) -> pd.Series:
        eps = 1e-12
        bid_px = get_levels("bid_price", row)
        bid_sz = get_levels("bid_size", row)
        ask_px = get_levels("ask_price", row)
        ask_sz = get_levels("ask_size", row)

        if len(bid_px) == 0 or len(ask_px) == 0 or len(bid_sz) == 0 or len(ask_sz) == 0:
            return pd.Series(
                {
                    "midprice": np.nan,
                    "spread_bps": np.nan,
                    "microprice": np.nan,
                    "obi_5": np.nan,
                    "obi_10": np.nan,
                    "bid_depth_5bps": np.nan,
                    "ask_depth_5bps": np.nan,
                    "depth_imb_5bps": np.nan,
                    "book_slope_bid": np.nan,
                    "book_slope_ask": np.nan,
                }
            )

        best_bid = bid_px[0]
        best_ask = ask_px[0]
        bb_size = bid_sz[0]
        ba_size = ask_sz[0]
        mid = (best_bid + best_ask) / 2.0
        spread_bps = 1e4 * (best_ask - best_bid) / (mid + eps)
        microprice = (best_ask * bb_size + best_bid * ba_size) / (bb_size + ba_size + eps)

        n5 = min(5, len(bid_sz), len(ask_sz))
        n10 = min(10, len(bid_sz), len(ask_sz))
        obi5 = (bid_sz[:n5].sum() - ask_sz[:n5].sum()) / (bid_sz[:n5].sum() + ask_sz[:n5].sum() + eps)
        obi10 = (bid_sz[:n10].sum() - ask_sz[:n10].sum()) / (bid_sz[:n10].sum() + ask_sz[:n10].sum() + eps)

        bid_depth_5bps = bid_sz[bid_px >= mid * (1.0 - 5.0 / 10000.0)].sum()
        ask_depth_5bps = ask_sz[ask_px <= mid * (1.0 + 5.0 / 10000.0)].sum()
        depth_imb_5bps = (bid_depth_5bps - ask_depth_5bps) / (bid_depth_5bps + ask_depth_5bps + eps)

        slope_bid = _book_slope_from_sizes(bid_sz[: min(10, len(bid_sz))])
        slope_ask = _book_slope_from_sizes(ask_sz[: min(10, len(ask_sz))])

        return pd.Series(
            {
                "midprice": mid,
                "spread_bps": spread_bps,
                "microprice": microprice,
                "obi_5": obi5,
                "obi_10": obi10,
                "bid_depth_5bps": bid_depth_5bps,
                "ask_depth_5bps": ask_depth_5bps,
                "depth_imb_5bps": depth_imb_5bps,
                "book_slope_bid": slope_bid,
                "book_slope_ask": slope_ask,
            }
        )

    feats = ob.apply(row_to_features, axis=1)
    feats.index = ob["timestamp"]
    feats = feats.groupby(pd.Grouper(freq="1min")).mean()
    return feats


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

    trades = client.trades_history(start=start, end=now, max_pages=cfg.max_trade_pages)
    if trades.empty:
        trade_features = pd.DataFrame(index=candles.index)
    else:
        trades = trades.set_index("time")
        trades["side"] = trades["side"].astype(str).str.lower()
        trades["notional"] = trades["price"] * trades["size"]
        grouped = trades.groupby(pd.Grouper(freq=f"{cfg.granularity_seconds}s"))
        trade_features = pd.DataFrame(index=candles.index)
        trade_features["number_of_trades"] = grouped.size()
        trade_features["quote_asset_volume"] = grouped["notional"].sum()
        trade_features["base_asset_volume_trades"] = grouped["size"].sum()

        buys = trades[trades["side"] == "buy"]
        buy_grouped = buys.groupby(pd.Grouper(freq=f"{cfg.granularity_seconds}s"))
        trade_features["taker_buy_base_asset_volume"] = buy_grouped["size"].sum()
        trade_features["taker_buy_quote_asset_volume"] = buy_grouped["notional"].sum()

        # Horizontal microstructure proxy: per-minute trade ladder by distance from close.
        bucket_freq = f"{cfg.granularity_seconds}s"
        ladder_df = trades.copy()
        ladder_df["bucket"] = ladder_df.index.floor(bucket_freq)
        close_map = candles["close"].to_dict()
        ladder_df["ref_close"] = ladder_df["bucket"].map(close_map)
        ladder_df = ladder_df[ladder_df["ref_close"].notna()].copy()
        eps = 1e-9
        ladder_df["abs_bps"] = (1e4 * (ladder_df["price"] / (ladder_df["ref_close"] + eps) - 1.0)).abs()

        lower = 0.0
        for i, upper in enumerate(TRADE_LADDER_BPS_BOUNDS, start=1):
            in_level = (ladder_df["abs_bps"] > lower) & (ladder_df["abs_bps"] <= upper)
            buy_level = ladder_df[in_level & (ladder_df["side"] == "buy")]
            sell_level = ladder_df[in_level & (ladder_df["side"] == "sell")]
            trade_features[f"trade_buy_notional_l{i}"] = buy_level.groupby("bucket")["notional"].sum()
            trade_features[f"trade_sell_notional_l{i}"] = sell_level.groupby("bucket")["notional"].sum()
            lower = upper

    frame = candles.join(trade_features, how="left")
    orderbook_features = load_orderbook_feature_frame(cfg.output_dir, cfg.orderbook_levels)
    if not orderbook_features.empty:
        frame = frame.join(orderbook_features, how="left")

    eps = 1e-9

    # Multi-horizon returns and realized volatility.
    frame["return_1m"] = frame["close"].pct_change(1)
    frame["return_5m"] = frame["close"].pct_change(5)
    frame["return_15m"] = frame["close"].pct_change(15)
    frame["return_30m"] = frame["close"].pct_change(30)
    frame["return_60m"] = frame["close"].pct_change(60)
    frame["log_return_1m"] = np.log(frame["close"].clip(lower=eps)).diff(1)
    frame["realized_vol_5m"] = frame["log_return_1m"].rolling(5, min_periods=1).std()
    frame["realized_vol_15m"] = frame["log_return_1m"].rolling(15, min_periods=1).std()
    frame["realized_vol_30m"] = frame["log_return_1m"].rolling(30, min_periods=1).std()
    frame["realized_vol_60m"] = frame["log_return_1m"].rolling(60, min_periods=1).std()
    frame["momentum_sum_5m"] = frame["return_1m"].rolling(5, min_periods=1).sum()
    frame["momentum_sum_15m"] = frame["return_1m"].rolling(15, min_periods=1).sum()
    frame["momentum_sum_30m"] = frame["return_1m"].rolling(30, min_periods=1).sum()
    frame["momentum_sum_60m"] = frame["return_1m"].rolling(60, min_periods=1).sum()
    frame["momentum_mean_5m"] = frame["return_1m"].rolling(5, min_periods=1).mean()
    frame["momentum_mean_15m"] = frame["return_1m"].rolling(15, min_periods=1).mean()
    frame["momentum_mean_30m"] = frame["return_1m"].rolling(30, min_periods=1).mean()
    frame["momentum_mean_60m"] = frame["return_1m"].rolling(60, min_periods=1).mean()

    # Candle-shape dynamics.
    frame["hl_range_frac"] = (frame["high"] - frame["low"]) / (frame["close"].abs() + eps)
    frame["oc_body_frac"] = (frame["close"] - frame["open"]).abs() / (frame["close"].abs() + eps)
    frame["upper_wick_frac"] = (frame["high"] - np.maximum(frame["open"], frame["close"])) / (frame["close"].abs() + eps)
    frame["lower_wick_frac"] = (np.minimum(frame["open"], frame["close"]) - frame["low"]) / (frame["close"].abs() + eps)
    frame["log_base_volume"] = np.log1p(frame["base_asset_volume"].clip(lower=0.0))
    frame["log_quote_volume"] = np.log1p(frame["quote_asset_volume"].clip(lower=0.0))

    # Volume and trade-flow pressure.
    if "base_asset_volume_trades" in frame.columns:
        frame["base_asset_volume"] = frame["base_asset_volume"].fillna(frame["base_asset_volume_trades"])

    base_vol_mean_30 = frame["base_asset_volume"].rolling(30, min_periods=1).mean()
    base_vol_std_30 = frame["base_asset_volume"].rolling(30, min_periods=1).std()
    quote_vol_mean_30 = frame["quote_asset_volume"].rolling(30, min_periods=1).mean()
    quote_vol_std_30 = frame["quote_asset_volume"].rolling(30, min_periods=1).std()
    frame["base_volume_zscore_30m"] = (frame["base_asset_volume"] - base_vol_mean_30) / (base_vol_std_30 + eps)
    frame["quote_volume_zscore_30m"] = (frame["quote_asset_volume"] - quote_vol_mean_30) / (quote_vol_std_30 + eps)
    frame["trade_count_roll5"] = frame["number_of_trades"].rolling(5, min_periods=1).mean()
    frame["trade_count_roll15"] = frame["number_of_trades"].rolling(15, min_periods=1).mean()

    frame["taker_sell_base_asset_volume"] = frame["base_asset_volume"] - frame["taker_buy_base_asset_volume"]
    frame["taker_sell_quote_asset_volume"] = frame["quote_asset_volume"] - frame["taker_buy_quote_asset_volume"]
    frame["buy_volume_share_base"] = frame["taker_buy_base_asset_volume"] / (frame["base_asset_volume"] + eps)
    frame["buy_volume_share_quote"] = frame["taker_buy_quote_asset_volume"] / (frame["quote_asset_volume"] + eps)
    frame["trade_imbalance_base"] = (frame["taker_buy_base_asset_volume"] - frame["taker_sell_base_asset_volume"]) / (
        frame["base_asset_volume"] + eps
    )
    frame["trade_imbalance_quote"] = (
        frame["taker_buy_quote_asset_volume"] - frame["taker_sell_quote_asset_volume"]
    ) / (frame["quote_asset_volume"] + eps)
    frame["trade_vwap"] = frame["quote_asset_volume"] / (frame["base_asset_volume"] + eps)
    frame["illiquidity_amihud"] = frame["return_1m"].abs() / (frame["quote_asset_volume"] + eps)
    for i in range(1, len(TRADE_LADDER_BPS_BOUNDS) + 1):
        buy_col = f"trade_buy_notional_l{i}"
        sell_col = f"trade_sell_notional_l{i}"
        frame[f"trade_notional_imbalance_l{i}"] = (frame[buy_col] - frame[sell_col]) / (frame[buy_col] + frame[sell_col] + eps)

    # Technical indicator priors.
    ema9 = frame["close"].ewm(span=9, adjust=False).mean()
    ema21 = frame["close"].ewm(span=21, adjust=False).mean()
    ema55 = frame["close"].ewm(span=55, adjust=False).mean()
    sma20 = frame["close"].rolling(20, min_periods=1).mean()
    sma60 = frame["close"].rolling(60, min_periods=1).mean()
    frame["ema_9_ratio"] = frame["close"] / (ema9 + eps) - 1.0
    frame["ema_21_ratio"] = frame["close"] / (ema21 + eps) - 1.0
    frame["ema_55_ratio"] = frame["close"] / (ema55 + eps) - 1.0
    frame["sma_20_ratio"] = frame["close"] / (sma20 + eps) - 1.0
    frame["sma_60_ratio"] = frame["close"] / (sma60 + eps) - 1.0
    frame["ema_cross_9_21"] = (ema9 - ema21) / (frame["close"] + eps)
    frame["ema_cross_21_55"] = (ema21 - ema55) / (frame["close"] + eps)

    delta = frame["close"].diff(1)
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + eps)
    frame["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    ema12 = frame["close"].ewm(span=12, adjust=False).mean()
    ema26 = frame["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    frame["macd_line"] = macd_line / (frame["close"] + eps)
    frame["macd_signal"] = macd_signal / (frame["close"] + eps)
    frame["macd_hist"] = (macd_line - macd_signal) / (frame["close"] + eps)

    prev_close = frame["close"].shift(1)
    tr = np.maximum(
        frame["high"] - frame["low"],
        np.maximum((frame["high"] - prev_close).abs(), (frame["low"] - prev_close).abs()),
    )
    atr14 = tr.rolling(14, min_periods=1).mean()
    frame["atr_14_frac"] = atr14 / (frame["close"] + eps)

    rolling_std_20 = frame["close"].rolling(20, min_periods=1).std()
    bb_upper = sma20 + 2.0 * rolling_std_20
    bb_lower = sma20 - 2.0 * rolling_std_20
    frame["bb_pos_20"] = (frame["close"] - bb_lower) / (bb_upper - bb_lower + eps)
    frame["bb_width_20"] = (bb_upper - bb_lower) / (sma20 + eps)
    frame["trend_strength_20_60"] = (sma20 - sma60) / (frame["close"] + eps)

    # Regime/context codes.
    vol_q1 = frame["realized_vol_30m"].quantile(0.33)
    vol_q2 = frame["realized_vol_30m"].quantile(0.66)
    frame["vol_regime_code"] = np.select(
        [frame["realized_vol_30m"] <= vol_q1, frame["realized_vol_30m"] <= vol_q2],
        [0.0, 1.0],
        default=2.0,
    )
    frame["trend_regime_code"] = np.select(
        [frame["trend_strength_20_60"] > 0.001, frame["trend_strength_20_60"] < -0.001],
        [2.0, 0.0],
        default=1.0,
    )
    spread_proxy = frame["hl_range_frac"].rolling(5, min_periods=1).mean()
    liq_q1 = spread_proxy.quantile(0.33)
    liq_q2 = spread_proxy.quantile(0.66)
    frame["liquidity_regime_code"] = np.select(
        [spread_proxy <= liq_q1, spread_proxy <= liq_q2],
        [2.0, 1.0],
        default=0.0,
    )

    # Order-book dynamics (if historical snapshots are available).
    if "midprice" in frame.columns:
        frame["d_midprice_1m"] = frame["midprice"].pct_change(1)
    if "spread_bps" in frame.columns:
        frame["d_spread_bps_1m"] = frame["spread_bps"].diff(1)
    if "obi_5" in frame.columns:
        frame["d_obi_5_1m"] = frame["obi_5"].diff(1)

    # Calendar regime features.
    minute = frame.index.minute.astype(float)
    hour = frame.index.hour.astype(float)
    dow = frame.index.dayofweek.astype(float)
    frame["minute_sin"] = np.sin(2 * np.pi * minute / 60.0)
    frame["minute_cos"] = np.cos(2 * np.pi * minute / 60.0)
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    frame["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    frame["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame.reset_index(names="timestamp")


def select_feature_columns(frame: pd.DataFrame, cfg: PrepareConfig) -> list[str]:
    selected = [c for c in FEATURE_COLUMNS if c in frame.columns]
    dropped_missing: list[str] = []
    kept: list[str] = []
    total = len(frame)
    for col in selected:
        non_na_ratio = float(frame[col].notna().sum()) / max(total, 1)
        if non_na_ratio == 0.0:
            dropped_missing.append(col)
        else:
            kept.append(col)
    selected = kept
    micro_cols = [
        c
        for c in selected
        if c.startswith("trade_")
        or c.startswith("taker_")
        or c in {"number_of_trades", "quote_asset_volume", "buy_volume_share_base", "buy_volume_share_quote"}
        or c
        in {
            "midprice",
            "spread_bps",
            "microprice",
            "obi_5",
            "obi_10",
            "bid_depth_5bps",
            "ask_depth_5bps",
            "depth_imb_5bps",
            "book_slope_bid",
            "book_slope_ask",
            "d_midprice_1m",
            "d_spread_bps_1m",
            "d_obi_5_1m",
        }
    ]
    dropped_sparse: list[str] = []
    for col in micro_cols:
        nonzero_ratio = float((frame[col].astype(float).abs() > 1e-12).sum()) / max(total, 1)
        if nonzero_ratio < cfg.min_microstruct_nonzero_ratio:
            selected.remove(col)
            dropped_sparse.append(col)
    if dropped_missing:
        print(f"Dropping {len(dropped_missing)} fully-missing features.")
    if dropped_sparse:
        print(
            f"Dropping {len(dropped_sparse)} sparse microstructure features "
            f"(nonzero_ratio < {cfg.min_microstruct_nonzero_ratio:.2%})."
        )
    return selected


def sanitize_selected_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    out = frame[feature_columns].copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.ffill().bfill().fillna(0.0)
    return out


def build_supervised(frame: pd.DataFrame, lookback: int, horizon: int, feature_columns: list[str]):
    frame = sanitize_selected_frame(frame, feature_columns)
    scaler = StandardScaler()
    arr = scaler.fit_transform(frame[feature_columns].astype(float).values)

    close_idx = feature_columns.index("close")
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
    selected_features = select_feature_columns(frame, cfg)
    frame_selected = sanitize_selected_frame(frame, selected_features)
    (cfg.output_dir / "feature_columns.txt").write_text("\n".join(selected_features) + "\n", encoding="utf-8")

    raw_path = cfg.output_dir / "raw_features.csv"
    try:
        frame_selected.to_csv(raw_path, index=False)
    except PermissionError:
        fallback = cfg.output_dir / f"raw_features_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}.csv"
        frame_selected.to_csv(fallback, index=False)
        print(f"Warning: could not write {raw_path} (file locked). Wrote {fallback} instead.")

    x, y, _ = build_supervised(frame_selected, lookback=cfg.lookback, horizon=cfg.horizon, feature_columns=selected_features)
    save_dataset_splits(x, y, cfg=cfg, output_dir=cfg.output_dir)

    print(f"Prepared dataset saved to: {cfg.output_dir / 'dataset.npz'}")
    print(
        f"Rows: {len(frame_selected)} | samples: {len(x)} | features={len(selected_features)} "
        f"| lookback={cfg.lookback} | horizon={cfg.horizon}"
    )


if __name__ == "__main__":
    main()
