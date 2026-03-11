# Coinbase BTC Futures-style Forecasting (PyTorch)

This project pulls BTC market data from Coinbase public REST API, preprocesses it, and compares four neural network forecasting architectures for multi-step prediction (next 5+ timesteps):

- Bidirectional LSTM
- Transformer Encoder
- LSTNet (simplified CNN+GRU variant)
- ResCNN + GRU

## Notes on timeframe

Coinbase Exchange candle API supports **minimum 60 seconds** granularity and accepts the following values:

- `60` (1 min)
- `300` (5 min)
- `900` (15 min)
- `3600` (1 hour)
- `21600` (6 hours)
- `86400` (1 day)

You can change it via `--granularity`.

## Features produced

The pipeline outputs:

- Open price, Close price, High price, Low price
- Base asset volume
- Number of trades (aggregated from trade feed)
- Quote asset volume (`sum(price * size)`)
- Taker buy base/quote volume (buy-side proxy from trade side)
- L1 order book fields: bid/ask price and quantity

> Coinbase public REST does not provide historical L1 snapshots for every past candle. Current L1 book is used as fallback for those columns.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run experiment

```bash
python -m trading_forecast.run_experiment \
  --product-id BTC-USD \
  --granularity 60 \
  --hours 336 \
  --lookback 60 \
  --horizon 5 \
  --epochs 10
```

The script creates the output directory automatically (default: `artifacts/`) before saving the dataset CSV and then prints MAE/RMSE/MAPE for each model.

## Test

```bash
pytest -q
```
