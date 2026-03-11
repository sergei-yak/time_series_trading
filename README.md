# Coinbase BTC Futures-style Forecasting (PyTorch)

This project pulls BTC market data from Coinbase public REST API, preprocesses it, and compares four neural network forecasting architectures for multi-step prediction (next 5+ timesteps):

- Bidirectional LSTM
- Transformer Encoder
- LSTNet (simplified CNN+GRU variant)
- ResCNN + GRU

## Notes on timeframe

Coinbase Exchange candle API supports **minimum 60 seconds** granularity. You can change granularity with `--granularity` (e.g. `60`, `300`, `900`).

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

The script saves dataset CSV and prints MAE/RMSE/MAPE for each model.

## Test

```bash
pytest -q
```
