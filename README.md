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

## Run experiment (GPU + plots)

```bash
python -m trading_forecast.run_experiment \
  --product-id BTC-USD \
  --granularity 60 \
  --hours 336 \
  --lookback 60 \
  --horizon 5 \
  --epochs 10 \
  --device auto \
  --output-csv artifacts/coinbase_dataset.csv \
  --plots-dir artifacts/plots
```

- `--device auto`: use CUDA if available, otherwise CPU
- `--device cuda`: force GPU and fail fast if CUDA is unavailable
- `--device cpu`: force CPU

The script creates output folders automatically and writes:

- Dataset CSV (default `artifacts/coinbase_dataset.csv`)
- Per-model interactive Plotly HTML diagnostics:
  - Train prediction vs real: `*_train_prediction.html`
  - Test prediction vs real: `*_test_prediction.html`
  - Learning curve (train loss vs test loss by epoch): `*_learning_curve.html`

These diagnostics help you detect overfitting/underfitting.

It also prints JSON with separate train/test metrics (`MAE`, `RMSE`, `MAPE`), final train/test loss, selected device, saved HTML plot paths, and package version.
- Per-model interactive Plotly HTML plots showing real vs predicted close price curves:
  - `artifacts/plots/bilstm_test_prediction.html`
  - `artifacts/plots/transformer_test_prediction.html`
  - `artifacts/plots/lstnet_test_prediction.html`
  - `artifacts/plots/rescnnplus_gru_test_prediction.html`

It also prints JSON with metrics (`MAE`, `RMSE`, `MAPE`), selected device, saved HTML plot paths, and package version.

## If you get `unrecognized arguments: --device ... --plots-dir ...`

You are running an **older installed package version** in your virtualenv.

Run these commands from project root:

```bash
pip uninstall -y time-series-trading
pip install -e . --upgrade
python -m trading_forecast.run_experiment --version
```

Expected output should be `trading_forecast 0.2.0` (or newer). If not, activate the correct `.venv` and reinstall.


## If you get `IndentationError` in `training/compare.py`

This usually means your local file is corrupted (for example, a bad merge/conflict resolution) or you are not running the same source version that is in GitHub.

From project root:

```bash
git status
git checkout -- src/trading_forecast/training/compare.py
pip uninstall -y time-series-trading
pip install -e . --upgrade --no-cache-dir
python -m trading_forecast.run_experiment --version
```

Then re-run the experiment command.

## Test

```bash
pytest -q
```

