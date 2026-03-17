# Time Series Trading Autoresearch

This repo is an experiment in autonomous model research for short-horizon BTC forecasting.

The core idea is simple:
- prepare a fixed dataset from Coinbase
- train a small model zoo on that dataset
- let an agent run many experiments
- keep only the changes that improve RMSE

## Repository Layout

- `prepare.py`
  - downloads and prepares Coinbase feature data
  - creates train/val/test windows
  - writes `artifacts/raw_features.csv` and `artifacts/dataset.npz`
- `train.py`
  - defines model architectures and training logic
  - runs model experiments
  - logs experiment rows to `results.tsv`
- `program.md`
  - operating instructions for autonomous experimentation
  - defines setup, loop, stop signals, and keep/discard rules
- `results.tsv`
  - experiment ledger with one row per model per run:
    - `commit`, `RMSE`, `status`, `model`, `description`

## Quick Start

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
python prepare.py
python train.py --status auto --description "baseline"
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python prepare.py
python train.py --status auto --description "baseline"
```

## Experiment Logging

`train.py` appends rows to `results.tsv` with:

- `status=keep` if a model's RMSE is the best (lowest) seen so far for that model
- `status=discard` otherwise
- `status=crash` when a run fails and no usable metric is produced

This means keep/discard is model-specific, not global.

## Status Values

The `results.tsv` `status` column is restricted to:
- `keep`
- `discard`
- `crash`

## Autonomous Mode

To run this as an autonomous research loop, follow `program.md`.

In short:
- run setup (`prepare.py`, baseline)
- iterate changes to `train.py`
- run experiments repeatedly
- stop only on explicit stop signal

## Notes

- `prepare.py` should be treated as read-only during normal hyperparameter search.
- Main tuning surface is `train.py`.
- Lower RMSE is better.
