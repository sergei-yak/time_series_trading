# time_series_trading

Repository restructured around three top-level workflow files:

- `prepare.py`: fixed constants, one-time Coinbase download, feature building, and train/val/test set creation.
- `train.py`: neural architectures, optimizers, and training loop (main file for iterative tuning).
- `program.md`: operating instructions for an agent that loops through hyperparameter search.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python prepare.py
python train.py
```

Artifacts are written to `artifacts/`.
