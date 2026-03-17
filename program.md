# Hyperparameter Search Program

This file is an operating guide for an autonomous agent that improves `train.py`.

## Goal
Find model/hyperparameter settings that improve test metrics for Coinbase BTC forecasting.

## Experiment tracking file
Before running training experiments, initialize `results.tsv` with **header only**:

```tsv
commit	val	status	description
```

- The first baseline run is recorded as the first data row after initialization.
- Log every experiment to `results.tsv` as **tab-separated** values (NOT comma-separated).
- Required columns:
  - `commit`: commit hash (or temporary run id before commit)
  - `val`: primary validation metric used for model selection
  - `status`: e.g., `baseline`, `improved`, `regressed`, `unstable`, `reverted`
  - `description`: short explanation of what changed

## Loop
1. **Prepare data once**
   - Run: `python prepare.py`
   - Verify `artifacts/dataset.npz` exists.

2. **Baseline run**
   - Run: `python train.py`
   - Save baseline test metrics.
   - Add a baseline row in `results.tsv`.

3. **Tune one axis at a time**
   - Allowed edit scope: `train.py` only.
   - Suggested order:
     - optimizer/lr schedule
     - batch size and epochs
     - model depth/width
     - regularization (dropout, weight decay)
     - loss variants (e.g., SmoothL1)

4. **Track each attempt**
   - Log each run to `results.tsv` (tab-separated).
   - Record a short changelog entry: what changed, why, and resulting metrics.
   - Keep only changes that beat baseline validation performance.

5. **Avoid overfitting**
   - Prefer configurations that improve validation and test together.
   - If validation improves but test degrades, mark as unstable and revert.

6. **Finalize**
   - Keep the best configuration in `train.py`.
   - Output a concise summary table with model and test metrics.
   - Ensure `results.tsv` includes baseline + all tried experiments.

## Constraints
- Do not edit `prepare.py` during hyperparameter search unless data integrity bug is confirmed.
- Do not hard-code results.
- Keep the script runnable from repo root with:
  - `python prepare.py`
  - `python train.py`
