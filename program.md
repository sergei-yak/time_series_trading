# Autonomous Research Program

This file is the operating guide for an autonomous research agent improving `train.py`.

## Objective
Drive validation performance down (lower is better) on Coinbase BTC forecasting while keeping the training script stable and runnable.

## Setup (one-time)
Before the experiment loop begins:

1. Read in-scope files for context:
   - `README.md`
   - `prepare.py` (read-only unless confirmed data bug)
   - `train.py` (primary edit surface)
2. Install dependencies (GPU setup):
   - `python -m venv .venv`
   - Activate venv
   - `pip install --upgrade pip setuptools wheel`
   - `pip install -e . --no-deps`
   - `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`
   - If editable install fails with a temp permission error, set local temp/cache vars and retry:
     - PowerShell: `$env:TEMP="$PWD\.tmp"; $env:TMP="$PWD\.tmp"; $env:PIP_CACHE_DIR="$PWD\.pip-cache"`
3. Prepare data once:
   - Run: `python prepare.py`
   - Verify: `artifacts/dataset.npz` exists.
4. Initialize `results.tsv`.  Create results.tsv with just the header row and 4 columns:

```tsv
commit	RMSE	status	model	description
```

5. Run baseline once:
   - Run: `python train.py --status keep --description "baseline"`
   - Confirm baseline rows were appended (one row per model).

After setup is complete, start autonomous experimentation immediately.

## Allowed Edit Scope
- Default: edit `train.py` only.
- Do not edit `prepare.py` unless a data-integrity bug is confirmed.
- Do not hard-code outcomes.

## Logging Contract
`results.tsv` is tab-separated (NOT comma-separated).

- Columns:
  - `commit`: commit hash or run id
  - `RMSE`: run RMSE metric (lower is better)
  - `status`: `keep`, `discard`, `crash`
  - `model`: model name (e.g., `bilstm`, `transformer`, `lstnet`, `rescnn_gru`)
  - `description`: short change summary
- One experiment run logs one row per model (same `commit`, model-specific `description`).
- For crashes, write `RMSE=0.000000` if no usable metric is produced and set `status=crash`.

## Autonomy Contract (Strict)
After setup and baseline are complete, immediately enter the experiment loop.

- Do not ask "should I continue?" or "is this a good stopping point?".
- Do not stop after a single experiment.
- Do not pause for approval between experiments.
- Continue until an explicit stop signal is received.

Stop signals:
1. User message exactly `STOP`
2. File `STOP_AUTORESEARCH` exists in repo root

## Reporting Policy
- Log every run to `results.tsv` (one row per model).
- Keep human-facing progress updates sparse:
  - report every 10 experiments, or
  - report immediately on crash, timeout, or repeated failure pattern
- Keep updates concise: experiment id, key val deltas, keep/discard decision.

## Runtime Limits
- Target duration per experiment: about 5 minutes total (plus small startup/eval overhead).
- Hard timeout: 10 minutes wall-clock.
- If a run exceeds 10 minutes, kill it, log failure (`discard` or `crash`), revert, and continue.

## Crash Handling
If a run fails (OOM, bug, runtime error):

- If the issue is trivial and fixable (typo, missing import, obvious small bug), fix and re-run quickly.
- If the idea itself is fundamentally broken, log `crash`, revert, and move on.
- If crash persists after a few quick fixes, stop retrying that idea and move to the next.

## Decision Policy
- If target-model `RMSE` improves and behavior is stable, mark `keep` and advance code.
- If `RMSE` is equal or worse, mark `discard` and revert to the last kept state.
- If `RMSE` improves but test behavior clearly degrades, mark `discard` and revert.
- Prefer simpler changes when outcomes are close.

## Autonomous Loop
After setup, LOOP FOREVER until a stop signal is received:

1. Inspect current `train.py` and latest `results.tsv` entries.
2. Propose one concrete experimental idea (single axis at a time when possible).
3. Edit `train.py`.
4. Run experiment:
   - `python train.py --status <keep|discard|crash> --description "<what changed>"`
5. If run crashes or times out, apply crash policy and log accordingly.
6. Compare results vs latest kept baseline for each model and the primary target model.
7. Keep or discard:
   - keep: leave code as new baseline and set status `keep`
   - discard: revert code to last kept state and set status `discard`/`reverted`
8. Repeat immediately with the next idea.

If ideas run out, think harder:
- revisit prior near-misses
- combine promising partial wins
- try broader optimizer/schedule/model changes
- revisit assumptions from `prepare.py` feature structure (without editing it)
