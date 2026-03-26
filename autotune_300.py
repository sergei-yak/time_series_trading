from __future__ import annotations

import json
import random
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
TRAIN_PY = ROOT / "train.py"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
TRIAL_LOG_PATH = ROOT / "artifacts" / "autotune_300_trials.jsonl"
BEST_PATH = ROOT / "artifacts" / "autotune_300_best.json"
TOTAL_TRIALS = 300
MODELS = ("bilstm", "transformer", "lstnet", "rescnn_gru", "nbeats", "nhits")


@dataclass
class TrialResult:
    trial_id: int
    model: str
    rmse: float
    params: dict[str, Any]
    run_id: str | None
    raw_output_tail: str


def _choice(rng: random.Random, values: list[Any]) -> Any:
    return values[rng.randrange(len(values))]


def _uniform_lr(rng: random.Random) -> float:
    # Log-uniform between 3e-4 and 1.2e-3.
    lo = -3.52287874528
    hi = -2.92081875395
    return round(10 ** rng.uniform(lo, hi), 8)


def sample_params(model: str, rng: random.Random) -> dict[str, Any]:
    params: dict[str, Any] = {
        "batch_size": _choice(rng, [64, 96, 128]),
        "epochs": _choice(rng, [2, 3, 4]),
        "learning_rate": _uniform_lr(rng),
        "weight_decay": _choice(rng, [0.0, 1e-5, 3e-5, 5e-5, 1e-4]),
    }
    if model == "bilstm":
        params["bilstm_hidden_dim"] = _choice(rng, [96, 128, 160, 192])
        params["bilstm_num_layers"] = _choice(rng, [1, 2, 3])
    elif model == "transformer":
        d_model = _choice(rng, [64, 96, 128, 160])
        nhead_candidates = [h for h in (4, 8) if d_model % h == 0]
        params["transformer_d_model"] = d_model
        params["transformer_nhead"] = _choice(rng, nhead_candidates)
        params["transformer_num_layers"] = _choice(rng, [1, 2, 3])
        params["transformer_dim_feedforward"] = _choice(rng, [128, 192, 256, 384, 512])
        params["transformer_dropout"] = _choice(rng, [0.0, 0.05, 0.1, 0.15])
    elif model == "lstnet":
        params["lstnet_cnn_channels"] = _choice(rng, [64, 80, 96, 128])
        params["lstnet_gru_hidden"] = _choice(rng, [64, 80, 96, 128])
        params["lstnet_kernel_size"] = _choice(rng, [3, 5, 7])
    elif model == "rescnn_gru":
        params["rescnn_channels"] = _choice(rng, [64, 72, 96, 128])
        params["rescnn_gru_hidden"] = _choice(rng, [64, 72, 96, 128])
    elif model == "nbeats":
        params["nbeats_hidden_dim"] = _choice(rng, [192, 256, 320, 384])
        params["nbeats_num_blocks"] = _choice(rng, [3, 4, 5])
        params["nbeats_num_layers"] = _choice(rng, [3, 4, 5])
        params["nbeats_trend_degree"] = _choice(rng, [3, 4, 5])
        params["nbeats_seasonality_harmonics"] = _choice(rng, [4, 6, 8])
        params["nbeats_dropout"] = _choice(rng, [0.0, 0.05, 0.1])
        params["nbeats_share_weights"] = _choice(rng, [False, True])
    elif model == "nhits":
        params["nhits_hidden_dim"] = _choice(rng, [256, 320, 384, 448])
        params["nhits_num_blocks"] = _choice(rng, [3, 4, 5])
        params["nhits_num_layers"] = _choice(rng, [2, 3, 4])
        params["nhits_dropout"] = _choice(rng, [0.0, 0.03, 0.05, 0.1])
        if params["nhits_num_blocks"] == 3:
            params["nhits_pool_sizes"] = _choice(rng, ["8,4,2", "6,3,1", "10,5,2"])
            params["nhits_downsample_frequencies"] = _choice(rng, ["8,4,2", "6,3,1", "10,5,2"])
        elif params["nhits_num_blocks"] == 4:
            params["nhits_pool_sizes"] = _choice(rng, ["8,4,2,1", "10,5,2,1", "12,6,3,1"])
            params["nhits_downsample_frequencies"] = _choice(rng, ["8,4,2,1", "10,5,2,1", "12,6,3,1"])
        else:
            params["nhits_pool_sizes"] = _choice(rng, ["12,8,4,2,1", "10,6,3,2,1", "8,6,4,2,1"])
            params["nhits_downsample_frequencies"] = _choice(rng, ["12,8,4,2,1", "10,6,3,2,1", "8,6,4,2,1"])
    else:
        raise ValueError(f"Unsupported model: {model}")
    return params


def build_command(model: str, trial_id: int, params: dict[str, Any]) -> list[str]:
    cmd = [
        str(PYTHON),
        "train.py",
        "--models",
        model,
        "--status",
        "auto",
        "--description",
        f"autotune300 trial={trial_id} model={model}",
        "--batch-size",
        str(params["batch_size"]),
        "--epochs",
        str(params["epochs"]),
        "--learning-rate",
        str(params["learning_rate"]),
        "--weight-decay",
        str(params["weight_decay"]),
    ]
    for key, value in params.items():
        if key in {"batch_size", "epochs", "learning_rate", "weight_decay"}:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        cmd.extend([flag, str(value)])
    return cmd


def parse_run(stdout: str, model: str) -> tuple[float, str | None]:
    rmse_match = re.search(rf"{re.escape(model)}:\s+test_loss=[0-9.]+,\s+mae=[0-9.]+,\s+rmse=([0-9.]+)", stdout)
    if not rmse_match:
        raise RuntimeError(f"Could not parse RMSE from output for model={model}")
    rmse = float(rmse_match.group(1))
    run_id_match = re.search(r"under commit=(\S+)", stdout)
    run_id = run_id_match.group(1) if run_id_match else None
    return rmse, run_id


def run_trial(trial_id: int, model: str, params: dict[str, Any]) -> TrialResult:
    cmd = build_command(model=model, trial_id=trial_id, params=params)
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    combined = (completed.stdout or "") + "\n" + (completed.stderr or "")
    if completed.returncode != 0:
        tail = "\n".join(combined.strip().splitlines()[-30:])
        raise RuntimeError(f"Trial {trial_id} failed (model={model}).\n{tail}")
    rmse, run_id = parse_run(combined, model=model)
    tail = "\n".join(combined.strip().splitlines()[-12:])
    return TrialResult(
        trial_id=trial_id,
        model=model,
        rmse=rmse,
        params=params,
        run_id=run_id,
        raw_output_tail=tail,
    )


def patch_train_defaults(best_by_model: dict[str, TrialResult]) -> None:
    text = TRAIN_PY.read_text(encoding="utf-8")
    nhits_best = best_by_model["nhits"]

    replacements = {
        r"(batch_size:\s*int\s*=\s*)[0-9]+": rf"\g<1>{nhits_best.params['batch_size']}",
        r"(epochs:\s*int\s*=\s*)[0-9]+": rf"\g<1>{nhits_best.params['epochs']}",
        r"(learning_rate:\s*float\s*=\s*)[0-9.eE+-]+": rf"\g<1>{nhits_best.params['learning_rate']}",
        r"(weight_decay:\s*float\s*=\s*)[0-9.eE+-]+": rf"\g<1>{nhits_best.params['weight_decay']}",
    }

    best_params = {m: best_by_model[m].params for m in MODELS}
    parser_updates = {
        "--bilstm-hidden-dim": best_params["bilstm"]["bilstm_hidden_dim"],
        "--bilstm-num-layers": best_params["bilstm"]["bilstm_num_layers"],
        "--transformer-d-model": best_params["transformer"]["transformer_d_model"],
        "--transformer-nhead": best_params["transformer"]["transformer_nhead"],
        "--transformer-num-layers": best_params["transformer"]["transformer_num_layers"],
        "--transformer-dim-feedforward": best_params["transformer"]["transformer_dim_feedforward"],
        "--transformer-dropout": best_params["transformer"]["transformer_dropout"],
        "--lstnet-cnn-channels": best_params["lstnet"]["lstnet_cnn_channels"],
        "--lstnet-gru-hidden": best_params["lstnet"]["lstnet_gru_hidden"],
        "--lstnet-kernel-size": best_params["lstnet"]["lstnet_kernel_size"],
        "--rescnn-channels": best_params["rescnn_gru"]["rescnn_channels"],
        "--rescnn-gru-hidden": best_params["rescnn_gru"]["rescnn_gru_hidden"],
        "--nbeats-hidden-dim": best_params["nbeats"]["nbeats_hidden_dim"],
        "--nbeats-num-blocks": best_params["nbeats"]["nbeats_num_blocks"],
        "--nbeats-num-layers": best_params["nbeats"]["nbeats_num_layers"],
        "--nbeats-trend-degree": best_params["nbeats"]["nbeats_trend_degree"],
        "--nbeats-seasonality-harmonics": best_params["nbeats"]["nbeats_seasonality_harmonics"],
        "--nbeats-dropout": best_params["nbeats"]["nbeats_dropout"],
        "--nhits-hidden-dim": best_params["nhits"]["nhits_hidden_dim"],
        "--nhits-num-blocks": best_params["nhits"]["nhits_num_blocks"],
        "--nhits-num-layers": best_params["nhits"]["nhits_num_layers"],
        "--nhits-dropout": best_params["nhits"]["nhits_dropout"],
    }

    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text, count=1)

    for flag, value in parser_updates.items():
        text = re.sub(
            rf"(parser\.add_argument\(\"{re.escape(flag)}\"[^\n]*default=)[^,\n]+",
            rf"\g<1>{value}",
            text,
            count=1,
        )

    TRAIN_PY.write_text(text, encoding="utf-8")


def main() -> int:
    if not PYTHON.exists():
        print(f"Missing venv python at {PYTHON}", file=sys.stderr)
        return 1
    TRIAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260325)

    best_by_model: dict[str, TrialResult] = {}
    history_by_model: dict[str, list[TrialResult]] = defaultdict(list)
    completed_trials = 0
    if TRIAL_LOG_PATH.exists():
        with TRIAL_LOG_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                result = TrialResult(
                    trial_id=int(payload["trial_id"]),
                    model=str(payload["model"]),
                    rmse=float(payload["rmse"]),
                    params=dict(payload["params"]),
                    run_id=payload.get("run_id"),
                    raw_output_tail="",
                )
                history_by_model[result.model].append(result)
                current_best = best_by_model.get(result.model)
                if current_best is None or result.rmse < current_best.rmse:
                    best_by_model[result.model] = result
                completed_trials = max(completed_trials, result.trial_id)

    for _ in range(completed_trials):
        # Advance RNG to preserve deterministic sampling when resuming.
        model = MODELS[_ % len(MODELS)]
        _ = sample_params(model=model, rng=rng)

    for trial_id in range(completed_trials + 1, TOTAL_TRIALS + 1):
        model = MODELS[(trial_id - 1) % len(MODELS)]
        params = sample_params(model=model, rng=rng)
        result = run_trial(trial_id=trial_id, model=model, params=params)
        history_by_model[model].append(result)
        current_best = best_by_model.get(model)
        if current_best is None or result.rmse < current_best.rmse:
            best_by_model[model] = result

        with TRIAL_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "trial_id": result.trial_id,
                        "model": result.model,
                        "rmse": result.rmse,
                        "params": result.params,
                        "run_id": result.run_id,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )

        print(
            f"trial={trial_id:03d}/{TOTAL_TRIALS} model={model} rmse={result.rmse:.6f} "
            f"best={best_by_model[model].rmse:.6f}",
            flush=True,
        )

    if set(best_by_model.keys()) != set(MODELS):
        missing = sorted(set(MODELS).difference(best_by_model.keys()))
        raise RuntimeError(f"Missing best results for models: {missing}")

    best_payload = {
        model: {
            "rmse": best_by_model[model].rmse,
            "run_id": best_by_model[model].run_id,
            "trial_id": best_by_model[model].trial_id,
            "params": best_by_model[model].params,
        }
        for model in MODELS
    }
    BEST_PATH.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    patch_train_defaults(best_by_model=best_by_model)
    print(f"Saved best trial details to {BEST_PATH}")
    print("Updated train.py defaults with best hyperparameters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
