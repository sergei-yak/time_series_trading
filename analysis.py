import argparse
import re
from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: matplotlib. Install with: "
        "pip install matplotlib"
    ) from exc


def extract_experiment_number(description: str) -> int | None:
    if not isinstance(description, str):
        return None
    match = re.search(r"exp(\d+)", description, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    if "baseline" in description.lower():
        return 0
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize RMSE vs experiment number from results.tsv"
    )
    parser.add_argument("--input", default="results.tsv", help="Path to results TSV file")
    parser.add_argument(
        "--output",
        default="artifacts/rmse_over_experiments.png",
        help="Path to save output chart image",
    )
    parser.add_argument(
        "--dpi", type=int, default=150, help="DPI for saved plot image"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, sep="\t")
    required_cols = {"RMSE", "status", "model", "description"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in TSV: {sorted(missing)}")

    df["RMSE"] = pd.to_numeric(df["RMSE"], errors="coerce")
    df = df.dropna(subset=["RMSE", "status", "model"])

    # Prefer explicit expN tags when present; otherwise use run/commit order so all runs appear.
    df["experiment_number"] = df["description"].apply(extract_experiment_number)
    explicit_ratio = float(df["experiment_number"].notna().mean()) if len(df) else 0.0
    if explicit_ratio < 0.5 and "commit" in df.columns:
        commit_order = {}
        next_idx = 0
        for commit in df["commit"].astype(str):
            if commit not in commit_order:
                commit_order[commit] = next_idx
                next_idx += 1
        df["experiment_number"] = df["commit"].astype(str).map(commit_order).astype(int)
    else:
        df = df.dropna(subset=["experiment_number"]).copy()
        df["experiment_number"] = df["experiment_number"].astype(int)

    models = sorted(df["model"].unique())
    cmap = plt.get_cmap("tab10")

    plt.figure(figsize=(14, 7))

    for idx, model in enumerate(models):
        color = cmap(idx % 10)
        mdf = df[df["model"] == model].sort_values("experiment_number")

        keep_df = mdf[mdf["status"].str.lower() == "keep"]
        discard_df = mdf[mdf["status"].str.lower() == "discard"]

        if not keep_df.empty:
            plt.plot(
                keep_df["experiment_number"],
                keep_df["RMSE"],
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=4,
                label=f"{model} (keep)",
            )

        if not discard_df.empty:
            plt.scatter(
                discard_df["experiment_number"],
                discard_df["RMSE"],
                color=color,
                s=18,
                alpha=0.65,
                marker=".",
                label=f"{model} (discard)",
            )

    plt.title("RMSE Over Experiments by Model")
    plt.xlabel("Experiment Number")
    plt.ylabel("RMSE")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=args.dpi)
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
