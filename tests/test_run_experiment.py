import argparse
from pathlib import Path

import pytest

from trading_forecast.run_experiment import ensure_output_parent, validate_args


def test_ensure_output_parent_creates_directory(tmp_path):
    out_file = tmp_path / "nested" / "artifacts" / "dataset.csv"
    result = ensure_output_parent(str(out_file))
    assert result == out_file
    assert out_file.parent.exists()


def test_validate_args_rejects_unsupported_granularity():
    args = argparse.Namespace(granularity=120, horizon=5, lookback=60)
    with pytest.raises(ValueError):
        validate_args(args)
