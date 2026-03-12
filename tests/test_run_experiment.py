import argparse

import pytest

from trading_forecast.run_experiment import ensure_dir, ensure_output_parent, resolve_device, validate_args


def test_ensure_output_parent_creates_directory(tmp_path):
    out_file = tmp_path / "nested" / "artifacts" / "dataset.csv"
    result = ensure_output_parent(str(out_file))
    assert result == out_file
    assert out_file.parent.exists()


def test_ensure_dir_creates_directory(tmp_path):
    out_dir = tmp_path / "plots"
    result = ensure_dir(str(out_dir))
    assert result == out_dir
    assert out_dir.exists()


def test_validate_args_rejects_unsupported_granularity():
    args = argparse.Namespace(granularity=120, horizon=5, lookback=60)
    with pytest.raises(ValueError):
        validate_args(args)


def test_resolve_device_cpu():
    assert resolve_device("cpu") == "cpu"

