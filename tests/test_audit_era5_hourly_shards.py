import json
from datetime import date

import numpy as np
import pytest

from utils.audit_era5_hourly_shards import (
    audit_shards,
    compare_objective_probes,
    expected_dates,
)


def _write_day(root, day, shape):
    stem = f"{day.isoformat()}_hourly"
    shard_path = root / f"{stem}.npy"
    np.save(shard_path, np.zeros(shape, dtype=np.float32))
    pressure = root / f"{stem}_pressure.nc"
    single = root / f"{stem}_single.nc"
    pressure.touch()
    single.touch()
    (root / f"{stem}.json").write_text(
        json.dumps(
            {
                "shape": list(shape),
                "dtype": "float32",
                "pressure_source": str(pressure),
                "single_source": str(single),
            }
        ),
        encoding="utf-8",
    )


def test_audit_reports_chronological_split(tmp_path):
    dates = expected_dates(date(2024, 3, 1), 2)
    shape = (3, 2, 4, 4)
    for day in dates:
        _write_day(tmp_path, day, shape)

    result = audit_shards(
        tmp_path,
        dates,
        shape,
        train_timesteps=2,
        val_timesteps=2,
        objective_start_date=date(2024, 6, 1),
    )

    assert result["status"] == "passed"
    assert result["train"]["last"] == {
        "index": 1,
        "date": "2024-03-01",
        "hour": 1,
    }
    assert result["validation"]["first"] == {
        "index": 2,
        "date": "2024-03-02",
        "hour": 0,
    }


def test_audit_rejects_missing_day(tmp_path):
    dates = expected_dates(date(2024, 3, 1), 2)
    _write_day(tmp_path, dates[0], (3, 2, 4, 4))

    with pytest.raises(ValueError, match="missing"):
        audit_shards(
            tmp_path,
            dates,
            (3, 2, 4, 4),
            train_timesteps=2,
            val_timesteps=2,
            objective_start_date=date(2024, 6, 1),
        )


def test_compare_objective_probes_reports_exact_match():
    objective = np.arange(3 * 2 * 4 * 4, dtype=np.float32).reshape(3, 2, 4, 4)

    result = compare_objective_probes(
        objective,
        channel_indices=[0, 2],
        time_indices=[0, 1],
        y_indices=[0, 3],
        x_indices=[0, 3],
        expected_value=lambda c, t, y, x: objective[c, t, y, x],
    )

    assert result["comparisons"] == 16
    assert result["max_abs_difference"] == 0.0


def test_compare_objective_probes_rejects_mismatch():
    objective = np.zeros((1, 1, 1, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="provenance mismatch"):
        compare_objective_probes(
            objective,
            channel_indices=[0],
            time_indices=[0],
            y_indices=[0],
            x_indices=[0],
            expected_value=lambda *_: 1.0,
        )
