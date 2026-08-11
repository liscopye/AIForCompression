#!/usr/bin/env python3
"""Audit chronological ERA5 hourly shards before CAESAR training."""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=Path("/workspace/Data/ERA5/hourly_center512_shards_20240301_90d"),
    )
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2024, 3, 1))
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--channels", type=int, default=268)
    parser.add_argument("--times", type=int, default=24)
    parser.add_argument("--height", type=int, default=513)
    parser.add_argument("--width", type=int, default=513)
    parser.add_argument("--train-timesteps", type=int, default=1920)
    parser.add_argument("--val-timesteps", type=int, default=240)
    parser.add_argument(
        "--objective-start-date",
        type=date.fromisoformat,
        default=date(2024, 6, 1),
    )
    parser.add_argument("--objective-timesteps", type=int, default=16)
    parser.add_argument("--objective-array", type=Path, default=None)
    parser.add_argument("--objective-raw-dir", type=Path, default=None)
    parser.add_argument(
        "--mean-std-dir",
        type=Path,
        default=Path("/workspace/AIForCompression/models/CRA5/cra5/dataset"),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def expected_dates(start_date: date, days: int) -> list[date]:
    if days <= 0:
        raise ValueError("days must be positive")
    return [start_date + timedelta(days=offset) for offset in range(days)]


def sample_indices(size: int) -> list[int]:
    return sorted({0, size // 2, size - 1})


def compare_objective_probes(
    objective: np.ndarray,
    channel_indices: list[int],
    time_indices: list[int],
    y_indices: list[int],
    x_indices: list[int],
    expected_value: Callable[[int, int, int, int], float],
    atol: float = 1e-6,
) -> dict:
    """Compare sparse objective-array probes against independently read sources."""
    max_abs = 0.0
    comparisons = 0
    for time_index in time_indices:
        for channel_index in channel_indices:
            for y in y_indices:
                for x in x_indices:
                    actual = float(objective[channel_index, time_index, y, x])
                    expected = float(expected_value(channel_index, time_index, y, x))
                    if not np.isfinite(actual) or not np.isfinite(expected):
                        raise ValueError(
                            "Non-finite objective provenance value at "
                            f"C={channel_index}, T={time_index}, Y={y}, X={x}"
                        )
                    difference = abs(actual - expected)
                    max_abs = max(max_abs, difference)
                    comparisons += 1
                    if difference > atol:
                        raise ValueError(
                            "Objective provenance mismatch at "
                            f"C={channel_index}, T={time_index}, Y={y}, X={x}: "
                            f"array={actual}, raw={expected}, abs={difference}"
                        )
    return {
        "comparisons": comparisons,
        "max_abs_difference": max_abs,
        "absolute_tolerance": atol,
    }


def audit_objective_provenance(
    objective_array: Path,
    raw_dir: Path,
    objective_start_date: date,
    objective_timesteps: int,
    mean_std_dir: Path,
) -> dict:
    """Prove that the objective array is the normalized June daily test corpus."""
    try:
        from utils.prepare_era5_finetune_data import (
            PRESSURE_LEVELS,
            VNAMES,
            build_channel_stats,
            load_mean_std,
        )
    except ModuleNotFoundError:
        # Support both ``python -m utils.audit_...`` and direct script execution.
        from prepare_era5_finetune_data import (
            PRESSURE_LEVELS,
            VNAMES,
            build_channel_stats,
            load_mean_std,
        )

    expected_shape = (268, objective_timesteps, 721, 1440)
    objective = np.load(objective_array, mmap_mode="r")
    if objective.shape != expected_shape:
        raise ValueError(
            f"Expected objective shape {expected_shape}, got {objective.shape}: "
            f"{objective_array}"
        )
    if objective.dtype != np.float32:
        raise ValueError(
            f"Expected objective float32, got {objective.dtype}: {objective_array}"
        )

    dates = expected_dates(objective_start_date, objective_timesteps)
    pairs = []
    for day in dates:
        timestamp = f"{day.isoformat()}T00:00:00"
        pressure = raw_dir / f"{timestamp}_pressure.nc"
        single = raw_dir / f"{timestamp}_single.nc"
        if not pressure.is_file():
            raise FileNotFoundError(pressure)
        if not single.is_file():
            raise FileNotFoundError(single)
        pairs.append((pressure, single))

    mean_std, mean_std_single = load_mean_std(str(mean_std_dir))
    means, stds = build_channel_stats(mean_std, mean_std_single)
    channel_indices = sorted(
        {
            0,
            36,
            37,
            73,
            74,
            110,
            111,
            147,
            148,
            184,
            185,
            221,
            222,
            258,
            *range(259, 268),
        }
    )
    time_indices = sample_indices(objective_timesteps)
    y_indices = sample_indices(expected_shape[2])
    x_indices = sample_indices(expected_shape[3])

    import xarray as xr

    datasets: dict[int, tuple[object, object]] = {}

    def expected_value(channel: int, time_index: int, y: int, x: int) -> float:
        if time_index not in datasets:
            pressure_path, single_path = pairs[time_index]
            pressure_ds = xr.open_dataset(pressure_path, engine="netcdf4")
            single_ds = xr.open_dataset(single_path, engine="netcdf4")
            expected_time = np.datetime64(
                f"{dates[time_index].isoformat()}T00:00:00"
            )
            for path, dataset in (
                (pressure_path, pressure_ds),
                (single_path, single_ds),
            ):
                time_name = "valid_time" if "valid_time" in dataset.coords else "time"
                actual_time = np.asarray(dataset[time_name].values).reshape(-1)
                if actual_time.size != 1 or actual_time[0] != expected_time:
                    pressure_ds.close()
                    single_ds.close()
                    raise ValueError(
                        f"Unexpected timestamp in {path}: {actual_time.tolist()}"
                    )
            datasets[time_index] = (pressure_ds, single_ds)

        pressure_ds, single_ds = datasets[time_index]
        if channel < 259:
            variable_index, level_index = divmod(channel, len(PRESSURE_LEVELS))
            variable = VNAMES["pressure"][variable_index]
            field = pressure_ds[variable].isel(
                valid_time=0,
                pressure_level=level_index,
                latitude=y,
                longitude=x,
            )
        else:
            variable = VNAMES["single"][channel - 259]
            field = single_ds[variable].isel(
                valid_time=0,
                latitude=y,
                longitude=x,
            )
        raw_value = float(field.values)
        if variable == "tp":
            raw_value *= 1000.0
        normalized = np.float32(
            (np.float32(raw_value) - means[channel])
            / max(stds[channel], np.float32(1e-8))
        )
        return float(normalized)

    try:
        comparison = compare_objective_probes(
            objective,
            channel_indices,
            time_indices,
            y_indices,
            x_indices,
            expected_value,
        )
    finally:
        for pressure_ds, single_ds in datasets.values():
            pressure_ds.close()
            single_ds.close()

    return {
        "status": "passed",
        "array": str(objective_array.resolve()),
        "raw_dir": str(raw_dir.resolve()),
        "shape": list(expected_shape),
        "dtype": str(objective.dtype),
        "first_timestamp": f"{dates[0].isoformat()}T00:00:00",
        "last_timestamp": f"{dates[-1].isoformat()}T00:00:00",
        "cadence": "daily",
        "normalization": "CRA5 fixed per-channel z-score",
        "sampled_indices": {
            "channels": channel_indices,
            "times": time_indices,
            "y": y_indices,
            "x": x_indices,
        },
        **comparison,
    }


def audit_shards(
    shard_dir: Path,
    dates: list[date],
    expected_shape: tuple[int, int, int, int],
    train_timesteps: int,
    val_timesteps: int,
    objective_start_date: date,
) -> dict:
    channels, times, height, width = expected_shape
    total_timesteps = len(dates) * times
    if train_timesteps + val_timesteps != total_timesteps:
        raise ValueError(
            f"train+val={train_timesteps + val_timesteps} does not equal "
            f"available timesteps={total_timesteps}"
        )
    if dates[-1] >= objective_start_date:
        raise ValueError(
            f"Training/validation ends on {dates[-1]}, not before objective test "
            f"start {objective_start_date}"
        )

    expected_stems = {f"{day.isoformat()}_hourly" for day in dates}
    actual_stems = {path.stem for path in shard_dir.glob("*_hourly.npy")}
    missing = sorted(expected_stems - actual_stems)
    extra = sorted(actual_stems - expected_stems)
    if missing or extra:
        raise ValueError(f"Shard date mismatch: missing={missing}, extra={extra}")

    channel_samples = sample_indices(channels)
    time_samples = sample_indices(times)
    y_samples = sample_indices(height)
    x_samples = sample_indices(width)
    records = []
    total_bytes = 0
    for day in dates:
        stem = f"{day.isoformat()}_hourly"
        shard_path = shard_dir / f"{stem}.npy"
        metadata_path = shard_dir / f"{stem}.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)

        shard = np.load(shard_path, mmap_mode="r")
        if shard.shape != expected_shape:
            raise ValueError(
                f"Expected {expected_shape}, got {shard.shape}: {shard_path}"
            )
        if shard.dtype != np.float32:
            raise ValueError(f"Expected float32, got {shard.dtype}: {shard_path}")
        probe = shard[np.ix_(channel_samples, time_samples, y_samples, x_samples)]
        if not np.isfinite(probe).all():
            raise ValueError(f"Non-finite audit sample in {shard_path}")
        del shard

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if tuple(metadata.get("shape", ())) != expected_shape:
            raise ValueError(f"Metadata shape mismatch: {metadata_path}")
        if metadata.get("dtype") != "float32":
            raise ValueError(f"Metadata dtype mismatch: {metadata_path}")
        for source_field in ("pressure_source", "single_source"):
            source = Path(metadata.get(source_field, ""))
            if not source.is_file():
                raise FileNotFoundError(
                    f"Missing {source_field} referenced by {metadata_path}: {source}"
                )

        size = shard_path.stat().st_size
        total_bytes += size
        records.append(
            {
                "date": day.isoformat(),
                "shard": str(shard_path.resolve()),
                "bytes": size,
                "pressure_source": metadata["pressure_source"],
                "single_source": metadata["single_source"],
            }
        )

    train_last_index = train_timesteps - 1
    val_first_index = train_timesteps
    val_last_index = train_timesteps + val_timesteps - 1

    def timestamp_record(index: int) -> dict:
        day_index, hour = divmod(index, times)
        return {"index": index, "date": dates[day_index].isoformat(), "hour": hour}

    return {
        "status": "passed",
        "shard_dir": str(shard_dir.resolve()),
        "shape_per_day": list(expected_shape),
        "days": len(dates),
        "total_timesteps": total_timesteps,
        "total_bytes": total_bytes,
        "train": {
            "timesteps": train_timesteps,
            "first": timestamp_record(0),
            "last": timestamp_record(train_last_index),
        },
        "validation": {
            "timesteps": val_timesteps,
            "first": timestamp_record(val_first_index),
            "last": timestamp_record(val_last_index),
        },
        "objective_start_date": objective_start_date.isoformat(),
        "sampled_indices": {
            "channels": channel_samples,
            "times": time_samples,
            "y": y_samples,
            "x": x_samples,
        },
        "records": records,
    }


def main() -> None:
    args = parse_args()
    if (args.objective_array is None) != (args.objective_raw_dir is None):
        raise ValueError(
            "--objective-array and --objective-raw-dir must be provided together"
        )
    result = audit_shards(
        args.shard_dir,
        expected_dates(args.start_date, args.days),
        (args.channels, args.times, args.height, args.width),
        args.train_timesteps,
        args.val_timesteps,
        args.objective_start_date,
    )
    if args.objective_array is not None:
        result["objective"] = audit_objective_provenance(
            args.objective_array,
            args.objective_raw_dir,
            args.objective_start_date,
            args.objective_timesteps,
            args.mean_std_dir,
        )
    output = args.output or args.shard_dir / "audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(
        f"ERA5 shard audit passed: {result['days']} days, "
        f"{result['total_timesteps']} timesteps, {result['total_bytes'] / 1e9:.2f} GB"
    )


if __name__ == "__main__":
    main()
