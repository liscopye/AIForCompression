#!/usr/bin/env python3
"""Convert cropped daily ERA5 NetCDF pairs into normalized mmap training shards."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
import sys
import time
from pathlib import Path

import netCDF4
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.era5_netcdf_dataset import (
    N_CHANNELS,
    PRESSURE_LEVELS,
    PRESSURE_VARIABLES,
    SINGLE_VARIABLES,
    load_cra5_channel_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/workspace/Data/ERA5/hourly_center512_20240301_90d"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/Data/ERA5/hourly_center512_shards_20240301_90d"),
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=Path("/workspace/AIForCompression/models/CRA5/cra5/dataset"),
    )
    parser.add_argument("--expected-days", type=int, default=90)
    parser.add_argument("--expected-times", type=int, default=24)
    parser.add_argument("--expected-height", type=int, default=513)
    parser.add_argument("--expected-width", type=int, default=513)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    return parser.parse_args()


def output_path_for(pressure_path: Path, output_dir: Path) -> Path:
    name = pressure_path.name
    for suffix in ("_pressure.nc", "_pressure.grib"):
        if name.endswith(suffix):
            name = name.removesuffix(suffix) + ".npy"
            break
    else:
        raise ValueError(f"Unsupported pressure source: {pressure_path}")
    return output_dir / name


def convert_pair(
    pressure_path: Path,
    single_path: Path,
    output_path: Path,
    means: np.ndarray,
    stds: np.ndarray,
    expected_shape: tuple[int, int, int] | None = None,
) -> None:
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.unlink(missing_ok=True)
    with ExitStack() as stack:
        single_ds = stack.enter_context(netCDF4.Dataset(single_path, "r"))
        if pressure_path.suffix == ".grib":
            import xarray as xr

            pressure_ds = stack.enter_context(
                xr.open_dataset(
                    pressure_path,
                    engine="cfgrib",
                    backend_kwargs={"indexpath": ""},
                    decode_timedelta=False,
                )
            )
            time_dimension = "time"
            level_dimension = "isobaricInhPa"
            times = int(pressure_ds.sizes[time_dimension])
            height = int(pressure_ds.sizes["latitude"])
            width = int(pressure_ds.sizes["longitude"])
            available_levels = np.asarray(pressure_ds[level_dimension].values)

            def read_pressure(variable: str, level_index: int) -> np.ndarray:
                return np.asarray(
                    pressure_ds[variable]
                    .isel({level_dimension: level_index})
                    .transpose(time_dimension, "latitude", "longitude")
                    .values,
                    dtype=np.float32,
                )

        elif pressure_path.suffix == ".nc":
            pressure_ds = stack.enter_context(netCDF4.Dataset(pressure_path, "r"))
            times = len(pressure_ds.dimensions["valid_time"])
            height = len(pressure_ds.dimensions["latitude"])
            width = len(pressure_ds.dimensions["longitude"])
            available_levels = np.asarray(
                pressure_ds.variables["pressure_level"][:]
            )

            def read_pressure(variable: str, level_index: int) -> np.ndarray:
                return np.asarray(
                    pressure_ds.variables[variable][:, level_index, :, :],
                    dtype=np.float32,
                )

        else:
            raise ValueError(f"Unsupported pressure source: {pressure_path}")

        if len(single_ds.dimensions["valid_time"]) != times:
            raise ValueError(f"Mismatched time dimensions for {pressure_path}")
        actual_shape = (times, height, width)
        if expected_shape is not None and actual_shape != expected_shape:
            source_kind = "GRIB" if pressure_path.suffix == ".grib" else "NetCDF"
            raise ValueError(
                f"Expected {source_kind} shape {expected_shape}, got {actual_shape} "
                f"for {pressure_path}"
            )
        if (
            len(single_ds.dimensions["latitude"]) != height
            or len(single_ds.dimensions["longitude"]) != width
        ):
            raise ValueError(f"Mismatched spatial dimensions for {pressure_path}")

        level_indices = [
            int(np.flatnonzero(np.isclose(available_levels, level))[0])
            for level in PRESSURE_LEVELS
        ]
        output = np.lib.format.open_memmap(
            tmp_path,
            mode="w+",
            dtype=np.float32,
            shape=(N_CHANNELS, times, height, width),
        )
        channel = 0
        for variable in PRESSURE_VARIABLES:
            for level_index in level_indices:
                values = read_pressure(variable, level_index)
                normalized = (values - means[channel]) / stds[channel]
                if not np.isfinite(normalized).all():
                    raise ValueError(
                        f"Non-finite values in pressure channel {channel} "
                        f"({variable}, level index {level_index})"
                    )
                output[channel] = normalized
                channel += 1
        for variable in SINGLE_VARIABLES:
            values = np.asarray(single_ds.variables[variable][:], dtype=np.float32)
            if variable == "tp":
                values *= 1000.0
            normalized = (values - means[channel]) / stds[channel]
            if not np.isfinite(normalized).all():
                raise ValueError(
                    f"Non-finite values in single-level channel {channel} ({variable})"
                )
            output[channel] = normalized
            channel += 1
        if channel != N_CHANNELS:
            raise RuntimeError(f"Wrote {channel} channels instead of {N_CHANNELS}")
        output.flush()
        del output

    os.replace(tmp_path, output_path)
    metadata = {
        "pressure_source": str(pressure_path.resolve()),
        "single_source": str(single_path.resolve()),
        "shape": [N_CHANNELS, times, height, width],
        "dtype": "float32",
        "normalization": "CRA5 fixed per-channel z-score; tp converted m->mm first",
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def available_pairs(input_dir: Path) -> list[tuple[Path, Path]]:
    pressure_by_stem: dict[str, Path] = {}
    for pressure_path in sorted(input_dir.glob("*_pressure.nc")):
        pressure_by_stem[pressure_path.name.removesuffix("_pressure.nc")] = pressure_path
    for pressure_path in sorted(input_dir.glob("*_pressure.grib")):
        stem = pressure_path.name.removesuffix("_pressure.grib")
        pressure_by_stem.setdefault(stem, pressure_path)

    pairs = []
    for stem, pressure_path in sorted(pressure_by_stem.items()):
        single_path = pressure_path.with_name(f"{stem}_single.nc")
        if single_path.is_file():
            pairs.append((pressure_path, single_path))
    return pairs


def main() -> None:
    args = parse_args()
    if args.expected_days <= 0 or args.poll_seconds <= 0:
        raise ValueError("--expected-days and --poll-seconds must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    means, stds = load_cra5_channel_stats(args.stats_dir)

    while True:
        pairs = available_pairs(args.input_dir)
        for pressure_path, single_path in pairs:
            output_path = output_path_for(pressure_path, args.output_dir)
            if output_path.is_file():
                continue
            print(f"Converting {pressure_path.name} -> {output_path.name}", flush=True)
            started = time.time()
            convert_pair(
                pressure_path,
                single_path,
                output_path,
                means,
                stds,
                (args.expected_times, args.expected_height, args.expected_width),
            )
            print(
                f"Finished {output_path.name} in {time.time() - started:.1f}s "
                f"({output_path.stat().st_size / 1e9:.2f} GB)",
                flush=True,
            )

        completed = len(list(args.output_dir.glob("*_hourly.npy")))
        print(
            f"Shard progress: {completed}/{args.expected_days}; "
            f"downloaded pairs: {len(pairs)}",
            flush=True,
        )
        if completed >= args.expected_days or not args.watch:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
