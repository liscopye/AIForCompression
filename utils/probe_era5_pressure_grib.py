#!/usr/bin/env python3
"""Measure native-GRIB ERA5 retrieval and compare it with a NetCDF retrieval."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cdsapi
import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.download_era5 import (
    PRESSURE_LEVELS,
    PRESSURE_VARIABLES,
    output_lock,
    retrieve_with_retry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--reference-netcdf", type=Path, required=True)
    parser.add_argument("--area", nargs=4, type=float, default=[64, -64, -64, 64])
    parser.add_argument("--max-retries", type=int, default=20)
    parser.add_argument("--retry-wait", type=int, default=300)
    return parser.parse_args()


def retrieve_grib(args: argparse.Namespace) -> tuple[Path, float]:
    day = np.datetime64(args.date, "D")
    year, month, date = str(day).split("-")
    output = args.data_dir / f"{args.date}_hourly_pressure.grib"
    request = {
        "product_type": ["reanalysis"],
        "variable": PRESSURE_VARIABLES,
        "pressure_level": PRESSURE_LEVELS,
        "year": year,
        "month": month,
        "day": date,
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "data_format": "grib",
        "area": args.area,
    }
    started = time.monotonic()
    with output_lock(output):
        if not output.is_file() or output.stat().st_size <= 1_000_000:
            tmp = output.with_suffix(output.suffix + ".tmp")
            retrieve_with_retry(
                cdsapi.Client(),
                "reanalysis-era5-pressure-levels",
                request,
                tmp,
                args.max_retries,
                args.retry_wait,
            )
            tmp.replace(output)
    return output, time.monotonic() - started


def compare(grib_path: Path, netcdf_path: Path) -> dict:
    variable_names = ("z", "q", "u", "v", "t", "r", "w")
    comparisons = []
    with (
        xr.open_dataset(
            grib_path,
            engine="cfgrib",
            backend_kwargs={"indexpath": ""},
            decode_timedelta=False,
        ) as grib,
        xr.open_dataset(netcdf_path) as netcdf,
    ):
        grib_level = "isobaricInhPa"
        netcdf_level = "pressure_level"
        for variable in variable_names:
            if variable not in grib or variable not in netcdf:
                raise ValueError(
                    f"Missing {variable}: GRIB={list(grib.data_vars)}, "
                    f"NetCDF={list(netcdf.data_vars)}"
                )
            for time_index in (0, 12, 23):
                for level in (1000.0, 500.0, 1.0):
                    for y, x in ((0, 0), (256, 256), (512, 512)):
                        actual = float(
                            grib[variable]
                            .isel(time=time_index, latitude=y, longitude=x)
                            .sel({grib_level: level})
                        )
                        expected = float(
                            netcdf[variable]
                            .isel(valid_time=time_index, latitude=y, longitude=x)
                            .sel({netcdf_level: level})
                        )
                        comparisons.append(
                            {
                                "variable": variable,
                                "time_index": time_index,
                                "level": level,
                                "y": y,
                                "x": x,
                                "grib": actual,
                                "netcdf": expected,
                                "abs_error": abs(actual - expected),
                            }
                        )
    errors = np.asarray([item["abs_error"] for item in comparisons])
    references = np.asarray([abs(item["netcdf"]) for item in comparisons])
    return {
        "comparisons": len(comparisons),
        "max_abs_error": float(errors.max(initial=0.0)),
        "max_relative_error": float(
            np.max(errors / np.maximum(references, 1e-12), initial=0.0)
        ),
        "worst": comparisons[int(np.argmax(errors))] if comparisons else None,
    }


def main() -> None:
    args = parse_args()
    if not args.reference_netcdf.is_file():
        raise FileNotFoundError(args.reference_netcdf)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    grib_path, elapsed = retrieve_grib(args)
    report = {
        "date": args.date,
        "grib_path": str(grib_path.resolve()),
        "grib_bytes": grib_path.stat().st_size,
        "request_and_transfer_seconds": elapsed,
        "reference_netcdf": str(args.reference_netcdf.resolve()),
        "comparison": compare(grib_path, args.reference_netcdf),
    }
    output = args.data_dir / f"{args.date}_hourly_pressure_grib_probe.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
