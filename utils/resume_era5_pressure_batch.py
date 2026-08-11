#!/usr/bin/env python3
"""Resume a completed CDS pressure result and publish validated daily GRIBs."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

import cdsapi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.download_era5 import (
    PRESSURE_LEVELS,
    PRESSURE_VARIABLES,
    batch_output_paths,
    download_result_resumable,
    output_lock,
    pressure_output_path,
    resolve_times,
    split_grib_by_day,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--dates", nargs="+", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--max-retries", type=int, default=500)
    parser.add_argument("--retry-wait", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    days = [date.fromisoformat(value) for value in args.dates]
    times = resolve_times("00:00", ["hourly"])
    outputs = {
        day: pressure_output_path(
            batch_output_paths(args.data_dir, day, times)[0], "grib"
        )
        for day in days
    }
    first, last = days[0], days[-1]
    batch = args.data_dir / (
        f".pressure_batch_{first.isoformat()}_{last.isoformat()}_"
        f"{len(days)}d.grib"
    )
    partial = batch.with_suffix(batch.suffix + ".tmp")
    expected_per_day = len(PRESSURE_VARIABLES) * len(PRESSURE_LEVELS) * len(times)

    result = cdsapi.Client().client.get_results(args.request_id)
    expected_bytes = int(result.content_length)
    print(
        f"Resuming request {args.request_id}: "
        f"{partial.stat().st_size if partial.exists() else 0}/{expected_bytes}",
        flush=True,
    )
    with output_lock(batch):
        download_result_resumable(
            result,
            partial,
            max_retries=args.max_retries,
            retry_wait=args.retry_wait,
        )
        partial.replace(batch)
        counts = split_grib_by_day(
            batch,
            outputs,
            expected_messages_per_day=expected_per_day,
        )
        batch.unlink()

    for day in days:
        print(
            f"Published {outputs[day]} "
            f"({outputs[day].stat().st_size} bytes, {counts[day]} messages)",
            flush=True,
        )


if __name__ == "__main__":
    main()
