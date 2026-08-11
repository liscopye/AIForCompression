#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--days", type=int, default=16)
    parser.add_argument("--hour", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=240)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.days <= 0 or not 0 <= args.hour < 24:
        raise ValueError("--days must be positive and --hour must be in [0, 23]")
    shard_dir = Path(args.shard_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    start = date.fromisoformat(args.start_date)
    dates = [start + timedelta(days=index) for index in range(args.days)]

    first_path = shard_dir / f"{dates[0].isoformat()}_hourly.npy"
    first = np.load(first_path, mmap_mode="r")
    if first.ndim != 4 or first.shape[:2] != (268, 24):
        raise ValueError(f"Expected [268,24,H,W], got {first.shape}: {first_path}")
    crop_h = min(args.crop_size, first.shape[-2])
    crop_w = min(args.crop_size, first.shape[-1])
    y0 = (first.shape[-2] - crop_h) // 2
    x0 = (first.shape[-1] - crop_w) // 2

    array = np.lib.format.open_memmap(
        output,
        mode="w+",
        dtype=np.float32,
        shape=(268, args.days, crop_h, crop_w),
    )
    for time_index, day in enumerate(dates):
        path = shard_dir / f"{day.isoformat()}_hourly.npy"
        shard = np.load(path, mmap_mode="r")
        if shard.shape != first.shape:
            raise ValueError(f"Shape mismatch {shard.shape} != {first.shape}: {path}")
        array[:, time_index] = shard[
            :, args.hour, y0 : y0 + crop_h, x0 : x0 + crop_w
        ]
        array.flush()
    del array

    metadata = {
        "shape": [268, args.days, crop_h, crop_w],
        "dtype": "float32",
        "dates": [day.isoformat() for day in dates],
        "hour": args.hour,
        "cadence": "daily",
        "source_shard_dir": str(shard_dir.resolve()),
        "crop": {"y0": y0, "x0": x0, "height": crop_h, "width": crop_w},
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
