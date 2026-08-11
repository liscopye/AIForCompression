#!/usr/bin/env python3
"""Build a compact held-out ERA5 probe for final CAESAR checkpoint screening."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.era5_netcdf_dataset import (
    PRESSURE_LEVELS,
    PRESSURE_VARIABLES,
    SINGLE_VARIABLES,
    discover_npy_shard_frames,
)


def representative_channels() -> list[int]:
    level_indices = [
        PRESSURE_LEVELS.index(level)
        for level in (1000.0, 500.0, 100.0)
    ]
    channels = [
        variable_index * len(PRESSURE_LEVELS) + level_index
        for variable_index in range(len(PRESSURE_VARIABLES))
        for level_index in level_indices
    ]
    single_start = len(PRESSURE_VARIABLES) * len(PRESSURE_LEVELS)
    channels.extend(single_start + index for index in range(len(SINGLE_VARIABLES)))
    return channels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=Path("/workspace/Data/ERA5/hourly_center512_shards_20240301_90d"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/Data/ERA5/hourly_center512_validation_probe.npy"),
    )
    parser.add_argument("--train-timesteps", type=int, default=1920)
    parser.add_argument("--probe-timesteps", type=int, default=64)
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Source-frame spacing inside the probe; use 24 for same-hour daily ERA5.",
    )
    parser.add_argument("--crop-size", type=int, default=240)
    return parser.parse_args()


def build_probe(
    shard_dir: Path,
    output_path: Path,
    train_timesteps: int,
    probe_timesteps: int,
    frame_step: int,
    crop_size: int,
) -> dict:
    frames = discover_npy_shard_frames(shard_dir)
    if probe_timesteps <= 0 or frame_step <= 0:
        raise ValueError("probe_timesteps and frame_step must be positive")
    required = train_timesteps + (probe_timesteps - 1) * frame_step + 1
    if len(frames) < required:
        raise ValueError(f"Need {required} frames, found {len(frames)} in {shard_dir}")
    channels = representative_channels()
    probe_frames = frames[train_timesteps:required:frame_step]
    first = np.load(probe_frames[0].shard_path, mmap_mode="r")
    height, width = map(int, first.shape[-2:])
    del first
    if crop_size > height or crop_size > width:
        raise ValueError(f"crop_size={crop_size} exceeds shard field {height}x{width}")
    y0 = (height - crop_size) // 2
    x0 = (width - crop_size) // 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.unlink(missing_ok=True)
    output = np.lib.format.open_memmap(
        tmp_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(channels), probe_timesteps, crop_size, crop_size),
    )
    cache: OrderedDict[str, np.memmap] = OrderedDict()
    for time_index, frame in enumerate(probe_frames):
        shard = cache.pop(frame.shard_path, None)
        if shard is None:
            shard = np.load(frame.shard_path, mmap_mode="r")
        cache[frame.shard_path] = shard
        while len(cache) > 4:
            cache.popitem(last=False)
        output[:, time_index] = shard[
            channels,
            frame.local_time_index,
            y0:y0 + crop_size,
            x0:x0 + crop_size,
        ]
    output.flush()
    del output
    os.replace(tmp_path, output_path)

    metadata = {
        "source_shard_dir": str(shard_dir.resolve()),
        "shape": [len(channels), probe_timesteps, crop_size, crop_size],
        "dtype": "float32",
        "source_time_range": [train_timesteps, required],
        "source_time_indices": list(range(train_timesteps, required, frame_step)),
        "frame_step": frame_step,
        "source_channel_indices": channels,
        "pressure_levels_selected": [1000.0, 500.0, 100.0],
        "single_variables": list(SINGLE_VARIABLES),
        "normalization": "CRA5 fixed per-channel z-score inherited from daily shards",
        "selection_role": "held-out checkpoint screening only; not objective test",
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    args = parse_args()
    metadata = build_probe(
        args.shard_dir,
        args.output,
        args.train_timesteps,
        args.probe_timesteps,
        args.frame_step,
        args.crop_size,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
