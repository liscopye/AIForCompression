#!/usr/bin/env python3
"""Arrange scattered objective run directories by dataset without changing results."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DATASETS = {
    "e3sm_npz", "era5_npy", "hurricane", "kodak", "lysozyme", "nyx",
    "s2c", "tomo", "turb_rot_npz", "uvg_twilight_1080p",
}
EXCLUDED = {"objective_all_to_all_v1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("unified_results"))
    parser.add_argument("--destination", type=Path, default=Path("/workspace/tmp/aifc_objective_runs"))
    parser.add_argument("--execute", action="store_true", help="Move files; otherwise print the plan.")
    return parser.parse_args()


def move(source: Path, destination: Path, execute: bool) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    print(f"{source} -> {destination}")
    if execute:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))


def main() -> None:
    args = parse_args()
    roots = sorted(
        path for path in args.root.glob("objective_*")
        if path.is_dir() and path.name not in EXCLUDED and path != args.destination
    )
    for run_root in roots:
        dataset_dirs = sorted(path for path in run_root.iterdir() if path.is_dir() and path.name in DATASETS)
        inferred_dataset = "era5_npy" if "era5" in run_root.name else None
        for dataset_dir in dataset_dirs:
            move(dataset_dir, args.destination / dataset_dir.name / run_root.name, args.execute)
        leftovers = [path for path in run_root.iterdir() if path.name not in DATASETS]
        if leftovers:
            bucket = inferred_dataset or "_shared"
            move(run_root, args.destination / bucket / run_root.name / "_run_files", args.execute)
        elif args.execute:
            run_root.rmdir()


if __name__ == "__main__":
    main()
