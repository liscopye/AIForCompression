#!/usr/bin/env python3
"""Backfill scientific Objective-v1 LPIPS with one deterministic decode per point."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["e3sm_npz", "era5_npy", "hurricane", "nyx", "turb_rot_npz", "tomo", "lysozyme", "s2c"]
MODELS = ["DCAE", "HPCM", "cuSZ-Hi", "nvJPEG2000", "DCMVC-I", "DCVC-RT-I"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path("unified_results/objective_all_to_all_v1"))
    parser.add_argument("--output", type=Path, default=Path("unified_results/objective_lpips_backfill_v2"))
    parser.add_argument("--gpus", nargs="+", default=[str(index) for index in range(8)])
    return parser.parse_args()


def controls(schedule: dict, dataset: str, curve: str) -> list[str]:
    return [str(value) for value in schedule[dataset][curve]["controls"]]


def command(args: argparse.Namespace, schedule: dict, dataset: str, gpu: str, phase: str) -> list[str]:
    rows = json.loads((args.target / dataset / "summary.json").read_text(encoding="utf-8"))
    j2k = sorted({float(row["control"]) for row in rows if row.get("model_name") == "nvJPEG2000"})
    selected_models = MODELS if phase == "base" else [phase]
    caesar_curve = phase if phase in {"CAESAR-V", "CAESAR-D"} else "CAESAR-V"
    return [
        sys.executable, str(PROJECT_ROOT / "scripts/run_objective_benchmark.py"),
        "--dataset", dataset, "--gpu", gpu,
        "--input-root", str(args.target), "--output-root", str(args.output),
        "--models", *selected_models,
        "--caesar-eb", *controls(schedule, dataset, caesar_curve),
        "--cusz-eb", *controls(schedule, dataset, "cuSZ-Hi"),
        "--j2k-psnr", *map(str, j2k), "--warmups", "0", "--repeats", "1",
    ]


def main() -> None:
    args = parse_args()
    if len(args.gpus) < len(DATASETS):
        raise ValueError(f"Need {len(DATASETS)} GPU identifiers")
    schedule = json.loads((args.target / "eb_schedule.json").read_text(encoding="utf-8"))
    failed = []
    for phase in ("base", "CAESAR-V", "CAESAR-D"):
        processes = []
        for dataset, gpu in zip(DATASETS, args.gpus):
            log_path = Path("/tmp") / f"aifc_lpips_{dataset}.log"
            log = log_path.open("a", encoding="utf-8")
            process = subprocess.Popen(
                command(args, schedule, dataset, gpu, phase), cwd=PROJECT_ROOT,
                stdout=log, stderr=subprocess.STDOUT,
            )
            processes.append((dataset, process, log, log_path))
            print(f"started {dataset} {phase} on GPU {gpu}: {log_path}")
        for dataset, process, log, log_path in processes:
            returncode = process.wait()
            log.close()
            print(f"finished {dataset} {phase}: returncode={returncode}")
            if returncode:
                failed.append((dataset, log_path))
        if failed:
            break
    if failed:
        raise SystemExit("failed: " + ", ".join(f"{dataset} ({path})" for dataset, path in failed))


if __name__ == "__main__":
    main()
