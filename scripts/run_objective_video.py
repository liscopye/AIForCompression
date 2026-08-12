#!/usr/bin/env python3
"""Run the objective-v1 UVG temporal track with repeated end-to-end timing."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "aifc-objective-v1"
DATASET_ID = "uvg_twilight_1080p"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--root", type=Path, default=Path("unified_results/objective_v1"))
    parser.add_argument("--models", nargs="+", choices=["dcvc", "dcmvc"], default=["dcvc", "dcmvc"])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def hardware_manifest(gpu: str) -> dict:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    code = (
        "import json,torch; p=torch.cuda.get_device_properties(0); "
        "print(json.dumps({'gpu_physical_index':%r,'gpu_name':p.name,"
        "'gpu_total_memory_bytes':p.total_memory,'torch_version':torch.__version__,"
        "'cuda_version':torch.version.cuda}))" % str(gpu)
    )
    return json.loads(subprocess.check_output([sys.executable, "-c", code], env=env, text=True))


def ensure_uvg_frames(root: Path) -> Path:
    """Export the frozen canonical UVG frames when they are not present yet."""
    frames_dir = root / DATASET_ID / "frames"
    pngs = sorted(frames_dir.glob("im*.png"))
    if len(pngs) == 30 and (frames_dir / "manifest.json").is_file():
        return frames_dir
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/export_objective_uvg_frames.py"),
        "--root",
        str(root),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    pngs = sorted(frames_dir.glob("im*.png"))
    if len(pngs) != 30 or not (frames_dir / "manifest.json").is_file():
        raise RuntimeError(f"UVG frame export incomplete: expected 30 PNGs in {frames_dir}, got {len(pngs)}")
    return frames_dir


def run_repetition(args: argparse.Namespace, model: str, index: int, measured_lpips: bool) -> list[dict]:
    run_dir = args.root / DATASET_ID / "video_runs" / model / f"run{index:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts/run_uvg_pframe_codecs.py"),
        "--model", model,
        "--data_dir", str(args.root / DATASET_ID / "frames"),
        "--output_dir", str(run_dir),
        "--max_frames", "30",
    ]
    if model == "dcmvc":
        command.extend(["--dcmvc_timing_mode", "memory_bitstream"])
    if not measured_lpips:
        command.append("--no_lpips")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONHASHSEED="20260722")
    with (run_dir / "run.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def formalize(
    source: dict,
    repetitions: list[dict],
    manifest: dict,
    normalization: dict,
    hardware: dict,
) -> dict:
    symbols = int(manifest["canonical_symbol_count"])
    valid_symbols = int(manifest["canonical_valid_symbol_count"])
    payload = int(source["bitstream_bytes"])
    mse = float(source["mse"])
    timing = [
        {
            "encode_seconds": float(row["encode_time_total"]),
            "decode_seconds": float(row["decode_time_total"]),
            "roundtrip_seconds": float(row["sample_wall_time_total"]),
            "scientific_bpp_with_side_info": payload * 8.0 / symbols,
            "psnr": -10.0 * math.log10(max(float(row["mse"]), 1e-30)),
        }
        for row in repetitions
    ]
    walls = [row["roundtrip_seconds"] for row in timing]
    encodes = [row["encode_seconds"] for row in timing]
    decodes = [row["decode_seconds"] for row in timing]
    return {
        **source,
        **manifest,
        "protocol_id": PROTOCOL_ID,
        "track_id": "video_temporal",
        "metric_protocol": PROTOCOL_ID,
        "timing_protocol": PROTOCOL_ID,
        "external_input_manifest": normalization,
        "hardware_manifest": hardware,
        "rate_denominator": "canonical_grid_symbols",
        "payload_bytes": payload,
        "side_info_bytes": 0,
        "total_bytes_with_side_info": payload,
        "scientific_bpp": payload * 8.0 / symbols,
        "scientific_bpp_with_side_info": payload * 8.0 / symbols,
        "canonical_symbol_count": symbols,
        "canonical_valid_symbol_count": valid_symbols,
        "normalized_mse": mse,
        "normalized_psnr": -10.0 * math.log10(max(mse, 1e-30)),
        "psnr": -10.0 * math.log10(max(mse, 1e-30)),
        "timing_repetitions": timing,
        "sample_wall_time_total": statistics.median(walls),
        "sample_wall_time_p10": float(np.percentile(walls, 10)),
        "sample_wall_time_p90": float(np.percentile(walls, 90)),
        "encode_time_avg": statistics.median(encodes),
        "decode_time_avg": statistics.median(decodes),
        "original_bytes": symbols * 4,
        "sample_wall_throughput_MBps": symbols * 4 / statistics.median(walls) / 1e6,
        "deterministic_seed": 20260722,
        "partition_policy": "one 30-frame GOP with amortized I frame",
    }


def main() -> None:
    args = parse_args()
    dataset_dir = args.root / DATASET_ID
    manifests = json.loads((dataset_dir / "samples.json").read_text(encoding="utf-8"))
    if len(manifests) != 1:
        raise ValueError("The UVG objective track requires exactly one canonical sequence")
    manifest = manifests[0]
    normalization = json.loads((dataset_dir / "normalization.json").read_text(encoding="utf-8"))
    ensure_uvg_frames(args.root)
    hardware = hardware_manifest(args.gpu)
    output = dataset_dir / "video_summary.json"
    rows = [] if args.force or not output.exists() else json.loads(output.read_text(encoding="utf-8"))

    for model in args.models:
        all_runs = []
        for index in range(args.warmups + args.repeats):
            print(f"[{model}] repetition {index + 1}/{args.warmups + args.repeats}", flush=True)
            all_runs.append(run_repetition(args, model, index, measured_lpips=index == args.warmups))
        measured = all_runs[args.warmups:]
        by_id = {row["model_id"]: row for row in measured[0]}
        for model_id, source in by_id.items():
            point_repetitions = [next(row for row in run if row["model_id"] == model_id) for run in measured]
            lpips_source = next(row for row in all_runs[args.warmups] if row["model_id"] == model_id)
            source = dict(source)
            source["lpips"] = lpips_source.get("lpips")
            rows.append(formalize(source, point_repetitions, manifest, normalization, hardware))
        output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
