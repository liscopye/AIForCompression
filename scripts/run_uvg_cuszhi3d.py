#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compression_pipeline.adapters.uvg import UVGAdapter
from compression_pipeline.canonical import CanonicalSample
from compression_pipeline.metrics import make_lpips_fn
from scripts.run_external_scientific_codecs import (
    DEFAULT_CUSZHI,
    run_cuszhi_stack_sample,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cuSZ-Hi on UVG/Twilight as one RGB-frame 3D stack.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_frames", type=int, default=30)
    parser.add_argument(
        "--frames_per_chunk",
        type=int,
        default=4,
        help="Number of RGB frames per cuSZ 3D block. 4 keeps 4K Twilight below cuSZ internal size limits.",
    )
    parser.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--eb", type=float, nargs="+", default=[0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001])
    parser.add_argument("--cuszhi", default=str(DEFAULT_CUSZHI))
    parser.add_argument("--cuszhi_scheme", choices=["cr", "tp"], default="cr")
    parser.add_argument("--cuszhi_predictor", default="spline3")
    parser.add_argument("--cuszhi_min_abs_eb", type=float, default=3e-6)
    parser.add_argument("--cuszhi_eb_reference", choices=["range", "robust"], default="range")
    parser.add_argument("--cuszhi_robust_low", type=float, default=0.1)
    parser.add_argument("--cuszhi_robust_high", type=float, default=99.9)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--no_lpips", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary = []
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    done = {(r.get("model_id"), r.get("sample_id"), r.get("eb")) for r in summary if "error" not in r}

    adapter = UVGAdapter(args.data_root)
    sequence, _timestamps = adapter.load_sequence(max_samples=args.max_frames, resolution=tuple(args.resolution) if args.resolution else None)
    channels, frames, height, width = sequence.shape
    samples = []
    chunk = max(1, int(args.frames_per_chunk))
    for start in range(0, frames, chunk):
        end = min(frames, start + chunk)
        # sequence is [RGB, T, H, W].  Pack time and RGB into the Z axis,
        # matching the Kodak full-stack convention [image_rgb_channel, H, W].
        arr = np.ascontiguousarray(
            np.moveaxis(sequence[:, start:end], 1, 0).reshape((end - start) * channels, height, width).astype(np.float32)
        )
        samples.append(CanonicalSample(
            dataset_id="uvg",
            sample_id=f"twilight_frames{start:03d}-{end - 1:03d}_rgbstack",
            kind="video",
            array=arr,
            layout="channel_height_width",
            metadata={
                "source_path": str(adapter._find_yuv()),
                "source_layout": "RGB,T,H,W",
                "cuszhi_pack_z": True,
                "z_axis": "frame_rgb_channel",
                "frame_range": [int(start), int(end)],
                "frame_count": int(end - start),
                "channels": int(arr.shape[0]),
                "height": int(height),
                "width": int(width),
                "dtype": "float32",
            },
        ))
    args.lpips_fn = None if args.no_lpips else make_lpips_fn("cuda")
    runner_args = SimpleNamespace(**vars(args))
    runner_args.cuszhi_sample_mode = "whole3d"
    runner_args.lysozyme_invalid_policy = "raw"
    runner_args.lysozyme_invalid_threshold = 4.294967e9

    for eb in args.eb:
        model_id = f"cuSZ-Hi-3D_eb{str(eb).replace('.', 'p')}"
        sample_id = f"twilight_frames000-{frames - 1:03d}_rgbstack_chunk{chunk}"
        key = (model_id, sample_id, eb)
        if key in done:
            print(f"[skip] {model_id} {sample_id}", flush=True)
            continue
        chunk_rows = []
        try:
            for sample in samples:
                print(f"[run] {model_id} {sample.sample_id} shape={sample.array.shape}", flush=True)
                chunk_rows.append(run_cuszhi_stack_sample(
                    sample,
                    runner_args,
                    float(eb),
                    output_dir,
                    sample.array,
                    Path(args.cuszhi),
                    requested_mode="whole3d",
                ))
            result = aggregate_chunks(chunk_rows, sample_id)
            result.update(
                {
                    "model_name": "cuSZ-Hi",
                    "model_id": model_id,
                    "label": "cuSZ-Hi-3D",
                    "metric": "error_bound",
                    "eb": float(eb),
                    "dataset_id": "uvg",
                    "sample_id": sample_id,
                    "sample_kind": "video",
                    "shape": [list(s.array.shape) for s in samples],
                    "aggregated_samples": len(samples),
                    "sample_ids": [s.sample_id for s in samples],
                    "frames_per_chunk": int(chunk),
                    "total_frames": int(frames),
                }
            )
            summary.append(result)
            done.add(key)
            print(
                f"[ok] {model_id} bpp={result.get('scientific_bpp_with_side_info', result.get('bpp')):.4g} "
                f"psnr={result.get('psnr'):.4g}",
                flush=True,
            )
        except Exception as exc:
            summary.append(
                {
                    "model_name": "cuSZ-Hi",
                    "model_id": model_id,
                    "label": "cuSZ-Hi-3D",
                    "metric": "error_bound",
                    "eb": float(eb),
                    "dataset_id": sample.dataset_id,
                    "sample_id": sample_id,
                    "error": str(exc),
                }
            )
            print(f"[error] {model_id}: {exc}", flush=True)
        finally:
            write_summary(summary_path, summary)
    print(f"Results: {summary_path}", flush=True)


def aggregate_chunks(rows: list[dict], sample_id: str) -> dict:
    if len(rows) == 1:
        row = dict(rows[0])
        row["sample_id"] = sample_id
        row["aggregated_samples"] = 1
        return row
    first = dict(rows[0])
    voxel_counts = np.array([voxel_count(r) for r in rows], dtype=np.float64)
    voxel_counts = np.maximum(voxel_counts, 1)
    total_voxels = float(np.sum(voxel_counts))
    mse = float(np.sum([float(r["mse"]) * v for r, v in zip(rows, voxel_counts)]) / total_voxels)
    data_min = min(float(r.get("original_min", 0.0)) for r in rows)
    data_max = max(float(r.get("original_max", 1.0)) for r in rows)
    data_range = max(data_max - data_min, 1e-8)
    bitstream_bytes = int(sum(int(r.get("bitstream_bytes", 0)) for r in rows))
    side_info_bytes = int(sum(int(r.get("side_info_bytes", 0)) for r in rows))
    total_bytes = bitstream_bytes + side_info_bytes
    original_bytes = int(sum(int(r.get("original_bytes", 0)) for r in rows))
    encode_time = float(sum(float(r.get("encode_time_avg", 0.0)) for r in rows))
    decode_time = float(sum(float(r.get("decode_time_avg", 0.0)) for r in rows))
    first.update({
        "sample_id": sample_id,
        "aggregated_samples": len(rows),
        "shape": [r.get("shape") for r in rows],
        "voxel_count": int(total_voxels),
        "original_min": data_min,
        "original_max": data_max,
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "psnr": float("inf") if mse < 1e-30 else float(10 * math.log10((data_range * data_range) / mse)),
        "average_variable_psnr": mean_present(rows, "average_variable_psnr"),
        "average_frame_psnr": mean_present(rows, "average_frame_psnr"),
        "lpips": mean_present(rows, "lpips"),
        "bitstream_bytes": bitstream_bytes,
        "side_info_bytes": side_info_bytes,
        "total_bytes_with_side_info": total_bytes,
        "original_bytes": original_bytes,
        "bpp": bitstream_bytes * 8.0 / total_voxels,
        "scientific_bpp": bitstream_bytes * 8.0 / total_voxels,
        "scientific_bpp_with_side_info": total_bytes * 8.0 / total_voxels,
        "compression_ratio": original_bytes / total_bytes if total_bytes > 0 else float("inf"),
        "encode_time_avg": encode_time,
        "decode_time_avg": decode_time,
        "encode_time_total": encode_time,
        "decode_time_total": decode_time,
        "encode_throughput": original_bytes / encode_time if encode_time > 0 else None,
        "decode_throughput": original_bytes / decode_time if decode_time > 0 else None,
        "encode_throughput_MBps": original_bytes / encode_time / 1e6 if encode_time > 0 else None,
        "decode_throughput_MBps": original_bytes / decode_time / 1e6 if decode_time > 0 else None,
        "memory_usage_MB": max((r.get("memory_usage_MB") or 0.0) for r in rows),
        "memory_reserved_MB": max((r.get("memory_reserved_MB") or 0.0) for r in rows),
    })
    return first


def mean_present(rows: list[dict], key: str):
    values = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float)) and math.isfinite(float(r[key]))]
    return float(np.mean(values)) if values else None


def voxel_count(row: dict) -> int:
    value = row.get("voxel_count")
    if isinstance(value, (int, float)) and value > 0:
        return int(value)
    shape = row.get("shape")
    if isinstance(shape, list) and shape and all(isinstance(x, (int, float)) for x in shape):
        return int(np.prod(shape))
    original_bytes = row.get("original_bytes")
    if isinstance(original_bytes, (int, float)) and original_bytes > 0:
        return int(original_bytes) // np.dtype(np.float32).itemsize
    return 1


if __name__ == "__main__":
    main()
