#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "models" / "CAESAR"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.finetune_caesar_era5 import load_caesar_d_vae, load_caesar_v
from utils.era5_netcdf_dataset import (
    PRESSURE_LEVELS,
    PRESSURE_VARIABLES,
    SINGLE_VARIABLES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["V", "D"], required=True)
    parser.add_argument("--original-checkpoint", required=True)
    parser.add_argument("--tuned-checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--time-start", type=int, default=0)
    parser.add_argument("--time-count", type=int, default=16)
    parser.add_argument("--crop-size", type=int, default=240)
    return parser.parse_args()


def channel_name(index: int) -> str:
    pressure_count = len(PRESSURE_VARIABLES) * len(PRESSURE_LEVELS)
    if index < pressure_count:
        variable = PRESSURE_VARIABLES[index // len(PRESSURE_LEVELS)]
        level = PRESSURE_LEVELS[index % len(PRESSURE_LEVELS)]
        return f"{variable}_{level}hPa"
    return SINGLE_VARIABLES[index - pressure_count]


def center_crop_and_pad(raw: torch.Tensor, crop_size: int) -> tuple[torch.Tensor, int]:
    crop_h = min(crop_size, raw.shape[-2])
    crop_w = min(crop_size, raw.shape[-1])
    y0 = (raw.shape[-2] - crop_h) // 2
    x0 = (raw.shape[-1] - crop_w) // 2
    raw = raw[..., y0 : y0 + crop_h, x0 : x0 + crop_w]
    target = 256 if max(crop_h, crop_w) <= 256 else math.ceil(
        max(crop_h, crop_w) / 32
    ) * 32
    pad_h = target - crop_h
    pad_w = target - crop_w
    if pad_h or pad_w:
        raw = F.pad(
            raw,
            (
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
                0,
                0,
            ),
            mode="reflect",
        )
    return raw, pad_h // 2


def evaluate(
    model_type: str,
    checkpoint: str,
    source: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, float]]:
    device = torch.device(args.device)
    model = (
        load_caesar_v(checkpoint, device)
        if model_type == "V"
        else load_caesar_d_vae(checkpoint, device)
    )
    model.eval()
    n_frame = 8 if model_type == "V" else 16
    time_starts = range(0, args.time_count, n_frame) if model_type == "V" else (0,)
    stats = [
        {
            "squared_error": 0.0,
            "normalized_squared_error": 0.0,
            "count": 0,
            "bits": 0.0,
            "minimum": float("inf"),
            "maximum": float("-inf"),
            "scale_sum": 0.0,
            "scale_count": 0,
        }
        for _ in range(source.shape[0])
    ]
    with torch.no_grad():
        for local_t0 in time_starts:
            t0 = args.time_start + local_t0
            t1 = t0 + n_frame
            for c0 in range(0, source.shape[0], args.batch_size):
                c1 = min(c0 + args.batch_size, source.shape[0])
                raw = torch.from_numpy(
                    np.array(
                        source[c0:c1, t0:t1],
                        dtype=np.float32,
                        copy=True,
                    )
                ).unsqueeze(1)
                raw, crop_pad = center_crop_and_pad(raw, args.crop_size)
                offset = raw.mean(dim=(2, 3, 4), keepdim=True)
                scale = (
                    raw.amax(dim=(2, 3, 4), keepdim=True)
                    - raw.amin(dim=(2, 3, 4), keepdim=True)
                ).clamp_min(torch.finfo(raw.dtype).eps)
                normalized = ((raw - offset) / scale).to(device)
                target = normalized
                if model_type == "D":
                    indices = torch.arange(0, n_frame, 3, device=device)
                    target = normalized.index_select(2, indices)
                result = model(target)
                reconstructed = result["output"].cpu() * scale + offset
                normalized_reconstructed = result["output"].cpu()
                raw_target = raw if model_type == "V" else raw[:, :, ::3]
                if crop_pad:
                    reconstructed = reconstructed[
                        ..., crop_pad:-crop_pad, crop_pad:-crop_pad
                    ]
                    raw_target = raw_target[
                        ..., crop_pad:-crop_pad, crop_pad:-crop_pad
                    ]
                    normalized_reconstructed = normalized_reconstructed[
                        ..., crop_pad:-crop_pad, crop_pad:-crop_pad
                    ]
                    normalized_target = target.cpu()[
                        ..., crop_pad:-crop_pad, crop_pad:-crop_pad
                    ]
                else:
                    normalized_target = target.cpu()
                source_error = (reconstructed - raw_target).square()
                normalized_error = (
                    normalized_reconstructed - normalized_target
                ).square()
                bpp = result["bpp"].detach().float().cpu().reshape(c1 - c0, -1)
                for local, channel in enumerate(range(c0, c1)):
                    item = stats[channel]
                    item["squared_error"] += float(source_error[local].sum())
                    item["normalized_squared_error"] += float(
                        normalized_error[local].sum()
                    )
                    item["count"] += source_error[local].numel()
                    item["bits"] += float(bpp[local].mean())
                    item["minimum"] = min(
                        item["minimum"], float(raw_target[local].min())
                    )
                    item["maximum"] = max(
                        item["maximum"], float(raw_target[local].max())
                    )
                    item["scale_sum"] += float(scale[local].mean())
                    item["scale_count"] += 1
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return stats


def finalize(stats: list[dict[str, float]]) -> list[dict[str, float | str | int]]:
    rows = []
    for index, item in enumerate(stats):
        mse = item["squared_error"] / item["count"]
        normalized_mse = item["normalized_squared_error"] / item["count"]
        data_range = item["maximum"] - item["minimum"]
        rows.append(
            {
                "channel": index,
                "name": channel_name(index),
                "mse": mse,
                "normalized_mse": normalized_mse,
                "psnr": (
                    10 * math.log10(data_range**2 / mse)
                    if mse > 0 and data_range > 0
                    else float("inf")
                ),
                "theoretical_bpp": item["bits"] / item["scale_count"],
                "data_range": data_range,
                "mean_patch_scale": item["scale_sum"] / item["scale_count"],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    source = np.load(args.data, mmap_mode="r")
    if source.ndim != 4 or source.shape[0] != 268:
        raise ValueError(f"Expected [268,T,H,W], got {source.shape}")
    if args.time_start + args.time_count > source.shape[1]:
        raise ValueError("Requested time range exceeds input")

    original = finalize(
        evaluate(args.model_type, args.original_checkpoint, source, args)
    )
    tuned = finalize(
        evaluate(args.model_type, args.tuned_checkpoint, source, args)
    )
    channels = []
    for base, candidate in zip(original, tuned, strict=True):
        row = {
            **candidate,
            "original_mse": base["mse"],
            "original_psnr": base["psnr"],
            "original_bpp": base["theoretical_bpp"],
            "mse_change_percent": 100 * (candidate["mse"] / base["mse"] - 1),
            "psnr_delta_db": candidate["psnr"] - base["psnr"],
            "bpp_change_percent": 100
            * (candidate["theoretical_bpp"] / base["theoretical_bpp"] - 1),
        }
        channels.append(row)

    base_total_mse = float(np.mean([row["mse"] for row in original]))
    tuned_total_mse = float(np.mean([row["mse"] for row in tuned]))
    scales = np.asarray([row["mean_patch_scale"] for row in channels])
    changes = np.asarray([row["mse_change_percent"] for row in channels])
    result = {
        "model_type": args.model_type,
        "data": str(Path(args.data).resolve()),
        "time_start": args.time_start,
        "time_count": args.time_count,
        "crop_size": args.crop_size,
        "original_checkpoint": str(Path(args.original_checkpoint).resolve()),
        "tuned_checkpoint": str(Path(args.tuned_checkpoint).resolve()),
        "summary": {
            "mean_channel_mse_change_percent": 100
            * (tuned_total_mse / base_total_mse - 1),
            "mean_channel_psnr_delta_db": float(
                np.mean([row["psnr_delta_db"] for row in channels])
            ),
            "improved_channel_count": sum(
                row["mse_change_percent"] < 0 for row in channels
            ),
            "regressed_channel_count": sum(
                row["mse_change_percent"] > 0 for row in channels
            ),
            "mse_change_vs_scale_correlation": float(
                np.corrcoef(changes, scales)[0, 1]
            ),
        },
        "largest_regressions": sorted(
            channels, key=lambda row: row["mse_change_percent"], reverse=True
        )[:20],
        "largest_improvements": sorted(
            channels, key=lambda row: row["mse_change_percent"]
        )[:20],
        "channels": channels,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
