#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CAESAR_ROOT = ROOT / "models" / "CAESAR"
if str(CAESAR_ROOT) not in sys.path:
    sys.path.insert(0, str(CAESAR_ROOT))

from scripts.finetune_caesar_era5 import load_caesar_d_vae, load_caesar_v


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["V", "D"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data",
        default="/workspace/Data/ERA5/finetune_processed/era5_test.npy",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def tensor_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    delta = left.detach().float().cpu() - right.detach().float().cpu()
    return {
        "max_abs": float(delta.abs().max()),
        "mse": float(delta.square().mean()),
        "exact_fraction": float((delta == 0).float().mean()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    n_frame = 8 if args.model_type == "V" else 16
    source = np.load(args.data, mmap_mode="r")
    raw = torch.from_numpy(
        np.array(
            source[: args.batch_size, :n_frame],
            dtype=np.float32,
            copy=True,
        )
    ).unsqueeze(1)
    crop_h = min(240, raw.shape[-2])
    crop_w = min(240, raw.shape[-1])
    crop_y = (raw.shape[-2] - crop_h) // 2
    crop_x = (raw.shape[-1] - crop_w) // 2
    raw = raw[
        ...,
        crop_y : crop_y + crop_h,
        crop_x : crop_x + crop_w,
    ]
    pad_h = 256 - raw.shape[-2]
    pad_w = 256 - raw.shape[-1]
    if pad_h < 0 or pad_w < 0:
        raise ValueError(
            f"Expected input no larger than 256x256, got {raw.shape[-2:]}"
        )
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
    offset = raw.mean(dim=(2, 3, 4), keepdim=True)
    scale = (
        raw.amax(dim=(2, 3, 4), keepdim=True)
        - raw.amin(dim=(2, 3, 4), keepdim=True)
    )
    full_x = ((raw - offset) / scale).to(device)
    model_x = full_x
    if args.model_type == "D":
        indices = torch.arange(0, n_frame, 3, device=device)
        model_x = full_x.index_select(2, indices)
        model = load_caesar_d_vae(args.checkpoint, device)
    else:
        model = load_caesar_v(args.checkpoint, device)
    model.eval()

    with torch.no_grad():
        forward = model(model_x)
        compressed = model.compress(model_x, return_latent=True)
        if args.model_type == "D":
            direct_q = compressed["q_latent"].to(device)
            direct_q_flat = direct_q.reshape(-1, *direct_q.shape[2:])
            range_q = model.decompress(*compressed["compressed"], device=device)
            range_q_flat = range_q.permute(0, 2, 1, 3, 4).reshape(
                -1, *range_q.shape[1:2], *range_q.shape[-2:]
            )
            direct_output_flat = model.decode(direct_q_flat)
            range_output_flat = model.decode(range_q_flat)
            batch_size, time = model_x.shape[0], model_x.shape[2]
            direct_output = direct_output_flat.reshape(
                batch_size, time, *direct_output_flat.shape[1:]
            ).permute(0, 2, 1, 3, 4)
            range_output = range_output_flat.reshape(
                batch_size, time, *range_output_flat.shape[1:]
            ).permute(0, 2, 1, 3, 4)
            forward_q = forward["q_latent"]
            compressed_q = direct_q_flat
        else:
            range_output = model.decompress(
                *compressed["compressed"], device=device
            )
            direct_output = model.decode(
                compressed["q_latent"].to(device), model_x.shape[0]
            )
            forward_q = forward["q_latent"]
            compressed_q = compressed["q_latent"]

    source_scale = scale.to(device)
    normalized_forward_mse = float(
        (forward["output"] - model_x).square().mean()
    )
    normalized_range_mse = float((range_output - model_x).square().mean())
    source_forward_mse = float(
        ((forward["output"] - model_x) * source_scale).square().mean()
    )
    source_range_mse = float(
        ((range_output - model_x) * source_scale).square().mean()
    )

    result = {
        "model_type": args.model_type,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data": str(Path(args.data).resolve()),
        "input_shape": list(model_x.shape),
        "forward_vs_direct_decode": tensor_metrics(
            forward["output"], direct_output
        ),
        "forward_vs_range_decode": tensor_metrics(
            forward["output"], range_output
        ),
        "direct_vs_range_decode": tensor_metrics(direct_output, range_output),
        "forward_vs_compress_q_latent": tensor_metrics(
            forward_q, compressed_q
        ),
        "normalized_forward_mse": normalized_forward_mse,
        "normalized_range_mse": normalized_range_mse,
        "source_forward_mse": source_forward_mse,
        "source_range_mse": source_range_mse,
        "real_bits": float(compressed["bpf_real"].sum()),
        "theoretical_bits": float(compressed["bpf_entropy"].sum()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
