#!/usr/bin/env python3
"""Locate CAESAR-D reconstruction loss between its VAE and temporal sampler.

This is a diagnostic, not a benchmark codec.  All non-oracle modes range-code
the same six keyframes.  They differ only in how the ten missing latent frames
are reconstructed.  The oracle mode encodes all frames only to measure the VAE
reconstruction ceiling; its reported keyframe bit count is not an oracle rate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression_pipeline.metrics import (
    base_metrics,
    calculate_axis0_average_psnr,
    calculate_psnr,
)
from compression_pipeline.objective_caesar import (
    load_caesar_compressor,
    prepare_caesar_corpus,
)
from compression_pipeline.objective_data import (
    derive_dataset_normalization,
    load_objective_samples,
)


MODES = (
    "official",
    "deterministic_posterior",
    "zero_start_deterministic",
    "linear_start_deterministic",
    "linear_refine",
    "diffusion_ensemble",
    "linear_latent",
    "linear_pixel",
    "nearest_latent",
    "all_frame_vae_oracle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--max-variables", type=int, default=32)
    parser.add_argument("--variable-start", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--interpo-rate", type=int, default=3)
    parser.add_argument(
        "--condition-indices",
        type=int,
        nargs="*",
        default=None,
        help="Explicit condition-frame indices; overrides --interpo-rate.",
    )
    parser.add_argument("--ensemble-size", type=int, default=4)
    parser.add_argument(
        "--refine-start-timestep",
        type=int,
        default=3,
        help="First reverse timestep for --mode linear_refine (inclusive).",
    )
    return parser.parse_args()


def normalize_condition_latent(
    latent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    minimum = latent.amin(dim=(1, 2, 3, 4), keepdim=True)
    maximum = latent.amax(dim=(1, 2, 3, 4), keepdim=True)
    scale = (maximum - minimum + 1e-8) / 2
    offset = minimum + scale
    return (latent - offset) / scale, offset, scale


def interpolate_latent(
    keyframes: torch.Tensor,
    cond_idx: torch.Tensor,
    total_frames: int,
    *,
    nearest: bool = False,
) -> torch.Tensor:
    """Fill a full latent sequence from regularly spaced keyframes."""
    full = keyframes.new_empty(
        keyframes.shape[0], keyframes.shape[1], total_frames,
        keyframes.shape[3], keyframes.shape[4]
    )
    full.index_copy_(2, cond_idx, keyframes)
    cond = cond_idx.tolist()
    for left_pos, right_pos in zip(range(len(cond) - 1), range(1, len(cond))):
        left_t, right_t = cond[left_pos], cond[right_pos]
        left = keyframes[:, :, left_pos]
        right = keyframes[:, :, right_pos]
        for frame in range(left_t + 1, right_t):
            alpha = (frame - left_t) / (right_t - left_t)
            if nearest:
                full[:, :, frame] = left if alpha < 0.5 else right
            else:
                full[:, :, frame] = left.lerp(right, alpha)
    if cond[0] > 0:
        full[:, :, : cond[0]] = keyframes[:, :, :1]
    if cond[-1] + 1 < total_frames:
        full[:, :, cond[-1] + 1 :] = keyframes[:, :, -1:]
    return full


def decode_latents(
    compressor: Any,
    full_latent: torch.Tensor,
    compressed: dict[str, Any],
) -> torch.Tensor:
    batch, _, frames = full_latent.shape[:3]
    flat = full_latent.permute(0, 2, 1, 3, 4).reshape(
        -1, full_latent.shape[1], full_latent.shape[3], full_latent.shape[4]
    )
    decoded = compressor.keyframe_model.decode(flat).detach()
    decoded = decoded.reshape(batch, frames, 1, *decoded.shape[-2:]).permute(0, 2, 1, 3, 4)
    scale = compressed["scale"].to(decoded.device)
    offset = compressed["offset"].to(decoded.device)
    return decoded * scale + offset


@torch.inference_mode()
def reconstruct_from_keyframes(
    compressor: Any,
    compressed_batches: list[dict[str, Any]],
    shape: tuple[int, ...],
    mode: str,
    refine_start_timestep: int,
    ensemble_size: int,
) -> torch.Tensor:
    reconstruction = torch.zeros(shape)
    cond_idx = compressor.cond_idx.to(compressor.device)
    pred_idx = compressor.pred_idx.to(compressor.device)

    for compressed in compressed_batches:
        keyframes = compressor.keyframe_model.decompress(
            *compressed["compressed"], device=compressor.device
        )
        batch, channels, _, height, width = keyframes.shape

        decoded: torch.Tensor | None = None
        if mode in {"linear_latent", "nearest_latent"}:
            full_latent = interpolate_latent(
                keyframes,
                cond_idx,
                compressor.n_frame,
                nearest=mode == "nearest_latent",
            )
        elif mode == "linear_pixel":
            decoded_keyframes = decode_latents(compressor, keyframes, compressed)
            decoded = interpolate_latent(
                decoded_keyframes, cond_idx, compressor.n_frame
            )
        else:
            condition = keyframes.new_zeros(
                batch, channels, compressor.n_frame, height, width
            )
            condition.index_copy_(2, cond_idx, keyframes)
            condition, offset, scale = normalize_condition_latent(condition)
            compressor.diffusion_model.zero_noise = mode in {
                "deterministic_posterior",
                "zero_start_deterministic",
                "linear_start_deterministic",
            }
            start_img = None
            if mode == "zero_start_deterministic":
                start_img = keyframes.new_zeros(batch, channels, int(pred_idx.sum()), height, width)
            elif mode == "linear_start_deterministic":
                linear = interpolate_latent(keyframes, cond_idx, compressor.n_frame)
                linear = (linear - offset) / scale
                start_img = linear[:, :, pred_idx]
            if mode == "diffusion_ensemble":
                if ensemble_size <= 0:
                    raise ValueError("--ensemble-size must be positive")
                decoded_samples = []
                for _ in range(ensemble_size):
                    sample_condition = condition.clone()
                    predicted = compressor.diffusion_model.sample(
                        sample_condition,
                        compressor.interpo_rate,
                        cond_idx=cond_idx,
                        batch_size=batch,
                    )
                    sample_condition[:, :, pred_idx] = predicted
                    sample_latent = sample_condition * scale + offset
                    decoded_samples.append(
                        decode_latents(compressor, sample_latent, compressed)
                    )
                decoded = torch.stack(decoded_samples).mean(dim=0)
                full_latent = condition
            elif mode == "linear_refine":
                if not 0 <= refine_start_timestep < compressor.diffusion_model.num_timesteps:
                    raise ValueError(
                        "--refine-start-timestep must be in "
                        f"[0, {compressor.diffusion_model.num_timesteps - 1}]"
                    )
                linear = interpolate_latent(keyframes, cond_idx, compressor.n_frame)
                linear = ((linear - offset) / scale)[:, :, pred_idx]
                timestep = torch.full(
                    (batch,), refine_start_timestep, device=linear.device, dtype=torch.long
                )
                predicted = compressor.diffusion_model.q_sample(linear, timestep)
                compressor.diffusion_model.cond_idx = cond_idx
                compressor.diffusion_model.noise_mask = pred_idx
                for step in reversed(range(refine_start_timestep + 1)):
                    predicted = compressor.diffusion_model.p_sample(
                        condition,
                        predicted,
                        torch.full(
                            (batch,), step, device=linear.device, dtype=torch.long
                        ),
                        clip_denoised=True,
                    )
            else:
                predicted = compressor.diffusion_model.sample(
                    condition,
                    compressor.interpo_rate,
                    cond_idx=cond_idx,
                    batch_size=batch,
                    start_img=start_img,
                )
            condition[:, :, pred_idx] = predicted
            full_latent = condition * scale + offset

        if decoded is None:
            decoded = decode_latents(compressor, full_latent, compressed)
        decoded = decoded.cpu()
        idx0, idx1, start_t, end_t = compressed["index"]
        for item in range(batch):
            reconstruction[
                idx0[item], idx1[item], start_t[item] : end_t[item]
            ] = decoded[item]
    return reconstruction


@torch.inference_mode()
def reconstruct_oracle(compressor: Any, corpus: Any) -> torch.Tensor:
    reconstruction = torch.zeros(tuple(corpus.dataset.data_input.shape))
    for batch_data in corpus.loader:
        model_input = batch_data["input"].to(compressor.device)
        full_latent = compressor.keyframe_model.inference_qlatent(model_input)
        packed = {
            "scale": batch_data["scale"],
            "offset": batch_data["offset"],
        }
        decoded = decode_latents(compressor, full_latent, packed).cpu()
        idx0, idx1, start_t, end_t = batch_data["index"]
        for item in range(decoded.shape[0]):
            reconstruction[
                idx0[item], idx1[item], start_t[item] : end_t[item]
            ] = decoded[item]
    return reconstruction


def subset_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    indices: list[int],
) -> dict[str, float] | None:
    if not indices:
        return None
    selected_original = original[:, :, indices]
    selected_reconstruction = reconstructed[:, :, indices]
    psnr, mse = calculate_psnr(selected_original, selected_reconstruction)
    average_variable_psnr = calculate_axis0_average_psnr(
        selected_original, selected_reconstruction
    )
    return {
        "psnr": psnr,
        "mse": mse,
        "average_variable_psnr": float(average_variable_psnr),
    }


def main() -> None:
    args = parse_args()
    if args.max_variables <= 0:
        raise ValueError("--max-variables must be positive")
    if args.variable_start < 0:
        raise ValueError("--variable-start cannot be negative")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    samples = load_objective_samples("era5_npy")
    normalization = derive_dataset_normalization("era5_npy", samples)
    variable_end = min(
        args.variable_start + args.max_variables,
        samples[0].raw.shape[0],
    )
    if args.variable_start >= variable_end:
        raise ValueError(
            f"--variable-start {args.variable_start} is outside the available "
            f"{samples[0].raw.shape[0]} variables"
        )
    normalized = normalization.normalize(samples[0].raw)[
        args.variable_start : variable_end
    ]
    mask = samples[0].mask
    if mask is not None:
        mask = mask[args.variable_start : variable_end]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    corpus = prepare_caesar_corpus(
        normalized,
        mask,
        "caesar_d",
        ROOT / "models/CAESAR",
        args.output.parent,
        f"era5-d-temporal-{args.mode}",
        batch_size=args.batch_size,
        norm_type="mean_range",
    )
    try:
        compressor = load_caesar_compressor(
            "caesar_d",
            ROOT / "models/CAESAR",
            args.checkpoint,
            args.device,
        )
        if args.condition_indices is None:
            if args.interpo_rate <= 0:
                raise ValueError("--interpo-rate must be positive")
            condition_indices = list(range(0, compressor.n_frame, args.interpo_rate))
        else:
            condition_indices = sorted(set(args.condition_indices))
            if not condition_indices:
                raise ValueError("--condition-indices cannot be empty")
            if condition_indices[0] < 0 or condition_indices[-1] >= compressor.n_frame:
                raise ValueError(
                    f"--condition-indices must be in [0, {compressor.n_frame - 1}]"
                )
        compressor.interpo_rate = args.interpo_rate
        compressor.cond_idx = torch.tensor(condition_indices, dtype=torch.long)
        compressor.pred_idx = ~torch.isin(
            torch.arange(compressor.n_frame), compressor.cond_idx
        )
        if not bool(compressor.pred_idx.any()) and args.mode != "all_frame_vae_oracle":
            raise ValueError("Diffusion diagnostics require at least one predicted frame")
        compressor.diffusion_model.num_frames = int(compressor.pred_idx.sum())
        compressor.transform_shape = corpus.dataset.deblocking_hw
        start = time.perf_counter()
        compressed_batches, latent_bytes_tensor = compressor.compress_caesar_d(corpus.loader)
        latent_bytes = int(round(float(latent_bytes_tensor.item())))
        encode_seconds = time.perf_counter() - start

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        decode_start = time.perf_counter()
        if args.mode == "all_frame_vae_oracle":
            blocked = reconstruct_oracle(compressor, corpus)
        else:
            blocked = reconstruct_from_keyframes(
                compressor,
                compressed_batches,
                tuple(corpus.dataset.data_input.shape),
                args.mode,
                args.refine_start_timestep,
                args.ensemble_size,
            )
        reconstructed = corpus.dataset.recons_data(
            compressor.transform_shape(blocked)
        ).detach().cpu().numpy()
        decode_seconds = time.perf_counter() - decode_start

        metrics = base_metrics(
            corpus.original,
            reconstructed,
            latent_bytes,
            (encode_seconds, decode_seconds),
            group_count=len(corpus.dataset),
        )
        keyframes = compressor.cond_idx.tolist()
        predicted = [index for index in range(corpus.n_frame) if index not in keyframes]
        per_frame = []
        for frame in range(corpus.n_frame):
            psnr, mse = calculate_psnr(
                corpus.original[:, :, frame], reconstructed[:, :, frame]
            )
            per_frame.append({"frame": frame, "psnr": psnr, "mse": mse})

        result = {
            "role": "diagnostic_only_not_a_formal_codec_result",
            "mode": args.mode,
            "checkpoint": str(args.checkpoint.resolve()),
            "variables": int(normalized.shape[0]),
            "variable_start": args.variable_start,
            "variable_end_exclusive": variable_end,
            "shape": list(corpus.original.shape),
            "seed": args.seed,
            "interpo_rate": args.interpo_rate,
            "condition_indices_override": args.condition_indices,
            "ensemble_size": (
                args.ensemble_size if args.mode == "diffusion_ensemble" else None
            ),
            "refine_start_timestep": (
                args.refine_start_timestep if args.mode == "linear_refine" else None
            ),
            "condition_frame_count": len(keyframes),
            "latent_bytes_for_condition_frames": latent_bytes,
            "scientific_bpp_for_condition_stream": latent_bytes * 8 / corpus.original.size,
            "all_frames": metrics,
            "keyframes": subset_metrics(corpus.original, reconstructed, keyframes),
            "predicted_frames": subset_metrics(corpus.original, reconstructed, predicted),
            "per_frame": per_frame,
            "keyframe_indices": keyframes,
            "predicted_indices": predicted,
            "oracle_rate_warning": (
                "The oracle reconstruction encodes all 16 frames only through forward quantization, "
                "while the displayed bit count contains only the configured condition-frame stream "
                "and is not a valid oracle BPP."
                if args.mode == "all_frame_vae_oracle"
                else None
            ),
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        corpus.close()


if __name__ == "__main__":
    main()
