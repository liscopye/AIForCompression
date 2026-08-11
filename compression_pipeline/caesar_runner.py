from __future__ import annotations

import sys
import tempfile
import time
import types
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from compression_pipeline.metrics import (
    calculate_axis0_average_psnr,
    calculate_frame_average_psnr,
    calculate_psnr,
    make_lpips_fn,
    reset_torch_peak_memory,
    torch_memory_usage_mb,
)
from compression_pipeline.views import CaesarView, build_caesar_view


CAESAR_N_FRAMES = {
    "caesar_v": 8,
    "caesar_d": 16,
}


@dataclass(frozen=True)
class CaesarWindow:
    view: CaesarView
    timestamps: list[str]
    start_index: int


def build_caesar_window(
    sequence_vthw: np.ndarray,
    timestamps: list[str],
    n_frame: int,
    start_index: int = 0,
    sample_id: str = "era5_sequence",
) -> CaesarWindow:
    if sequence_vthw.ndim != 4:
        raise ValueError(f"CAESAR sequence must be [V,T,H,W], got {sequence_vthw.shape}")
    if len(timestamps) != sequence_vthw.shape[1]:
        raise ValueError(f"timestamps length {len(timestamps)} does not match sequence T={sequence_vthw.shape[1]}")
    if start_index < 0:
        raise ValueError(f"start_index must be non-negative, got {start_index}")
    end_index = start_index + n_frame
    if end_index > sequence_vthw.shape[1]:
        raise ValueError(f"CAESAR requires {n_frame} contiguous frames from {start_index}, got T={sequence_vthw.shape[1]}")

    window_timestamps = timestamps[start_index:end_index]
    validate_regular_timestamps(window_timestamps)
    window = sequence_vthw[:, start_index:end_index]
    return CaesarWindow(
        view=build_caesar_view(window, sample_id=sample_id, n_frame=n_frame),
        timestamps=window_timestamps,
        start_index=start_index,
    )


def validate_regular_timestamps(timestamps: list[str]) -> None:
    if len(timestamps) <= 2:
        return
    parsed = [_parse_timestamp(ts) for ts in timestamps]
    expected_delta = parsed[1] - parsed[0]
    if expected_delta.total_seconds() <= 0:
        raise ValueError(f"CAESAR timestamps must be strictly increasing: {timestamps[:2]}")
    tolerance = timedelta(microseconds=100)
    for left, right in zip(parsed[1:], parsed[2:]):
        delta = right - left
        if abs(delta - expected_delta) > tolerance:
            raise ValueError(f"CAESAR requires a regular contiguous time window, got timestamps={timestamps}")


def run_caesar_sequence(
    sequence_vthw: np.ndarray,
    timestamps: list[str],
    model_name: str,
    caesar_root: str | Path,
    ckpt_dir: str | Path,
    output_dir: str | Path,
    device: str,
    batch_size: int = 8,
    eb: float = 1e-4,
    start_index: int = 0,
    sample_id: str = "caesar_sequence",
    collect_lpips: bool = True,
    use_pca_postprocess: bool = True,
    valid_mask_vthw: np.ndarray | None = None,
    metric_extras: dict | None = None,
    norm_type: str = "mean_range",
) -> dict[str, Any]:
    if model_name not in CAESAR_N_FRAMES:
        raise ValueError(f"Unsupported CAESAR model: {model_name}")
    n_frame = CAESAR_N_FRAMES[model_name]
    window = build_caesar_window(sequence_vthw, timestamps, n_frame=n_frame, start_index=start_index, sample_id=sample_id)

    caesar_root = Path(caesar_root)
    ckpt_dir = Path(ckpt_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(caesar_root))
    from CAESAR.compressor import CAESAR
    from dataset import ScientificDataset

    setup_start = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=f"_{model_name}_era5.npz", dir=output_dir, delete=False) as tmp:
        npz_path = Path(tmp.name)
    try:
        np.savez(npz_path, data=window.view.tensor)
        data_arg = {
            "data_path": str(npz_path),
            "name": f"ERA5-{sequence_vthw.shape[0]}-{model_name}",
            "variable_idx": list(range(sequence_vthw.shape[0])),
            "section_range": [0, 1],
            "frame_range": [0, n_frame],
            "n_frame": n_frame,
            "train": False,
            "test_size": (256, 256),
            "inst_norm": True,
            "norm_type": norm_type,
        }
        dataset = ScientificDataset(data_arg)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        setup_time = time.perf_counter() - setup_start
        model_load_start = time.perf_counter()
        compressor = CAESAR(
            model_path=str(_resolve_caesar_checkpoint(ckpt_dir, model_name)),
            use_diffusion=(model_name == "caesar_d"),
            device=device,
            n_frame=n_frame,
        )
        model_load_time = time.perf_counter() - model_load_start
        if not use_pca_postprocess:
            _disable_caesar_pca_postprocess(compressor)
        params = _count_caesar_params(compressor, model_name)

        reset_torch_peak_memory(device)
        codec_wall_start = time.perf_counter()
        t0 = time.time()
        compressed, compressed_size = compressor.compress(loader, eb=eb)
        t1 = time.time()
        reconstructed = compressor.decompress(compressed)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t2 = time.time()

        original = dataset.input_data().numpy()
        recon = dataset.recons_data(reconstructed).detach().cpu().numpy()
        e2e_end = time.perf_counter()
        eval_mask = None
        if valid_mask_vthw is not None:
            eval_mask = valid_mask_vthw[:, start_index : start_index + n_frame]
            if eval_mask.ndim == original.ndim - 1 and original.shape[1] == 1:
                eval_mask = eval_mask[:, None, ...]
            if eval_mask.shape != original.shape:
                raise ValueError(f"valid_mask_vthw window shape {eval_mask.shape} does not match original {original.shape}")
        psnr, mse = calculate_psnr(original, recon, eval_mask)
        lpips_value = make_lpips_fn(device)(original, recon) if collect_lpips else None
        memory_metrics = torch_memory_usage_mb(device)
        compressed_bytes = float(compressed_size.item() if hasattr(compressed_size, "item") else compressed_size)
        original_bytes = int(original.size * 4)
        scientific_bpp = compressed_bytes * 8.0 / original.size
        result_model_id = model_name if use_pca_postprocess else f"{model_name}_no_pca"
        result = {
            "model_name": "CAESAR",
            "model_id": result_model_id,
            "sample_id": sample_id,
            "metric": "mse",
            "params": params,
            "model_view": "caesar_vsthw",
            "caesar_postprocess": "pca" if use_pca_postprocess else "none",
            "caesar_norm_type": norm_type,
            "timestamps": window.timestamps,
            "start_index": start_index,
            "shape": list(original.shape),
            "original_min": float(np.min(original[eval_mask] if eval_mask is not None else original)),
            "original_max": float(np.max(original[eval_mask] if eval_mask is not None else original)),
            "voxel_count": int(np.count_nonzero(eval_mask) if eval_mask is not None else original.size),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "psnr": psnr,
            "average_variable_psnr": calculate_axis0_average_psnr(original, recon, eval_mask),
            "average_frame_psnr": calculate_frame_average_psnr(original, recon, eval_mask),
            "bpp": scientific_bpp,
            "scientific_bpp": scientific_bpp,
            "scientific_bpp_with_side_info": scientific_bpp,
            "bitstream_bytes": compressed_bytes,
            "side_info_bytes": 0,
            "total_bytes_with_side_info": compressed_bytes,
            "original_bytes": original_bytes,
            "compression_ratio": original_bytes / compressed_bytes if compressed_bytes > 0 else float("inf"),
            "encode_time_avg": t1 - t0,
            "decode_time_avg": t2 - t1,
            "encode_throughput": original_bytes / (t1 - t0) if t1 > t0 else None,
            "decode_throughput": original_bytes / (t2 - t1) if t2 > t1 else None,
            "sample_setup_time": setup_time,
            "model_load_time": model_load_time,
            "sample_wall_time_total": setup_time + (e2e_end - codec_wall_start),
            "sample_wall_throughput_MBps": original_bytes / (setup_time + (e2e_end - codec_wall_start)) / 1e6,
            "lpips": lpips_value,
            "memory_usage_MB": memory_metrics.get("memory_usage_MB"),
            "memory_reserved_MB": memory_metrics.get("memory_reserved_MB"),
        }
        if metric_extras:
            result.update(metric_extras)
        return result
    finally:
        npz_path.unlink(missing_ok=True)


def _disable_caesar_pca_postprocess(compressor: Any) -> None:
    def postprocessing_encoding_noop(self, original_data, recons_data, nrmse):
        del original_data, recons_data, nrmse
        return {"data_bytes": 0, "postprocess": "none"}, None

    def postprocessing_decoding_noop(self, recons_data, meta_data, compressed_data, padding):
        del meta_data, compressed_data
        return self.unpadding(recons_data, padding)

    compressor.postprocessing_encoding = types.MethodType(postprocessing_encoding_noop, compressor)
    compressor.postprocessing_decoding = types.MethodType(postprocessing_decoding_noop, compressor)


def _parse_timestamp(timestamp: str) -> datetime:
    normalized = timestamp.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass
    # Tomography angle timestamps: angle_X.XXXX → treat as seconds from epoch
    if timestamp.startswith("angle_"):
        try:
            angle_sec = float(timestamp.removeprefix("angle_"))
            return datetime(2000, 1, 1) + timedelta(seconds=angle_sec)
        except ValueError:
            pass
    # Reconstructed slice timestamps: slice_NNNN → treat as seconds from epoch
    if timestamp.startswith("slice_"):
        try:
            slice_idx = int(timestamp.removeprefix("slice_"))
            return datetime(2000, 1, 1) + timedelta(seconds=slice_idx)
        except ValueError:
            pass
    if timestamp.startswith("turb_rot_t"):
        try:
            frame_idx = int(timestamp.removeprefix("turb_rot_t"))
            return datetime(2000, 1, 1) + timedelta(seconds=frame_idx)
        except ValueError:
            pass
    if timestamp.startswith("e3sm_t"):
        try:
            frame_idx = int(timestamp.removeprefix("e3sm_t"))
            return datetime(2000, 1, 1) + timedelta(seconds=frame_idx)
        except ValueError:
            pass
    if timestamp.startswith("era5_t"):
        try:
            frame_idx = int(timestamp.removeprefix("era5_t"))
            return datetime(2000, 1, 1) + timedelta(seconds=frame_idx)
        except ValueError:
            pass
    raise ValueError(f"Unsupported timestamp format for CAESAR: {timestamp}")


def _resolve_caesar_checkpoint(ckpt_dir: str | Path, model_name: str) -> Path:
    ckpt_dir = Path(ckpt_dir)
    if ckpt_dir.is_file():
        return ckpt_dir
    exact = ckpt_dir / f"{model_name}.pth"
    if exact.exists():
        return exact
    candidates: list[Path] = []
    for extension in (".pth", ".pt", ".pth.tar"):
        candidates.extend(sorted(ckpt_dir.glob(f"{model_name}*{extension}")))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"No checkpoint found for {model_name} in {ckpt_dir}; expected {exact.name} "
        f"or a file matching {model_name}*.pt/.pth"
    )


def _count_caesar_params(compressor: Any, model_name: str) -> int:
    if model_name == "caesar_v":
        return sum(p.numel() for p in compressor.compressor_v.parameters())
    total = sum(p.numel() for p in compressor.keyframe_model.parameters())
    diffusion = getattr(compressor, "diffusion_model", None)
    if diffusion is not None:
        total += sum(p.numel() for p in diffusion.parameters())
    return total
