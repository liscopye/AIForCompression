from __future__ import annotations

import math
import os
import resource
from functools import lru_cache
from typing import Callable

import numpy as np


def calculate_psnr(
    original: np.ndarray,
    reconstructed: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> tuple[float, float]:
    orig64 = original.astype(np.float64)
    recon64 = reconstructed.astype(np.float64)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != original.shape:
            raise ValueError(f"valid_mask shape {mask.shape} does not match original shape {original.shape}")
        if not np.any(mask):
            raise ValueError("valid_mask has no valid elements")
        orig64 = orig64[mask]
        recon64 = recon64[mask]
    mse = float(np.mean((orig64 - recon64) ** 2))
    if mse < 1e-30:
        return 300.0, mse
    data_range = float(orig64.max() - orig64.min())
    if data_range < 1e-8:
        data_range = 1.0
    return float(10 * np.log10(data_range ** 2 / mse)), mse


def calculate_axis0_average_psnr(
    original: np.ndarray,
    reconstructed: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> float | None:
    """Average PSNR over the first axis, useful for multi-variable scientific fields."""
    if original.shape != reconstructed.shape or original.ndim < 3:
        return None
    values = []
    masks = valid_mask if valid_mask is not None else [None] * original.shape[0]
    for orig_item, recon_item, mask_item in zip(original, reconstructed, masks):
        psnr, _ = calculate_psnr(orig_item, recon_item, mask_item)
        if math.isfinite(psnr):
            values.append(psnr)
    if not values:
        return None
    return float(np.mean(values))


def calculate_frame_average_psnr(
    original: np.ndarray,
    reconstructed: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> float | None:
    """Average per-frame PSNR using each frame's own dynamic range.

    For CAESAR-style tensors [V,S,T,H,W], the frame axis is T.  For image-style
    tensors [C,H,W], the single image PSNR is already the frame PSNR.
    """
    if original.shape != reconstructed.shape or original.ndim < 3:
        return None
    if original.ndim == 3:
        psnr, _ = calculate_psnr(original, reconstructed, valid_mask)
        return psnr
    if original.ndim >= 5:
        frame_axis = -3
        values = []
        for index in range(original.shape[frame_axis]):
            orig_frame = np.take(original, index, axis=frame_axis)
            recon_frame = np.take(reconstructed, index, axis=frame_axis)
            mask_frame = np.take(valid_mask, index, axis=frame_axis) if valid_mask is not None else None
            psnr, _ = calculate_psnr(orig_frame, recon_frame, mask_frame)
            if math.isfinite(psnr):
                values.append(psnr)
        if not values:
            return None
        return float(np.mean(values))
    return None


def base_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    bitstream_bytes: int,
    elapsed: tuple[float, float],
    group_count: int = 1,
    side_info_bytes: int = 0,
    valid_mask: np.ndarray | None = None,
    extra_metrics: dict[str, float | int | None] | None = None,
) -> dict[str, float | int | None]:
    psnr, mse = calculate_psnr(original, reconstructed, valid_mask)
    original_bytes = int(original.size * original.dtype.itemsize)
    encode_time, decode_time = elapsed
    groups = max(int(group_count), 1)
    spatial_pixels = int(original.shape[-2] * original.shape[-1])
    scientific_symbols = int(original.size)
    total_bytes_with_side_info = int(bitstream_bytes + max(side_info_bytes, 0))
    encode_throughput = original_bytes / encode_time if encode_time > 0 else None
    decode_throughput = original_bytes / decode_time if decode_time > 0 else None
    metrics = {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "psnr": psnr,
        "average_variable_psnr": calculate_axis0_average_psnr(original, reconstructed, valid_mask),
        "average_frame_psnr": calculate_frame_average_psnr(original, reconstructed, valid_mask),
        "image_bpp": bitstream_bytes * 8.0 / spatial_pixels,
        "bpp": bitstream_bytes * 8.0 / scientific_symbols,
        "scientific_bpp": bitstream_bytes * 8.0 / scientific_symbols,
        "scientific_bpp_with_side_info": total_bytes_with_side_info * 8.0 / scientific_symbols,
        "bitstream_bytes": int(bitstream_bytes),
        "side_info_bytes": int(max(side_info_bytes, 0)),
        "total_bytes_with_side_info": total_bytes_with_side_info,
        "original_bytes": original_bytes,
        "compression_ratio": original_bytes / total_bytes_with_side_info if total_bytes_with_side_info > 0 else float("inf"),
        "group_count": groups,
        "encode_time_total": encode_time,
        "decode_time_total": decode_time,
        "encode_time_per_group_avg": encode_time / groups,
        "decode_time_per_group_avg": decode_time / groups,
        "encode_throughput_MBps": encode_throughput / 1e6 if encode_throughput is not None else None,
        "decode_throughput_MBps": decode_throughput / 1e6 if decode_throughput is not None else None,
        "sample_wall_time_total": None,
        "sample_wall_throughput_MBps": None,
        # Legacy names kept for existing plot/aggregation scripts. These are totals for grouped samples.
        "encode_time_avg": encode_time,
        "decode_time_avg": decode_time,
        "encode_throughput": encode_throughput,
        "decode_throughput": decode_throughput,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    return metrics


def process_memory_usage_mb() -> float | None:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / 1024**2)
    except Exception:
        try:
            return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
        except Exception:
            return None


def torch_memory_usage_mb(device: str = "cuda") -> dict[str, float | None]:
    if not str(device).startswith("cuda"):
        return {"memory_usage_MB": process_memory_usage_mb(), "memory_reserved_MB": None}
    try:
        import torch

        if not torch.cuda.is_available():
            return {"memory_usage_MB": process_memory_usage_mb(), "memory_reserved_MB": None}
        torch.cuda.synchronize()
        return {
            "memory_usage_MB": float(torch.cuda.max_memory_allocated() / 1024**2),
            "memory_reserved_MB": float(torch.cuda.max_memory_reserved() / 1024**2),
        }
    except Exception:
        return {"memory_usage_MB": process_memory_usage_mb(), "memory_reserved_MB": None}


def reset_torch_peak_memory(device: str = "cuda") -> None:
    if not str(device).startswith("cuda"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def make_lpips_fn(device: str = "cuda") -> Callable[[np.ndarray, np.ndarray], float | None]:
    model = _lpips_model(device)

    def _calculate(original: np.ndarray, reconstructed: np.ndarray) -> float | None:
        if model is None:
            return None
        try:
            import torch

            values = []
            with torch.no_grad():
                for original_image, reconstructed_image in _iter_lpips_image_pairs(original, reconstructed):
                    original_tensor = _lpips_tensor(original_image, device)
                    reconstructed_tensor = _lpips_tensor(reconstructed_image, device)
                    value = model(original_tensor, reconstructed_tensor)
                    values.append(float(value.mean().detach().cpu().item()))
            if not values:
                return None
            return float(np.mean(values))
        except Exception:
            return None

    return _calculate


@lru_cache(maxsize=4)
def _lpips_model(device: str):
    try:
        import lpips

        model = lpips.LPIPS(net="alex")
        model.eval()
        return model.to(device)
    except Exception:
        return None


def _lpips_tensor(array: np.ndarray, device: str):
    import torch

    arr = array.astype(np.float32, copy=False)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"LPIPS expects [C,H,W] or [H,W], got {array.shape}")
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    elif arr.shape[0] > 3:
        arr = arr[:3]
    elif arr.shape[0] == 2:
        arr = np.concatenate([arr, arr[-1:]], axis=0)
    arr_min = float(np.min(arr))
    arr_max = float(np.max(arr))
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    arr = arr * 2.0 - 1.0
    return torch.from_numpy(arr[None]).to(device=device, dtype=torch.float32)


def _iter_lpips_image_pairs(original: np.ndarray, reconstructed: np.ndarray):
    if original.shape != reconstructed.shape:
        raise ValueError(f"LPIPS arrays must have the same shape, got {original.shape} and {reconstructed.shape}")
    if original.ndim == 2:
        yield original, reconstructed
        return
    if original.ndim == 3 and original.shape[0] <= 4:
        yield original, reconstructed
        return
    if original.ndim < 3:
        raise ValueError(f"LPIPS expects at least 2D arrays, got {original.shape}")
    original_flat = original.reshape((-1,) + original.shape[-2:])
    reconstructed_flat = reconstructed.reshape((-1,) + reconstructed.shape[-2:])
    for original_image, reconstructed_image in zip(original_flat, reconstructed_flat):
        yield original_image, reconstructed_image
