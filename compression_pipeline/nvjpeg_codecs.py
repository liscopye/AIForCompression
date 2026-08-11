from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np

from compression_pipeline.canonical import CanonicalSample
from compression_pipeline.metrics import base_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_nvjpeg2k_binary(binary: Path | None = None) -> Path:
    binary = binary or PROJECT_ROOT / "tools" / "nvjpeg" / "nvjpeg2k_roundtrip"
    binary.parent.mkdir(parents=True, exist_ok=True)
    src = PROJECT_ROOT / "tools" / "nvjpeg" / "nvjpeg2k_roundtrip.cpp"
    include = Path("/usr/local/lib/python3.12/dist-packages/nvidia/nvjpeg2k/include")
    lib = Path("/usr/local/lib/python3.12/dist-packages/nvidia/nvjpeg2k/lib")
    if not src.exists():
        raise FileNotFoundError(f"missing nvJPEG2000 source: {src}")
    if not include.joinpath("nvjpeg2k.h").exists():
        raise FileNotFoundError(f"missing nvJPEG2000 header: {include / 'nvjpeg2k.h'}")
    so_path = lib / "libnvjpeg2k.so.0"
    if not so_path.exists():
        raise FileNotFoundError(f"missing nvJPEG2000 library: {so_path}")
    if binary.exists() and binary.stat().st_mtime >= src.stat().st_mtime:
        return binary
    cmd = [
        "g++",
        "-O3",
        "-std=c++17",
        str(src),
        "-I",
        str(include),
        "-I",
        "/usr/local/cuda/include",
        "-L",
        str(lib),
        "-L",
        "/usr/local/cuda/lib64",
        str(so_path),
        "-lcudart",
        f"-Wl,-rpath,{lib}",
        "-Wl,-rpath,/usr/local/cuda/lib64",
        "-o",
        str(binary),
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return binary


def build_nvjpeg_binary(binary: Path | None = None) -> Path:
    binary = binary or PROJECT_ROOT / "tools" / "nvjpeg" / "nvjpeg_roundtrip"
    binary.parent.mkdir(parents=True, exist_ok=True)
    src = PROJECT_ROOT / "tools" / "nvjpeg" / "nvjpeg_roundtrip.cpp"
    include = Path("/usr/local/cuda/include")
    lib = Path("/usr/local/cuda/lib64")
    if not include.joinpath("nvjpeg.h").exists():
        include = Path("/usr/local/cuda/targets/x86_64-linux/include")
        lib = Path("/usr/local/cuda/targets/x86_64-linux/lib")
    if not src.exists():
        raise FileNotFoundError(f"missing nvJPEG source: {src}")
    if not include.joinpath("nvjpeg.h").exists():
        raise FileNotFoundError(f"missing nvJPEG header: {include / 'nvjpeg.h'}")
    if binary.exists() and binary.stat().st_mtime >= src.stat().st_mtime:
        return binary
    cmd = [
        "g++",
        "-O3",
        "-std=c++17",
        str(src),
        "-I",
        str(include),
        "-L",
        str(lib),
        "-lnvjpeg",
        "-lcudart",
        f"-Wl,-rpath,{lib}",
        "-o",
        str(binary),
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return binary


def run_nvjpeg_sample(
    sample: CanonicalSample,
    quality: int,
    output_dir: Path,
    lpips_fn: Callable[[np.ndarray, np.ndarray], float | None] | None = None,
    memory_fn: Callable[[], dict[str, float | None]] | None = None,
    binary: Path | None = None,
    keep_tmp: bool = False,
    fixed_unit_range: bool = False,
) -> dict:
    binary = build_nvjpeg_binary(binary)
    original = _as_chw_native(sample.array)
    if fixed_unit_range and original.dtype != np.uint8:
        if float(original.min()) < -2e-5 or float(original.max()) > 1.00002:
            raise ValueError(f"fixed_unit_range requires [0,1] input, got [{original.min()}, {original.max()}]")
        rgb_u8 = np.rint(original.astype(np.float32) * 255.0).astype(np.uint8)
        side_info_bytes = 0
        normalization = "dataset_fixed_unit_range_uint8"
    else:
        rgb_u8, side_info_bytes, normalization = _as_rgb_u8(original)
    t0 = time.time()

    tmp_ctx = None
    if keep_tmp:
        tmp_root = output_dir / "nvjpeg_tmp" / f"{sample.sample_id}_q{quality}"
        tmp_root.mkdir(parents=True, exist_ok=True)
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="nvjpeg_", dir=str(output_dir))
        tmp_root = Path(tmp_ctx.name)

    try:
        decoded_u8, meta = _run_nvjpeg_roundtrip(binary, rgb_u8, int(quality), tmp_root)
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    if fixed_unit_range and original.dtype != np.uint8:
        reconstruction = decoded_u8.astype(np.float32) / 255.0
        original_eval = original.astype(np.float32, copy=False)
    else:
        reconstruction = decoded_u8.astype(np.float32)
        original_eval = original.astype(np.float32) if original.dtype == np.uint8 else rgb_u8.astype(np.float32)

    extra_metrics: dict[str, float | int | str | None] = {
        "lpips": lpips_fn(original_eval, reconstruction) if lpips_fn is not None else None,
        "quality": int(quality),
        "codec_backend": "nvjpeg",
        "normalization": normalization,
        "sample_wall_time_total": time.time() - t0,
    }
    if memory_fn is not None:
        extra_metrics.update(memory_fn())
    metrics = base_metrics(
        original_eval,
        reconstruction,
        int(meta["jpeg_bytes"]),
        (float(meta["encode_us"]) / 1e6, float(meta["decode_us"]) / 1e6),
        group_count=1,
        side_info_bytes=side_info_bytes,
        extra_metrics=extra_metrics,
    )
    metrics["sample_wall_throughput_MBps"] = (
        metrics["original_bytes"] / metrics["sample_wall_time_total"] / 1e6
        if metrics.get("sample_wall_time_total")
        else None
    )
    metrics.update(
        {
            "dataset_id": sample.dataset_id,
            "sample_id": sample.sample_id,
            "sample_kind": sample.kind,
            "shape": list(original_eval.shape),
            "groups": 1,
            "model_name": "nvJPEG",
            "model_id": f"nvjpeg_q{int(quality)}",
            "metric": "quality",
            "checkpoint": None,
            "params": 0,
            "image_eval_mode": "external",
        }
    )
    return metrics


def run_nvjpeg2k_sample(
    sample: CanonicalSample,
    target_psnr: float,
    output_dir: Path,
    lpips_fn: Callable[[np.ndarray, np.ndarray], float | None] | None = None,
    memory_fn: Callable[[], dict[str, float | None]] | None = None,
    valid_mask: np.ndarray | None = None,
    metric_extras: dict | None = None,
    binary: Path | None = None,
    keep_tmp: bool = False,
    fixed_unit_range: bool = False,
) -> dict:
    binary = build_nvjpeg2k_binary(binary)
    original = _as_chw(sample.array)
    mask = _as_chw_mask(valid_mask, original.shape) if valid_mask is not None else None
    reconstruction = np.empty_like(original, dtype=np.float32)
    bitstream_bytes = 0
    side_info_bytes = 0
    encode_time = 0.0
    decode_time = 0.0
    t0 = time.time()

    tmp_ctx = None
    if keep_tmp:
        tmp_root = output_dir / "nvjpeg2k_tmp" / f"{sample.sample_id}_psnr{target_psnr:g}"
        tmp_root.mkdir(parents=True, exist_ok=True)
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="nvjpeg2k_", dir=str(output_dir))
        tmp_root = Path(tmp_ctx.name)

    try:
        if fixed_unit_range:
            if float(original.min()) < -2e-5 or float(original.max()) > 1.00002:
                raise ValueError(f"fixed_unit_range requires [0,1] input, got [{original.min()}, {original.max()}]")
            quantized = np.rint(original * 65535.0).astype(np.uint16)
            mins = np.zeros(original.shape[0], dtype=np.float32)
            scales = np.ones(original.shape[0], dtype=np.float32)
        else:
            quantized, mins, scales = _quantize_chw_u16(original)
        decoded, meta = _run_roundtrip_stack(binary, quantized, float(target_psnr), tmp_root)
        reconstruction[...] = _dequantize_chw_u16(decoded, mins, scales)
        bitstream_bytes = int(meta["j2k_bytes"])
        side_info_bytes = 0 if fixed_unit_range else int(original.shape[0] * 2 * np.dtype(np.float32).itemsize)
        encode_time = float(meta["encode_us"]) / 1e6
        decode_time = float(meta["decode_us"]) / 1e6
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    extra_metrics: dict[str, float | int | str | None] = {
        "lpips": lpips_fn(original, reconstruction) if lpips_fn is not None else None,
        "target_jpeg2000_psnr": float(target_psnr),
        "codec_backend": "nvjpeg2k",
        "normalization": "dataset_fixed_unit_range_uint16" if fixed_unit_range else "per_channel_min_range_uint16",
        "sample_wall_time_total": time.time() - t0,
    }
    if memory_fn is not None:
        extra_metrics.update(memory_fn())
    if metric_extras:
        extra_metrics.update(metric_extras)

    metrics = base_metrics(
        original,
        reconstruction,
        bitstream_bytes,
        (encode_time, decode_time),
        group_count=int(original.shape[0]),
        side_info_bytes=side_info_bytes,
        valid_mask=mask,
        extra_metrics=extra_metrics,
    )
    metrics["sample_wall_throughput_MBps"] = (
        metrics["original_bytes"] / metrics["sample_wall_time_total"] / 1e6
        if metrics.get("sample_wall_time_total")
        else None
    )
    metrics.update(
        {
            "dataset_id": sample.dataset_id,
            "sample_id": sample.sample_id,
            "sample_kind": sample.kind,
            "shape": list(original.shape),
            "groups": int(original.shape[0]),
            "model_name": "nvJPEG2000",
            "model_id": f"nvjpeg2k_psnr{target_psnr:g}",
            "metric": "target_psnr",
            "checkpoint": None,
            "params": 0,
            "image_eval_mode": "external",
        }
    )
    return metrics


def _as_chw(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        return np.ascontiguousarray(arr[None])
    if arr.ndim == 3:
        if arr.shape[0] <= 512:
            return np.ascontiguousarray(arr)
        if arr.shape[-1] <= 4:
            return np.ascontiguousarray(np.moveaxis(arr, -1, 0))
        return np.ascontiguousarray(arr.reshape((-1,) + arr.shape[-2:]))
    if arr.ndim > 3:
        return np.ascontiguousarray(arr.reshape((-1,) + arr.shape[-2:]))
    raise ValueError(f"nvJPEG2000 expects at least 2D array, got {arr.shape}")


def _as_chw_native(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 2:
        return np.ascontiguousarray(arr[None])
    if arr.ndim == 3:
        if arr.shape[0] <= 512:
            return np.ascontiguousarray(arr)
        if arr.shape[-1] <= 4:
            return np.ascontiguousarray(np.moveaxis(arr, -1, 0))
    raise ValueError(f"expected CHW or HWC image-like array, got {arr.shape}")


def _as_rgb_u8(array: np.ndarray) -> tuple[np.ndarray, int, str]:
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError(f"nvJPEG expects one RGB CHW sample, got {array.shape}")
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array), 0, "native_uint8_rgb"
    arr = array.astype(np.float32, copy=False)
    mins = arr.reshape(3, -1).min(axis=1).astype(np.float32)
    maxs = arr.reshape(3, -1).max(axis=1).astype(np.float32)
    scales = np.where(maxs > mins, maxs - mins, 1.0).astype(np.float32)
    q = np.rint((arr - mins[:, None, None]) / scales[:, None, None] * 255.0)
    side_info_bytes = int(3 * 2 * np.dtype(np.float32).itemsize)
    return np.clip(q, 0, 255).astype(np.uint8), side_info_bytes, "per_channel_min_range_uint8"


def _run_nvjpeg_roundtrip(
    binary: Path,
    rgb_u8: np.ndarray,
    quality: int,
    tmp_root: Path,
) -> tuple[np.ndarray, dict]:
    _, h, w = rgb_u8.shape
    raw_in = tmp_root / "input.rgb"
    raw_out = tmp_root / "output.rgb"
    np.moveaxis(rgb_u8, 0, -1).copy().tofile(raw_in)
    proc = subprocess.run(
        [str(binary), str(raw_in), str(w), str(h), str(quality), str(raw_out)],
        check=True,
        text=True,
        capture_output=True,
    )
    meta = json.loads(proc.stdout)
    decoded = np.fromfile(raw_out, dtype=np.uint8).reshape(h, w, 3)
    return np.moveaxis(decoded, -1, 0), meta


def _as_chw_mask(mask: np.ndarray, expected_shape: tuple[int, int, int]) -> np.ndarray:
    mask_chw = _as_chw(np.asarray(mask, dtype=bool))
    if mask_chw.shape != expected_shape:
        raise ValueError(f"valid mask shape {mask_chw.shape} does not match nvJPEG2000 input {expected_shape}")
    return mask_chw


def _quantize_channel_u16(channel: np.ndarray) -> tuple[np.ndarray, float, float]:
    arr = channel.astype(np.float32, copy=False)
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    scale = vmax - vmin
    if scale <= 1e-12:
        scale = 1.0
    q = np.rint((arr - vmin) / scale * 65535.0)
    return np.clip(q, 0, 65535).astype(np.uint16), vmin, scale


def _dequantize_channel_u16(channel: np.ndarray, vmin: float, scale: float) -> np.ndarray:
    return channel.astype(np.float32) / 65535.0 * scale + vmin


def _quantize_chw_u16(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = array.reshape(array.shape[0], -1).astype(np.float32, copy=False)
    mins = flat.min(axis=1).astype(np.float32)
    maxs = flat.max(axis=1).astype(np.float32)
    scales = (maxs - mins).astype(np.float32)
    scales = np.where(scales > 1e-12, scales, 1.0).astype(np.float32)
    q = np.rint((array - mins[:, None, None]) / scales[:, None, None] * 65535.0)
    return np.clip(q, 0, 65535).astype(np.uint16), mins, scales


def _dequantize_chw_u16(array: np.ndarray, mins: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return array.astype(np.float32) / 65535.0 * scales[:, None, None] + mins[:, None, None]


def _run_roundtrip_stack(
    binary: Path,
    array_u16: np.ndarray,
    target_psnr: float,
    tmp_root: Path,
) -> tuple[np.ndarray, dict]:
    z, h, w = array_u16.shape
    raw_in = tmp_root / "input_zhw.u16"
    raw_out = tmp_root / "output_zhw.u16"
    np.ascontiguousarray(array_u16, dtype=np.uint16).tofile(raw_in)
    proc = subprocess.run(
        [str(binary), str(raw_in), str(w), str(h), str(z), str(target_psnr), str(raw_out)],
        check=True,
        text=True,
        capture_output=True,
    )
    meta = json.loads(proc.stdout)
    decoded = np.fromfile(raw_out, dtype=np.uint16).reshape(z, h, w)
    return decoded, meta


def _run_roundtrip(
    binary: Path,
    channel_u16: np.ndarray,
    target_psnr: float,
    tmp_root: Path,
    channel_idx: int,
) -> tuple[np.ndarray, dict]:
    h, w = channel_u16.shape
    raw_in = tmp_root / f"input_c{channel_idx:04d}.u16"
    raw_out = tmp_root / f"output_c{channel_idx:04d}.u16"
    channel_u16.astype(np.uint16, copy=False).tofile(raw_in)
    proc = subprocess.run(
        [str(binary), str(raw_in), str(w), str(h), str(target_psnr), str(raw_out)],
        check=True,
        text=True,
        capture_output=True,
    )
    meta = json.loads(proc.stdout)
    decoded = np.fromfile(raw_out, dtype=np.uint16).reshape(h, w)
    return decoded, meta
