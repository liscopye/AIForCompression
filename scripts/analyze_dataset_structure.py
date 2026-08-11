#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


ROOT = Path("/workspace")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_OUTPUT = Path("unified_results/analysis/benchmark_observations/dataset_structure.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure structure relevant to compression in benchmark tensors.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="*", default=[])
    return parser.parse_args()


def safe_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    mask = np.isfinite(left) & np.isfinite(right)
    left = left[mask].astype(np.float64, copy=False)
    right = right[mask].astype(np.float64, copy=False)
    if left.size < 4 or np.std(left) < 1e-20 or np.std(right) < 1e-20:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def finite_percentile(values: np.ndarray, percentiles: list[float]) -> list[float]:
    values = values[np.isfinite(values)]
    if not values.size:
        return [float("nan")] * len(percentiles)
    return [float(value) for value in np.percentile(values, percentiles)]


def summarize_volume(
    array: np.ndarray,
    *,
    z_semantics: str,
    invalid_threshold: float | None = None,
    seed: int = 20260721,
) -> dict[str, Any]:
    if array.ndim != 3:
        raise ValueError(f"Expected [Z,H,W], got {array.shape}")
    z_count, height, width = map(int, array.shape)
    rng = np.random.default_rng(seed)
    sample_count = min(500_000, max(50_000, z_count * 4096))
    zs = rng.integers(0, z_count, size=sample_count)
    ys = rng.integers(0, height, size=sample_count)
    xs = rng.integers(0, width, size=sample_count)
    values = np.asarray(array[zs, ys, xs], dtype=np.float64)
    valid = np.isfinite(values)
    if invalid_threshold is not None:
        valid &= values < invalid_threshold
    invalid_fraction = float(1.0 - np.mean(valid))
    values = values[valid]
    p0, p001, p01, p1, p50, p99, p999, p9999, p100 = finite_percentile(
        values, [0, 0.01, 0.1, 1, 50, 99, 99.9, 99.99, 100]
    )
    robust_range = max(p99 - p1, 1e-30)
    sampled_range = p100 - p0

    plane_indices = np.unique(np.linspace(0, z_count - 1, min(z_count, 64), dtype=int))
    pixel_count = min(8192, height * width)
    py = rng.integers(0, height, size=pixel_count)
    px = rng.integers(0, width, size=pixel_count)
    plane_matrix = np.asarray(array[plane_indices[:, None], py[None, :], px[None, :]], dtype=np.float64)
    if invalid_threshold is not None:
        plane_matrix[plane_matrix >= invalid_threshold] = np.nan

    plane_means = np.nanmean(plane_matrix, axis=1)
    plane_stds = np.nanstd(plane_matrix, axis=1)
    plane_low = np.nanpercentile(plane_matrix, 1, axis=1)
    plane_high = np.nanpercentile(plane_matrix, 99, axis=1)
    plane_ranges = plane_high - plane_low
    positive_ranges = plane_ranges[np.isfinite(plane_ranges) & (plane_ranges > 1e-20)]
    positive_stds = plane_stds[np.isfinite(plane_stds) & (plane_stds > 1e-20)]

    pair_count = min(max(z_count - 1, 0), 96)
    z_correlations: list[float] = []
    z_relative_diffs: list[float] = []
    if pair_count:
        starts = np.unique(np.linspace(0, z_count - 2, pair_count, dtype=int))
        pair_pixels = min(16384, height * width)
        qy = rng.integers(0, height, size=pair_pixels)
        qx = rng.integers(0, width, size=pair_pixels)
        for start in starts:
            left = np.asarray(array[start, qy, qx], dtype=np.float64)
            right = np.asarray(array[start + 1, qy, qx], dtype=np.float64)
            pair_valid = np.isfinite(left) & np.isfinite(right)
            if invalid_threshold is not None:
                pair_valid &= (left < invalid_threshold) & (right < invalid_threshold)
            corr = safe_corr(left[pair_valid], right[pair_valid])
            if corr is not None:
                z_correlations.append(corr)
            if np.any(pair_valid):
                z_relative_diffs.append(float(np.mean(np.abs(left[pair_valid] - right[pair_valid])) / robust_range))

    spatial_pixels = min(100_000, len(plane_indices) * 4096)
    si = rng.choice(plane_indices, size=spatial_pixels, replace=True)
    sy = rng.integers(0, max(height - 1, 1), size=spatial_pixels)
    sx = rng.integers(0, max(width - 1, 1), size=spatial_pixels)
    center = np.asarray(array[si, sy, sx], dtype=np.float64)
    right = np.asarray(array[si, sy, np.minimum(sx + 1, width - 1)], dtype=np.float64)
    down = np.asarray(array[si, np.minimum(sy + 1, height - 1), sx], dtype=np.float64)
    spatial_valid = np.isfinite(center) & np.isfinite(right) & np.isfinite(down)
    if invalid_threshold is not None:
        spatial_valid &= (center < invalid_threshold) & (right < invalid_threshold) & (down < invalid_threshold)

    pca_pixels = min(4096, pixel_count)
    pca_matrix = plane_matrix[:, :pca_pixels]
    row_means = np.nanmean(pca_matrix, axis=1, keepdims=True)
    row_stds = np.nanstd(pca_matrix, axis=1, keepdims=True)
    standardized = np.nan_to_num((pca_matrix - row_means) / np.maximum(row_stds, 1e-12))
    correlation = standardized @ standardized.T / max(standardized.shape[1], 1)
    eigvals = np.maximum(np.linalg.eigvalsh(correlation), 0)[::-1]
    eigsum = float(np.sum(eigvals))
    cumulative = np.cumsum(eigvals) / eigsum if eigsum > 0 else np.zeros_like(eigvals)

    clipped = np.clip(values, p1, p99)
    if p99 > p1:
        bins = np.minimum(((clipped - p1) / (p99 - p1) * 4095).astype(np.int64), 4095)
        counts = np.bincount(bins, minlength=4096)
        probs = counts[counts > 0] / counts.sum()
        entropy_12bit = float(-np.sum(probs * np.log2(probs)))
    else:
        entropy_12bit = 0.0

    def q(values_list: list[float], percentile: float) -> float | None:
        return float(np.percentile(values_list, percentile)) if values_list else None

    return {
        "shape": [z_count, height, width],
        "dtype": str(array.dtype),
        "z_semantics": z_semantics,
        "sampled_value_count": int(values.size),
        "invalid_fraction": invalid_fraction,
        "zero_fraction": float(np.mean(values == 0)) if values.size else None,
        "negative_fraction": float(np.mean(values < 0)) if values.size else None,
        "sampled_min": p0,
        "sampled_p001": p001,
        "sampled_p01": p01,
        "sampled_p1": p1,
        "sampled_median": p50,
        "sampled_p99": p99,
        "sampled_p999": p999,
        "sampled_p9999": p9999,
        "sampled_max": p100,
        "sampled_range_over_p1_p99_range": sampled_range / robust_range,
        "entropy_of_12bit_robust_quantization": entropy_12bit,
        "spatial_x_correlation": safe_corr(center[spatial_valid], right[spatial_valid]),
        "spatial_y_correlation": safe_corr(center[spatial_valid], down[spatial_valid]),
        "spatial_abs_diff_over_robust_range": float(
            np.mean((np.abs(center[spatial_valid] - right[spatial_valid]) + np.abs(center[spatial_valid] - down[spatial_valid])) / 2)
            / robust_range
        ),
        "adjacent_z_correlation_p10": q(z_correlations, 10),
        "adjacent_z_correlation_median": q(z_correlations, 50),
        "adjacent_z_correlation_min": min(z_correlations) if z_correlations else None,
        "adjacent_z_abs_diff_over_robust_range_median": q(z_relative_diffs, 50),
        "plane_robust_range_p95_over_p5": (
            float(np.percentile(positive_ranges, 95) / max(np.percentile(positive_ranges, 5), 1e-30))
            if positive_ranges.size > 1
            else None
        ),
        "plane_std_p95_over_p5": (
            float(np.percentile(positive_stds, 95) / max(np.percentile(positive_stds, 5), 1e-30))
            if positive_stds.size > 1
            else None
        ),
        "plane_mean_span_over_global_robust_range": float((np.nanmax(plane_means) - np.nanmin(plane_means)) / robust_range),
        "standardized_plane_pc1_fraction": float(eigvals[0] / eigsum) if eigsum > 0 else None,
        "standardized_plane_rank90": int(np.searchsorted(cumulative, 0.90) + 1) if eigsum > 0 else None,
        "standardized_plane_rank99": int(np.searchsorted(cumulative, 0.99) + 1) if eigsum > 0 else None,
    }


def load_volumes(selected: set[str]) -> dict[str, tuple[np.ndarray, str, float | None, Any]]:
    volumes: dict[str, tuple[np.ndarray, str, float | None, Any]] = {}

    if not selected or "e3sm_npz" in selected:
        handle = np.load(ROOT / "Data/E3SM/caesar_processed/e3sm_5vars_paperlike240_first800_caesar.npz")
        volumes["e3sm_npz"] = (handle["data"][0, 0], "time", None, handle)
    if not selected or "era5_npy" in selected:
        handle = np.load(ROOT / "Data/ERA5/finetune_processed/era5_test.npy", mmap_mode="r")
        volumes["era5_npy"] = (handle[:, 0], "heterogeneous variable", None, None)
    if not selected or "hurricane" in selected:
        path = ROOT / "Data/SDRBENCH-Hurricane-ISABEL-100x500x500/100x500x500/PRECIPf48.log10.bin.f32"
        handle = np.memmap(path, dtype=np.float32, mode="r", shape=(100, 500, 500))
        volumes["hurricane"] = (handle, "time", None, None)
    if not selected or "nyx" in selected:
        path = ROOT / "Data/SDRBENCH-EXASKY-NYX-512x512x512/SDRBENCH-EXASKY-NYX-512x512x512/baryon_density.f32"
        handle = np.memmap(path, dtype=np.float32, mode="r", shape=(512, 512, 512))
        volumes["nyx"] = (handle, "spatial z", None, None)
    if not selected or "turb_rot_npz" in selected:
        handle = np.load(ROOT / "Data/Turb_Rot_testset.npz")
        volumes["turb_rot_npz"] = (handle["data"][0, 0], "time/spatial section index", None, handle)
    if not selected or "lysozyme" in selected:
        handle = np.load(ROOT / "Data/lysozyme_processed/mmap/lysozyme_test_nf16.npy", mmap_mode="r")
        volumes["lysozyme"] = (handle[0, :32].reshape(-1, 1024, 1024)[:500], "frame across chunks", 4.294967e9, None)

    if not selected or "tomo" in selected:
        import h5py

        h5 = h5py.File(ROOT / "Data/tomo_00001.h5", "r")
        data = h5["exchange/data"][:512, 640:1152, 768:1280].astype(np.float32)
        volumes["tomo"] = (data, "projection angle", None, h5)

    if not selected or "kodak" in selected:
        from scripts.run_external_scientific_codecs import iter_kodak_image_volume_samples

        args = SimpleNamespace(data_root=str(ROOT / "Data/Kodac"), kodak_stack_images=24, max_samples=1)
        sample = next(iter_kodak_image_volume_samples(args, (512, 512)))
        volumes["kodak"] = (sample.array, "image then RGB channel", None, None)

    if not selected or "s2c" in selected:
        from scripts.run_external_scientific_codecs import iter_s2c_band_volume_samples

        s2c_root = ROOT / (
            "Data/S2C_MSIL2A_20260509T022531_N0512_R046_T51RUQ_20260509T055911/"
            "S2C_MSIL2A_20260509T022531_N0512_R046_T51RUQ_20260509T055911.SAFE"
        )
        args = SimpleNamespace(
            data_root=str(s2c_root),
            s2c_bands=["B02", "B03", "B04", "B08"],
            tile_size=1024,
            max_samples=1,
        )
        sample = next(iter_s2c_band_volume_samples(args, None))
        volumes["s2c"] = (sample.array, "spectral band", None, None)

    if not selected or "uvg_twilight_1080p" in selected:
        from compression_pipeline.adapters.uvg import UVGAdapter

        adapter = UVGAdapter(ROOT / "Data/UVG_Twilight_1080p")
        frames = [sample.array for sample in adapter.iter_samples(max_samples=30)]
        volumes["uvg_twilight_1080p"] = (np.concatenate(frames, axis=0), "time then RGB channel", None, None)
    return volumes


def main() -> None:
    args = parse_args()
    selected = set(args.datasets)
    volumes = load_volumes(selected)
    output: dict[str, Any] = {}
    for dataset_id, (array, semantics, threshold, owner) in volumes.items():
        print(f"[analyze] {dataset_id} {array.shape} {array.dtype}", flush=True)
        output[dataset_id] = summarize_volume(array, z_semantics=semantics, invalid_threshold=threshold)
        close = getattr(owner, "close", None)
        if close is not None:
            close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
