#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATASETS = [
    "e3sm_npz", "era5_npy", "hurricane", "nyx", "turb_rot_npz",
    "tomo", "lysozyme", "s2c", "kodak", "uvg_twilight_1080p",
]
DEFAULT_CUSZ_EBS = [0.5, 0.1, 0.02, 0.005, 0.001, 0.0001, 0.00001]
DEFAULT_CAESAR_EBS = [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
DEFAULT_J2K_PSNR = [20, 30, 40, 50, 60, 70, 80]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run codecs on exactly matched canonical tensors.")
    parser.add_argument("--dataset", choices=DEFAULT_DATASETS, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("unified_results/matched_validation"))
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--models", nargs="+", default=["DCAE", "HPCM", "CAESAR-V", "CAESAR-D", "cuSZ-Hi", "nvJPEG2000"])
    parser.add_argument("--image-checkpoints", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument("--cusz-eb", type=float, nargs="+", default=DEFAULT_CUSZ_EBS)
    parser.add_argument("--caesar-eb", type=float, nargs="+", default=DEFAULT_CAESAR_EBS)
    parser.add_argument("--j2k-psnr", type=float, nargs="+", default=DEFAULT_J2K_PSNR)
    parser.add_argument("--quick", action="store_true", help="Use one image checkpoint and three control points per swept codec.")
    parser.add_argument("--force", action="store_true", help="Replace existing rows for the requested model families.")
    return parser.parse_args()


def center_crop(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = array.shape[-2:]
    out_h, out_w = size
    if out_h > height or out_w > width:
        raise ValueError(f"Crop {size} exceeds {(height, width)}")
    top = (height - out_h) // 2
    left = (width - out_w) // 2
    return array[..., top : top + out_h, left : left + out_w]


def load_canonical(dataset_id: str) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    root = Path("/workspace")
    if dataset_id == "e3sm_npz":
        path = root / "Data/E3SM/caesar_processed/e3sm_5vars_paperlike240_first800_caesar.npz"
        with np.load(path) as handle:
            array = np.asarray(handle["data"][:3, 0, :16], dtype=np.float32)
        return array, None, {"source": str(path), "selection": "V=0:3,S=0,T=0:16", "axis": "variable,time,y,x"}

    if dataset_id == "era5_npy":
        path = root / "Data/ERA5/finetune_processed/era5_test.npy"
        channel_indices = [0, 74, 259]  # geopotential, u-wind, and first single-level variable
        data = np.load(path, mmap_mode="r")
        array = center_crop(np.asarray(data[channel_indices, :16], dtype=np.float32), (240, 240))
        return array, None, {
            "source": str(path), "selection": "channels=[0,74,259],T=0:16,center_crop=240x240",
            "channel_indices": channel_indices, "axis": "variable,time,y,x", "source_normalization": "CRA5 z-score",
        }

    if dataset_id == "hurricane":
        path = root / "Data/SDRBENCH-Hurricane-ISABEL-100x500x500/100x500x500/PRECIPf48.log10.bin.f32"
        data = np.memmap(path, dtype=np.float32, mode="r", shape=(100, 500, 500))
        return np.asarray(data[:16], dtype=np.float32)[None], None, {
            "source": str(path), "selection": "T=0:16", "field": "log10 precipitation", "axis": "variable,time,y,x",
        }

    if dataset_id == "nyx":
        path = root / "Data/SDRBENCH-EXASKY-NYX-512x512x512/SDRBENCH-EXASKY-NYX-512x512x512/baryon_density.f32"
        data = np.memmap(path, dtype=np.float32, mode="r", shape=(512, 512, 512))
        return np.asarray(data[:16], dtype=np.float32)[None], None, {
            "source": str(path), "selection": "Z=0:16", "field": "baryon_density", "axis": "variable,z,y,x",
        }

    if dataset_id == "turb_rot_npz":
        path = root / "Turb_Rot_testset.npz"
        with np.load(path) as handle:
            array = np.asarray(handle["data"][0, 0, :16], dtype=np.float32)[None]
        return array, None, {"source": str(path), "selection": "V=0,S=0,T=0:16", "axis": "variable,time,y,x"}

    if dataset_id == "tomo":
        import h5py

        path = root / "Data/tomo_00001.h5"
        with h5py.File(path, "r") as handle:
            array = center_crop(handle["exchange/data"][:16].astype(np.float32), (512, 512))[None]
        return array, None, {"source": str(path), "selection": "projection=0:16,center_crop=512x512", "axis": "variable,angle,y,x"}

    if dataset_id == "lysozyme":
        path = root / "Data/lysozyme_processed/mmap/lysozyme_test_nf16.npy"
        data = np.load(path, mmap_mode="r")
        raw = np.asarray(data[0, 0], dtype=np.float32)[None]
        mask = raw < 4.294967e9
        array = np.array(raw, copy=True)
        array[~mask] = 0.0
        return array, mask, {
            "source": str(path), "selection": "test chunk=0,frames=0:16", "axis": "variable,time,y,x",
            "invalid_policy": "raw>=4.294967e9 replaced by zero and excluded from metrics",
        }

    if dataset_id == "s2c":
        from scripts.run_external_scientific_codecs import iter_s2c_band_volume_samples

        path = root / (
            "Data/S2C_MSIL2A_20260509T022531_N0512_R046_T51RUQ_20260509T055911/"
            "S2C_MSIL2A_20260509T022531_N0512_R046_T51RUQ_20260509T055911.SAFE"
        )
        sample = next(iter_s2c_band_volume_samples(SimpleNamespace(
            data_root=str(path), s2c_bands=["B02", "B03", "B04", "B08"], tile_size=1024, max_samples=1,
        ), None))
        return np.asarray(sample.array, dtype=np.float32)[None], None, {
            "source": str(path), "selection": sample.sample_id, "axis": "variable,spectral_band,y,x",
            "caesar_compatible": False,
        }

    if dataset_id == "kodak":
        from compression_pipeline.adapters.kodak import KodakAdapter

        path = root / "Data/Kodac"
        sequence, _ = KodakAdapter(path).load_sequence(max_samples=16, resolution=(512, 512))
        return np.asarray(sequence, dtype=np.float32), None, {
            "source": str(path), "selection": "images=0:16,center_crop=512x512", "axis": "rgb,time,y,x",
            "image_grouping": "RGB per frame",
        }

    if dataset_id == "uvg_twilight_1080p":
        from compression_pipeline.adapters.uvg import UVGAdapter

        path = root / "Data/UVG_Twilight_1080p"
        sequence, _ = UVGAdapter(path).load_sequence(max_samples=16)
        return np.asarray(sequence, dtype=np.float32), None, {
            "source": str(path), "selection": "frames=0:16,1920x1080", "axis": "rgb,time,y,x",
            "image_grouping": "RGB per frame",
        }
    raise ValueError(dataset_id)


def checksum(array: np.ndarray, mask: np.ndarray | None) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(array).view(np.uint8))
    if mask is not None:
        digest.update(np.ascontiguousarray(mask).view(np.uint8))
    return digest.hexdigest()


def aggregate_rows(
    rows: list[dict[str, Any]],
    canonical: np.ndarray,
    mask: np.ndarray | None,
    fields: dict[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("No rows to aggregate")
    valid_count = int(np.count_nonzero(mask)) if mask is not None else int(canonical.size)
    values = canonical[mask] if mask is not None else canonical
    data_range = max(float(np.max(values) - np.min(values)), 1e-8)
    weighted_sse = sum(float(row["mse"]) * int(row.get("valid_voxel_count", row.get("voxel_count", 0))) for row in rows)
    if not weighted_sse and any(float(row.get("mse", 0.0)) > 0 for row in rows):
        weighted_sse = sum(float(row["mse"]) * int(row.get("symbol_count", 0)) for row in rows)
    mse = weighted_sse / valid_count
    bitstream = sum(float(row.get("bitstream_bytes", 0)) for row in rows)
    side_info = sum(float(row.get("side_info_bytes", 0)) for row in rows)
    total_bytes = bitstream + side_info
    original_bytes = int(canonical.size * canonical.dtype.itemsize)
    quality_key = "average_variable_psnr" if fields.get("model_name") == "CAESAR" else "psnr"
    per_variable = [
        float(row[quality_key]) for row in rows
        if isinstance(row.get(quality_key), (int, float)) and math.isfinite(float(row[quality_key]))
    ]
    wall = sum(float(row.get("sample_wall_time_total", 0)) for row in rows)
    result = dict(fields)
    result.update({
        "mse": mse,
        "rmse": math.sqrt(mse),
        "psnr": float("inf") if mse < 1e-30 else 10 * math.log10(data_range * data_range / mse),
        "average_variable_psnr": float(np.mean(per_variable)) if per_variable else None,
        "per_partition_psnr": per_variable,
        "bitstream_bytes": bitstream,
        "side_info_bytes": side_info,
        "total_bytes_with_side_info": total_bytes,
        "bpp": bitstream * 8 / canonical.size,
        "scientific_bpp": bitstream * 8 / canonical.size,
        "scientific_bpp_with_side_info": total_bytes * 8 / canonical.size,
        "compression_ratio": original_bytes / total_bytes if total_bytes else float("inf"),
        "original_bytes": original_bytes,
        "voxel_count": int(canonical.size),
        "valid_voxel_count": valid_count,
        "encode_time_avg": sum(float(row.get("encode_time_avg", 0)) for row in rows),
        "decode_time_avg": sum(float(row.get("decode_time_avg", 0)) for row in rows),
        "sample_wall_time_total": wall,
        "sample_wall_throughput_MBps": original_bytes / wall / 1e6 if wall > 0 else None,
        "partition_count": len(rows),
        "error_bound_satisfied": (
            all(row.get("error_bound_satisfied") is True for row in rows)
            if any("error_bound_satisfied" in row for row in rows)
            else None
        ),
        "max_abs_error": max((float(row.get("max_abs_error", 0.0)) for row in rows), default=None),
        "requested_abs_eb_by_partition": [row.get("requested_abs_eb") for row in rows if row.get("requested_abs_eb") is not None],
    })
    return result


def row_weight(row: dict[str, Any], symbol_count: int, valid_count: int) -> dict[str, Any]:
    out = dict(row)
    out["symbol_count"] = int(symbol_count)
    out["voxel_count"] = int(symbol_count)
    out["valid_voxel_count"] = int(valid_count)
    return out


def image_partitions(canonical: np.ndarray, mask: np.ndarray | None, general_rgb: bool):
    from compression_pipeline.canonical import CanonicalSample

    if general_rgb:
        for time_index in range(canonical.shape[1]):
            part_mask = mask[:, time_index] if mask is not None else None
            yield CanonicalSample("matched", f"frame{time_index:03d}", "image", canonical[:, time_index], "channel_height_width", {"dtype": str(canonical.dtype)}), part_mask
    else:
        for variable in range(canonical.shape[0]):
            part_mask = mask[variable] if mask is not None else None
            yield CanonicalSample("matched", f"variable{variable:03d}", "scientific_field", canonical[variable], "channel_height_width", {"dtype": str(canonical.dtype)}), part_mask


def run_image_models(args, canonical, mask, manifest, output_dir, append: Callable[[dict], None]) -> None:
    import torch
    from compression_pipeline.metrics import torch_memory_usage_mb
    from compression_pipeline.model_registry import image_model_jobs
    from compression_pipeline.runner import run_image_grouped_sample
    from compression_pipeline.torch_codecs import CompressAILikeCodec

    requested = set(args.models)
    jobs = list(image_model_jobs(PROJECT_ROOT, {"DCAE", "LIC-HPCM"}))
    selected = []
    for family, predicate in [
        ("DCAE", lambda job: job.model_name == "DCAE"),
        ("HPCM", lambda job: job.model_name == "LIC-HPCM" and "-base_" in job.model_id),
    ]:
        if family not in requested:
            continue
        group = [job for job in jobs if predicate(job)]
        indices = [3] if args.quick else args.image_checkpoints
        selected.extend(group[index - 1] for index in indices if 1 <= index <= len(group))

    general_rgb = args.dataset in {"kodak", "uvg_twilight_1080p"}
    for job in selected:
        print(f"[image-load] {job.model_id}", flush=True)
        model = job.loader("cuda")
        codec_cls = job.codec_cls or CompressAILikeCodec
        codec = codec_cls(model, device="cuda", divisor=job.divisor, **job.codec_kwargs)
        rows = []
        try:
            for sample, part_mask in image_partitions(canonical, mask, general_rgb):
                result = run_image_grouped_sample(sample, codec, memory_fn=lambda: torch_memory_usage_mb("cuda"), valid_mask=part_mask)
                rows.append(row_weight(result, sample.array.size, int(np.count_nonzero(part_mask)) if part_mask is not None else sample.array.size))
            append(aggregate_rows(rows, canonical, mask, {
                **manifest, "model_name": job.model_name, "model_id": job.model_id,
                "control": Path(job.checkpoint).name if job.checkpoint else job.model_id,
                "partition_policy": "RGB frame" if general_rgb else "time planes within variable",
            }))
        finally:
            del codec, model
            gc.collect()
            torch.cuda.empty_cache()


def run_cusz(args, canonical, mask, manifest, output_dir, append: Callable[[dict], None]) -> None:
    if "cuSZ-Hi" not in args.models:
        return
    from compression_pipeline.canonical import CanonicalSample
    from scripts.run_external_scientific_codecs import run_cuszhi_stack_sample

    controls = [0.1, 0.005, 0.0001] if args.quick else args.cusz_eb
    codec_args = SimpleNamespace(
        cuszhi=str(PROJECT_ROOT / "models/cuSZ-Hi/build/cuszhi"), cuszhi_scheme="huffman",
        cuszhi_predictor="lorenzo", cuszhi_min_abs_eb=1e-20, cuszhi_eb_reference="range",
        cuszhi_robust_low=0.1, cuszhi_robust_high=99.9, lpips_fn=None,
    )
    for eb in controls:
        rows = []
        try:
            for variable in range(canonical.shape[0]):
                arr = np.ascontiguousarray(canonical[variable], dtype=np.float32)
                part_mask = mask[variable] if mask is not None else None
                sample = CanonicalSample("matched", f"variable{variable:03d}", "scientific_field", arr, "channel_height_width", {})
                result = run_cuszhi_stack_sample(
                    sample, codec_args, float(eb), output_dir, arr, Path(codec_args.cuszhi), requested_mode="whole3d",
                    valid_mask=part_mask,
                )
                rows.append(row_weight(result, arr.size, int(np.count_nonzero(part_mask)) if part_mask is not None else arr.size))
        except Exception as exc:
            append({
                **manifest, "model_name": "cuSZ-Hi", "model_id": f"cuSZ-Hi-matched-eb{eb:g}",
                "control": float(eb), "eb": float(eb), "error": str(exc),
                "partition_policy": "one T,H,W volume per variable",
            })
            continue
        append(aggregate_rows(rows, canonical, mask, {
            **manifest, "model_name": "cuSZ-Hi", "model_id": f"cuSZ-Hi-matched-eb{eb:g}", "control": float(eb),
            "eb": float(eb), "partition_policy": "one T,H,W volume per variable",
        }))


def run_j2k(args, canonical, mask, manifest, output_dir, append: Callable[[dict], None]) -> None:
    if "nvJPEG2000" not in args.models:
        return
    from compression_pipeline.canonical import CanonicalSample
    from compression_pipeline.nvjpeg_codecs import run_nvjpeg2k_sample

    controls = [30, 50, 70] if args.quick else args.j2k_psnr
    for target in controls:
        rows = []
        for variable in range(canonical.shape[0]):
            arr = np.ascontiguousarray(canonical[variable], dtype=np.float32)
            part_mask = mask[variable] if mask is not None else None
            sample = CanonicalSample("matched", f"variable{variable:03d}", "scientific_field", arr, "channel_height_width", {})
            result = run_nvjpeg2k_sample(sample, float(target), output_dir, valid_mask=part_mask)
            rows.append(row_weight(result, arr.size, int(np.count_nonzero(part_mask)) if part_mask is not None else arr.size))
        append(aggregate_rows(rows, canonical, mask, {
            **manifest, "model_name": "nvJPEG2000", "model_id": f"nvJPEG2000-matched-psnr{target:g}",
            "control": float(target), "target_jpeg2000_psnr": float(target),
            "partition_policy": "one T,H,W stack per variable; per-plane uint16 normalization",
        }))


def run_caesar(args, canonical, mask, manifest, output_dir, append: Callable[[dict], None]) -> None:
    from compression_pipeline.caesar_runner import run_caesar_sequence

    if canonical.shape[1] < 16 or not manifest.get("caesar_compatible", True):
        print(f"[caesar-skip] T={canonical.shape[1]} compatible={manifest.get('caesar_compatible', True)}", flush=True)
        return
    controls = [0.03, 0.003, 0.0003] if args.quick else args.caesar_eb
    timestamps = [f"2024-01-{index + 1:02d}T00:00:00" for index in range(canonical.shape[1])]
    for cli_name, model_name, starts in [("CAESAR-V", "caesar_v", [0, 8]), ("CAESAR-D", "caesar_d", [0])]:
        if cli_name not in args.models:
            continue
        for eb in controls:
            rows = []
            try:
                for start in starts:
                    result = run_caesar_sequence(
                        canonical, timestamps, model_name, PROJECT_ROOT / "models/CAESAR", PROJECT_ROOT / "checkpoints/caesar",
                        output_dir, "cuda", batch_size=8, eb=float(eb), start_index=start,
                        sample_id=f"{args.dataset}_{model_name}_t{start:02d}", collect_lpips=False,
                        valid_mask_vthw=mask,
                    )
                    symbols = canonical[:, start : start + (8 if model_name == "caesar_v" else 16)].size
                    valid = int(np.count_nonzero(mask[:, start : start + (8 if model_name == "caesar_v" else 16)])) if mask is not None else symbols
                    rows.append(row_weight(result, symbols, valid))
            except Exception as exc:
                append({
                    **manifest, "model_name": "CAESAR", "model_id": f"{model_name}-matched-eb{eb:g}",
                    "control": float(eb), "eb": float(eb), "error": str(exc),
                    "partition_policy": "two 8-frame windows" if starts == [0, 8] else "one 16-frame window",
                })
                continue
            append(aggregate_rows(rows, canonical, mask, {
                **manifest, "model_name": "CAESAR", "model_id": f"{model_name}-matched-eb{eb:g}",
                "control": float(eb), "eb": float(eb), "partition_policy": "two 8-frame windows" if starts == [0, 8] else "one 16-frame window",
            }))


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(f"CUDA unavailable with CUDA_VISIBLE_DEVICES={args.gpu}")
    output_dir = args.output_root / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical, mask, selection = load_canonical(args.dataset)
    canonical = np.ascontiguousarray(canonical, dtype=np.float32)
    mask = np.ascontiguousarray(mask, dtype=bool) if mask is not None else None
    canonical_hash = checksum(canonical, mask)
    manifest = {
        "dataset_id": args.dataset,
        "canonical_shape": list(canonical.shape),
        "canonical_dtype": str(canonical.dtype),
        "canonical_sha256": canonical_hash,
        "canonical_valid_voxels": int(np.count_nonzero(mask)) if mask is not None else int(canonical.size),
        **selection,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary_path = output_dir / "summary.json"
    rows = json.loads(summary_path.read_text()) if summary_path.exists() else []
    if args.force:
        requested_names = set()
        if "DCAE" in args.models:
            requested_names.add("DCAE")
        if "HPCM" in args.models:
            requested_names.add("LIC-HPCM")
        if "cuSZ-Hi" in args.models:
            requested_names.add("cuSZ-Hi")
        if "nvJPEG2000" in args.models:
            requested_names.add("nvJPEG2000")
        if "CAESAR-V" in args.models or "CAESAR-D" in args.models:
            requested_names.add("CAESAR")
        rows = [row for row in rows if row.get("model_name") not in requested_names]
        summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    done = {(row.get("model_id"), str(row.get("control"))) for row in rows if "error" not in row}

    def append(row: dict[str, Any]) -> None:
        key = (row.get("model_id"), str(row.get("control")))
        if key in done:
            print(f"[skip-existing] {key}", flush=True)
            return
        row["gpu_physical_index"] = str(args.gpu)
        row["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rows.append(row)
        done.add(key)
        summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        if "error" in row:
            print(f"[error] {row.get('model_id')}: {row['error'][-500:]}", flush=True)
            return
        print(
            f"[result] {row['model_id']} bpp={row['scientific_bpp_with_side_info']:.5g} "
            f"PSNR={row['psnr']:.3f} avg-var={row.get('average_variable_psnr')} "
            f"wall={row.get('sample_wall_throughput_MBps'):.3f} MB/s",
            flush=True,
        )

    print(json.dumps(manifest, indent=2), flush=True)
    run_image_models(args, canonical, mask, manifest, output_dir, append)
    run_cusz(args, canonical, mask, manifest, output_dir, append)
    run_j2k(args, canonical, mask, manifest, output_dir, append)
    run_caesar(args, canonical, mask, manifest, output_dir, append)
    print(summary_path, flush=True)


if __name__ == "__main__":
    main()
