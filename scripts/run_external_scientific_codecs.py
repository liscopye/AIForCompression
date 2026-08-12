#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterator

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compression_pipeline.adapters.e3sm_npz import E3SMNPZAdapter
from compression_pipeline.adapters.era5 import ERA5Adapter
from compression_pipeline.adapters.hurricane import HurricaneAdapter
from compression_pipeline.adapters.isotropic1024 import Isotropic1024Adapter
from compression_pipeline.adapters.kodak import KodakAdapter
from compression_pipeline.adapters.lysozyme import LysozymeAdapter
from compression_pipeline.adapters.nyx import NYXAdapter
from compression_pipeline.adapters.s2c import S2CAdapter
from compression_pipeline.adapters.shanghai_xray import ShanghaiXrayAdapter
from compression_pipeline.adapters.tomo_h5 import TomoH5Adapter
from compression_pipeline.adapters.turb_rot_npz import TurbRotNPZAdapter
from compression_pipeline.canonical import CanonicalSample
from compression_pipeline.metrics import base_metrics, make_lpips_fn, process_memory_usage_mb
from compression_pipeline.gpu_memory import run_with_gpu_peak
from compression_pipeline.runner import _normalization_side_info_bytes
from compression_pipeline.views import build_image_groups, reconstruct_from_groups


DEFAULT_CUSZHI = PROJECT_ROOT / "models" / "cuSZ-Hi" / "build" / "cuszhi"
DEFAULT_VISMS = PROJECT_ROOT / "models" / "visemz" / "test" / "build" / "mscomp"
DEFAULT_VISEMZ_ANALYSIS = PROJECT_ROOT / "models" / "visemz" / "bmshj2018_factorized_a.pt"
DEFAULT_VISEMZ_SYNTHESIS = PROJECT_ROOT / "models" / "visemz" / "bmshj2018_factorized_s.pt"
DEFAULT_TORCH_LIB = Path("/workspace/ai4cp/lib/python3.12/site-packages/torch/lib")
DEFAULT_NVTX_LIB = Path("/workspace/ai4cp/lib/python3.12/site-packages/nvidia/nvtx/lib")
DEFAULT_GRAPHCOMP_LIB_SZ = PROJECT_ROOT / "models" / "GraphComp" / "sz3_wrapper" / "build" / "lib_sz.so"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external scientific codecs on canonical benchmark samples.")
    parser.add_argument(
        "--dataset",
        choices=[
            "turb_rot_npz",
            "e3sm_npz",
            "era5_npy",
            "era5",
            "kodak",
            "tomo",
            "hurricane",
            "s2c",
            "nyx",
            "shanghai_xray",
            "isot1024",
            "lysozyme",
        ],
        required=True,
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--models", nargs="+", default=["cuSZ-Hi-3D", "visemz", "GraphComp"])
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--npz_image_mode", choices=["auto", "variables", "sections"], default="auto")
    parser.add_argument("--npz_image_channels", type=int, default=3)
    parser.add_argument("--npz_variable_index", type=int, default=0)
    parser.add_argument("--npz_time_start", type=int, default=0)
    parser.add_argument("--section_index", type=int, default=0)
    parser.add_argument("--section_start", type=int, default=0)
    parser.add_argument("--era5_max_channels", type=int, default=3)
    parser.add_argument("--era5_time_start", type=int, default=0)
    parser.add_argument(
        "--era5_npy_3d_mode",
        choices=["channels", "time"],
        default="channels",
        help="For era5_npy samples, use different channels at one time or consecutive times from one channel.",
    )
    parser.add_argument("--era5_npy_variable_index", type=int, default=0)
    parser.add_argument("--tomo_group_frames", type=int, default=3)
    parser.add_argument(
        "--cuszhi_pack_z",
        action="store_true",
        help="Build cuSZ-Hi samples as thick [Z,H,W] volumes using dataset-specific z/time/band/image stacking.",
    )
    parser.add_argument(
        "--cuszhi_z_depth",
        type=int,
        default=0,
        help="Z depth for --cuszhi_pack_z. 0 uses the deepest block allowed by the data and voxel budget.",
    )
    parser.add_argument("--cuszhi_z_stride", type=int, default=0, help="Stride between packed-z samples. 0 uses z_depth.")
    parser.add_argument(
        "--cuszhi_max_voxels",
        type=int,
        default=150_000_000,
        help="Maximum voxels per packed cuSZ-Hi sample when --cuszhi_z_depth=0.",
    )
    parser.add_argument(
        "--s2c_bands",
        nargs="+",
        default=["B02", "B03", "B04", "B08"],
        help="Sentinel-2 bands to stack for --cuszhi_pack_z; defaults to 10m bands.",
    )
    parser.add_argument(
        "--kodak_stack_images",
        type=int,
        default=0,
        help="Number of Kodak images to stack for --cuszhi_pack_z. 0 uses all selected images.",
    )
    parser.add_argument("--tile_size", type=int, default=None)
    parser.add_argument("--max_channels", type=int, default=3)
    parser.add_argument("--eb", type=float, nargs="+", default=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    parser.add_argument("--cuszhi", default=str(DEFAULT_CUSZHI))
    parser.add_argument("--cuszhi_scheme", choices=["cr", "tp"], default="cr")
    parser.add_argument("--cuszhi_predictor", default="spline3")
    parser.add_argument("--cuszhi_min_abs_eb", type=float, default=3e-6)
    parser.add_argument("--cuszhi_eb_reference", choices=["range", "robust"], default="range")
    parser.add_argument("--cuszhi_robust_low", type=float, default=0.1)
    parser.add_argument("--cuszhi_robust_high", type=float, default=99.9)
    parser.add_argument(
        "--lysozyme_invalid_policy",
        choices=["zero", "median", "raw"],
        default="zero",
        help=(
            "How to handle Lysozyme uint32 sentinel pixels before compression. "
            "zero/median replace invalid pixels and evaluate only raw != 2^32-1 positions; raw disables masking."
        ),
    )
    parser.add_argument(
        "--lysozyme_invalid_threshold",
        type=float,
        default=4.294967e9,
        help="Values at or above this threshold are treated as Lysozyme detector sentinels.",
    )
    parser.add_argument(
        "--cuszhi_sample_mode",
        choices=["stack", "whole3d"],
        default="whole3d",
        help="cuSZ-Hi always compresses the complete [Z,H,W] sample as one 3D volume; stack is an alias.",
    )
    parser.add_argument("--visemz_bin", default=str(DEFAULT_VISMS))
    parser.add_argument("--visemz_analysis", default=str(DEFAULT_VISEMZ_ANALYSIS))
    parser.add_argument("--visemz_synthesis", default=str(DEFAULT_VISEMZ_SYNTHESIS))
    parser.add_argument("--graphcomp_lib_sz", default=str(DEFAULT_GRAPHCOMP_LIB_SZ))
    parser.add_argument("--graphcomp_scale", type=float, default=10.0)
    parser.add_argument("--graphcomp_sigma", type=float, default=1.0)
    parser.add_argument("--graphcomp_min_size", type=int, default=1)
    parser.add_argument("--gpu", default=None, help="Optional CUDA_VISIBLE_DEVICES override for LPIPS.")
    parser.add_argument("--no_lpips", action="store_true", help="Disable optional LPIPS calculation.")
    parser.add_argument("--verbose_records", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.lpips_fn = None if args.no_lpips else make_lpips_fn("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary = _load_summary(summary_path)
    done = {
        (r.get("model_id"), r.get("sample_id"), r.get("eb"))
        for r in summary
        if "error" not in r
    }

    samples = list(iter_samples(args))
    if not samples:
        raise SystemExit(f"No samples found for {args.dataset}: {args.data_root}")

    for requested in args.models:
        model, model_id_name, cuszhi_mode_override = normalize_model_request(requested)
        if model == "cuSZ-Hi":
            model_id_name = "cuSZ-Hi-3D"
        for eb in args.eb:
            for sample in samples:
                model_id = f"{model_id_name}_eb{format_eb(eb)}"
                key = (model_id, sample.sample_id, eb)
                if key in done:
                    print(f"[skip] {model_id} {sample.sample_id}", flush=True)
                    continue
                try:
                    print(f"[run] {model_id} {sample.sample_id}", flush=True)
                    if model == "cuSZ-Hi":
                        result = run_cuszhi_sample(sample, args, eb, output_dir, cuszhi_mode_override)
                    elif model == "visemz":
                        result = run_visemz_sample(sample, args, eb, output_dir)
                    elif model == "GraphComp":
                        result = run_graphcomp_sample(sample, args, eb, output_dir)
                    else:
                        raise ValueError(f"Unsupported model: {requested}")
                    result.update(
                        {
                            "model_name": model,
                            "model_id": model_id,
                            "label": model_id_name,
                            "metric": "error_bound",
                            "eb": eb,
                            "dataset_id": sample.dataset_id,
                            "sample_id": sample.sample_id,
                            "sample_kind": sample.kind,
                            "shape": list(sample.array.shape),
                        }
                    )
                    summary.append(result)
                    done.add(key)
                    if args.verbose_records:
                        print(json.dumps(result, indent=2), flush=True)
                    else:
                        print(
                            f"[ok] {model_id} {sample.sample_id} "
                            f"bpp={result.get('scientific_bpp_with_side_info', result.get('bpp')):.4g} "
                            f"psnr={result.get('average_frame_psnr', result.get('psnr')):.4g}",
                            flush=True,
                        )
                except Exception as exc:
                    summary.append(
                        {
                            "model_name": model,
                            "model_id": model_id,
                            "metric": "error_bound",
                            "eb": eb,
                            "dataset_id": sample.dataset_id,
                            "sample_id": sample.sample_id,
                            "error": str(exc),
                        }
                    )
                    print(f"[error] {model_id} {sample.sample_id}: {exc}", flush=True)
                finally:
                    write_summary(summary_path, summary)
    print(f"Results: {summary_path}", flush=True)


def iter_samples(args: argparse.Namespace) -> Iterator[CanonicalSample]:
    resolution = tuple(args.resolution) if args.resolution else None
    if args.cuszhi_pack_z:
        yield from iter_cuszhi_packed_z_samples(args, resolution)
        return
    if args.dataset == "kodak":
        yield from KodakAdapter(args.data_root).iter_samples(max_samples=args.max_samples)
        return
    if args.dataset == "tomo":
        yield from TomoH5Adapter(args.data_root, group_frames=args.tomo_group_frames).iter_samples(
            max_samples=args.max_samples,
            resolution=resolution,
        )
        return
    if args.dataset == "hurricane":
        yield from HurricaneAdapter(args.data_root).iter_samples(max_samples=args.max_samples)
        return
    if args.dataset == "s2c":
        yield from S2CAdapter(args.data_root, tile_size=args.tile_size).iter_samples(max_samples=args.max_samples)
        return
    if args.dataset == "nyx":
        yield from NYXAdapter(args.data_root).iter_samples(max_samples=args.max_samples)
        return
    if args.dataset == "shanghai_xray":
        yield from ShanghaiXrayAdapter(args.data_root).iter_samples(max_samples=args.max_samples)
        return
    if args.dataset == "isot1024":
        yield from Isotropic1024Adapter(args.data_root).iter_samples(max_samples=args.max_samples)
        return
    if args.dataset == "lysozyme":
        yield from LysozymeAdapter(args.data_root).iter_samples(max_samples=args.max_samples)
        return
    if args.dataset == "era5":
        yield from ERA5Adapter(args.data_root).iter_samples(
            max_samples=args.max_samples,
            max_channels=args.max_channels,
            resolution=resolution,
        )
        return
    if args.dataset == "turb_rot_npz":
        yield from TurbRotNPZAdapter(
            args.data_root,
            section_index=args.section_index,
            section_start=args.section_start,
            time_start=args.npz_time_start,
            image_group_mode=args.npz_image_mode,
            image_channel_count=args.npz_image_channels,
        ).iter_samples(max_samples=args.max_samples)
        return
    if args.dataset == "e3sm_npz":
        adapter = E3SMNPZAdapter(
            args.data_root,
            section_index=args.section_index,
            section_start=args.section_start,
            time_start=args.npz_time_start,
            image_group_mode=args.npz_image_mode,
            image_channel_count=args.npz_image_channels,
        )
        for sample in adapter.iter_samples(max_samples=args.max_samples):
            yield center_crop_sample(sample, resolution)
        return
    yield from iter_era5_npy_samples(
        args.data_root,
        max_samples=args.max_samples,
        max_channels=args.era5_max_channels,
        time_start=args.era5_time_start,
        sample_mode=args.era5_npy_3d_mode,
        variable_index=args.era5_npy_variable_index,
        resolution=resolution,
    )


def iter_cuszhi_packed_z_samples(
    args: argparse.Namespace,
    resolution: tuple[int, int] | None,
) -> Iterator[CanonicalSample]:
    if args.dataset in {"e3sm_npz", "turb_rot_npz"}:
        yield from iter_npz_time_volume_samples(args, resolution)
        return
    if args.dataset == "era5_npy":
        yield from iter_era5_npy_time_volume_samples(args, resolution)
        return
    if args.dataset == "hurricane":
        yield from iter_hurricane_time_volume_samples(args, resolution)
        return
    if args.dataset == "nyx":
        yield from iter_nyx_z_volume_samples(args, resolution)
        return
    if args.dataset == "tomo":
        yield from iter_tomo_z_volume_samples(args, resolution)
        return
    if args.dataset == "lysozyme":
        yield from iter_lysozyme_frame_volume_samples(args, resolution)
        return
    if args.dataset == "s2c":
        yield from iter_s2c_band_volume_samples(args, resolution)
        return
    if args.dataset == "kodak":
        yield from iter_kodak_image_volume_samples(args, resolution)
        return
    raise ValueError(f"--cuszhi_pack_z is not implemented for dataset={args.dataset}")


def _depth_from_budget(
    available: int,
    height: int,
    width: int,
    requested: int,
    max_voxels: int,
) -> int:
    if requested > 0:
        return min(int(requested), int(available))
    if max_voxels <= 0:
        return int(available)
    budget_depth = max(1, int(max_voxels) // max(1, int(height) * int(width)))
    return min(int(available), budget_depth)


def _block_starts(available: int, depth: int, max_samples: int, stride: int) -> list[int]:
    if depth <= 0 or available < depth:
        return []
    step = int(stride) if stride > 0 else depth
    starts = list(range(0, available - depth + 1, step))
    if max_samples > 0:
        starts = starts[:max_samples]
    return starts


def _crop_zhw(arr: np.ndarray, resolution: tuple[int, int] | None) -> np.ndarray:
    if resolution is None:
        return arr
    from compression_pipeline.adapters.era5 import center_crop_chw

    return center_crop_chw(arr, resolution)


def iter_npz_time_volume_samples(args: argparse.Namespace, resolution: tuple[int, int] | None) -> Iterator[CanonicalSample]:
    path = Path(args.data_root)
    handle = np.load(path, allow_pickle=False)
    try:
        data = handle["data"]
        if data.ndim != 5:
            raise ValueError(f"{args.dataset} packed-z expects [V,S,T,H,W], got {data.shape}")
        v = int(args.npz_variable_index)
        s = int(args.section_index)
        if v < 0 or v >= data.shape[0]:
            raise ValueError(f"npz_variable_index {v} out of range for V={data.shape[0]}")
        if s < 0 or s >= data.shape[1]:
            raise ValueError(f"section_index {s} out of range for S={data.shape[1]}")
        time_start = int(args.npz_time_start)
        available = data.shape[2] - time_start
        height, width = int(data.shape[3]), int(data.shape[4])
        depth = _depth_from_budget(available, height, width, args.cuszhi_z_depth, args.cuszhi_max_voxels)
        for start_offset in _block_starts(available, depth, args.max_samples, args.cuszhi_z_stride):
            t0 = time_start + start_offset
            arr = np.asarray(data[v, s, t0 : t0 + depth], dtype=np.float32)
            arr = _crop_zhw(arr, resolution)
            yield CanonicalSample(
                dataset_id=args.dataset,
                sample_id=f"var{v:03d}_section{s:03d}_time{t0:04d}-{t0 + depth - 1:04d}",
                kind="scientific_field",
                array=arr,
                layout="channel_height_width",
                metadata={
                    "source_path": str(path),
                    "source_layout": "V,S,T,H,W",
                    "cuszhi_pack_z": True,
                    "z_axis": "time",
                    "variable_index": v,
                    "section_index": s,
                    "time_range": [int(t0), int(t0 + depth)],
                    "channels": int(arr.shape[0]),
                    "height": int(arr.shape[1]),
                    "width": int(arr.shape[2]),
                    "dtype": "float32",
                },
            )
    finally:
        handle.close()


def iter_era5_npy_time_volume_samples(args: argparse.Namespace, resolution: tuple[int, int] | None) -> Iterator[CanonicalSample]:
    path = Path(args.data_root)
    data = np.load(path, mmap_mode="r")
    if data.ndim != 4:
        raise ValueError(f"ERA5 npy must be [C,T,H,W], got {data.shape}")
    time_count = data.shape[1]
    t = int(args.era5_time_start)
    if t < 0 or t >= time_count:
        raise ValueError(f"era5_time_start {t} out of range for T={time_count}")
    available = int(data.shape[0])
    depth = min(int(args.cuszhi_z_depth), available) if args.cuszhi_z_depth > 0 else available
    for start in _block_starts(available, depth, args.max_samples, args.cuszhi_z_stride):
        arr = np.asarray(data[start : start + depth, t], dtype=np.float32)
        arr = _crop_zhw(arr, resolution)
        yield CanonicalSample(
            dataset_id="era5_npy",
            sample_id=f"vars{start:03d}-{start + depth - 1:03d}_time{t:04d}",
            kind="scientific_field",
            array=arr,
            layout="channel_height_width",
            metadata={
                "source_path": str(path),
                "source_layout": "C,T,H,W",
                "cuszhi_pack_z": True,
                "z_axis": "variable",
                "variable_range": [int(start), int(start + depth)],
                "time_index": int(t),
                "channels": int(arr.shape[0]),
                "height": int(arr.shape[1]),
                "width": int(arr.shape[2]),
                "dtype": "float32",
            },
        )


def _infer_hurricane_shape(total: int) -> tuple[int, int, int]:
    if total == 100 * 500 * 500:
        return 100, 500, 500
    if total == 500 * 500 * 100:
        return 500, 500, 100
    raise ValueError(f"Cannot infer Hurricane shape for {total} elements")


def iter_hurricane_time_volume_samples(args: argparse.Namespace, resolution: tuple[int, int] | None) -> Iterator[CanonicalSample]:
    adapter = HurricaneAdapter(args.data_root)
    path = adapter._find_file()
    data = np.memmap(path, dtype=np.float32, mode="r")
    t, h, w = _infer_hurricane_shape(data.size)
    volume = data.reshape(t, h, w)
    depth = _depth_from_budget(t, h, w, args.cuszhi_z_depth, args.cuszhi_max_voxels)
    for start in _block_starts(t, depth, args.max_samples, args.cuszhi_z_stride):
        arr = np.asarray(volume[start : start + depth], dtype=np.float32)
        arr = _crop_zhw(arr, resolution)
        yield CanonicalSample(
            dataset_id="hurricane",
            sample_id=f"time{start:03d}-{start + depth - 1:03d}",
            kind="scientific_field",
            array=arr,
            layout="channel_height_width",
            metadata={
                "source_path": str(path),
                "source_layout": "T,H,W",
                "cuszhi_pack_z": True,
                "z_axis": "time",
                "time_range": [int(start), int(start + depth)],
                "channels": int(arr.shape[0]),
                "height": int(arr.shape[1]),
                "width": int(arr.shape[2]),
                "dtype": "float32",
            },
        )


def iter_nyx_z_volume_samples(args: argparse.Namespace, resolution: tuple[int, int] | None) -> Iterator[CanonicalSample]:
    adapter = NYXAdapter(args.data_root)
    path = adapter._find_file()
    d, h, w = 512, 512, 512
    data = np.memmap(path, dtype=np.float32, mode="r")
    if data.size != d * h * w:
        raise ValueError(f"Expected NYX {d}x{h}x{w} elements, got {data.size}")
    volume = data.reshape(d, h, w)
    depth = _depth_from_budget(d, h, w, args.cuszhi_z_depth, args.cuszhi_max_voxels)
    for start in _block_starts(d, depth, args.max_samples, args.cuszhi_z_stride):
        arr = np.asarray(volume[start : start + depth], dtype=np.float32)
        arr = _crop_zhw(arr, resolution)
        yield CanonicalSample(
            dataset_id="nyx",
            sample_id=f"z{start:03d}-{start + depth - 1:03d}",
            kind="scientific_field",
            array=arr,
            layout="channel_height_width",
            metadata={
                "source_path": str(path),
                "source_layout": "Z,H,W",
                "cuszhi_pack_z": True,
                "z_axis": "z",
                "slice_range": [int(start), int(start + depth)],
                "channels": int(arr.shape[0]),
                "height": int(arr.shape[1]),
                "width": int(arr.shape[2]),
                "dtype": "float32",
            },
        )


def iter_tomo_z_volume_samples(args: argparse.Namespace, resolution: tuple[int, int] | None) -> Iterator[CanonicalSample]:
    import h5py

    path = Path(args.data_root)
    with h5py.File(path, "r") as f:
        data_key, data = TomoH5Adapter._data_dataset(f)
        d, h, w = map(int, data.shape)
        out_h, out_w = resolution if resolution is not None else (h, w)
        depth = _depth_from_budget(d, out_h, out_w, args.cuszhi_z_depth, args.cuszhi_max_voxels)
        for start in _block_starts(d, depth, args.max_samples, args.cuszhi_z_stride):
            arr = data[start : start + depth].astype(np.float32)
            arr = _crop_zhw(arr, resolution)
            yield CanonicalSample(
                dataset_id="tomo",
                sample_id=f"slice{start:04d}-{start + depth - 1:04d}",
                kind="scientific_field",
                array=arr,
                layout="channel_height_width",
                metadata={
                    "source_path": str(path),
                    "data_key": data_key,
                    "source_layout": "Z,H,W",
                    "cuszhi_pack_z": True,
                    "z_axis": "z",
                    "slice_range": [int(start), int(start + depth)],
                    "channels": int(arr.shape[0]),
                    "height": int(arr.shape[1]),
                    "width": int(arr.shape[2]),
                    "dtype": "float32",
                    "source_dtype": str(data.dtype),
                },
            )


def iter_lysozyme_frame_volume_samples(args: argparse.Namespace, resolution: tuple[int, int] | None) -> Iterator[CanonicalSample]:
    adapter = LysozymeAdapter(args.data_root)
    processed, processed_path = adapter._load_processed()
    if processed is None:
        files = adapter._h5_files()
        if not files:
            raise FileNotFoundError(f"No Lysozyme data found in {args.data_root}")
        raise ValueError("Lysozyme packed-z currently expects processed [V,N,T,H,W] npy/npz data")
    arr5 = processed
    if arr5.ndim != 5:
        raise ValueError(f"Lysozyme processed data must be [V,N,T,H,W], got {arr5.shape}")
    chunks = int(arr5.shape[1])
    frames = int(arr5.shape[2])
    h, w = int(arr5.shape[3]), int(arr5.shape[4])
    flat_frames = chunks * frames
    depth = min(int(args.cuszhi_z_depth), flat_frames) if args.cuszhi_z_depth > 0 else min(500, flat_frames)
    flat = arr5[0].reshape(flat_frames, h, w)
    for start in _block_starts(flat_frames, depth, args.max_samples, args.cuszhi_z_stride):
        arr = np.asarray(flat[start : start + depth], dtype=np.float32)
        arr = _crop_zhw(arr, resolution)
        start_chunk, start_frame = divmod(start, frames)
        end_chunk, end_frame = divmod(start + depth - 1, frames)
        yield CanonicalSample(
            dataset_id="lysozyme",
            sample_id=(
                f"chunk{start_chunk:04d}_frame{start_frame:03d}-"
                f"chunk{end_chunk:04d}_frame{end_frame:03d}"
            ),
            kind="lysozyme",
            array=arr,
            layout="channel_height_width",
            metadata={
                "source_path": str(processed_path),
                "source_layout": "V,N,T,H,W",
                "cuszhi_pack_z": True,
                "z_axis": "frame",
                "flat_frame_range": [int(start), int(start + depth)],
                "chunk_frame_start": [int(start_chunk), int(start_frame)],
                "chunk_frame_end_inclusive": [int(end_chunk), int(end_frame)],
                "channels": int(arr.shape[0]),
                "height": int(arr.shape[1]),
                "width": int(arr.shape[2]),
                "dtype": "float32",
            },
        )


def iter_s2c_band_volume_samples(args: argparse.Namespace, resolution: tuple[int, int] | None) -> Iterator[CanonicalSample]:
    from PIL import Image

    adapter = S2CAdapter(args.data_root, bands=tuple(args.s2c_bands), use_tci=False, tile_size=None)
    readers = [adapter._resolve_path(band) for band in args.s2c_bands]
    images = []
    try:
        paths = []
        for zf, path in readers:
            if zf is not None:
                img = Image.open(zf.open(path))
            else:
                img = Image.open(path)
            images.append(img)
            paths.append(path)
        shapes = {(img.height, img.width) for img in images}
        if len(shapes) != 1:
            raise ValueError(f"S2C bands have mismatched shapes: {sorted(shapes)}")
        height, width = next(iter(shapes))

        if args.tile_size is None and resolution is None:
            planes = [np.asarray(img, dtype=np.float32) for img in images]
            arr = np.stack(planes, axis=0)
            yield CanonicalSample(
                dataset_id="s2c",
                sample_id=f"{Path(paths[0]).stem}_bands{len(args.s2c_bands)}_full",
                kind="s2c",
                array=arr,
                layout="channel_height_width",
                metadata={
                    "source_path": paths,
                    "source_format": "jp2",
                    "source_layout": "bands,H,W",
                    "cuszhi_pack_z": True,
                    "z_axis": "spectral_band",
                    "bands": list(args.s2c_bands),
                    "channels": int(arr.shape[0]),
                    "height": int(arr.shape[1]),
                    "width": int(arr.shape[2]),
                    "dtype": "float32",
                },
            )
            return

        target_h, target_w = resolution if resolution is not None else (int(args.tile_size), int(args.tile_size))
        if target_h > height or target_w > width:
            raise ValueError(f"S2C crop {(target_h, target_w)} exceeds band shape {(height, width)}")
        valid_h = (height // target_h) * target_h
        valid_w = (width // target_w) * target_w
        offset_h = (height - valid_h) // 2
        offset_w = (width - valid_w) // 2
        tiles_h = valid_h // target_h
        tiles_w = valid_w // target_w
        count = 0
        for tile_h in range(tiles_h):
            for tile_w in range(tiles_w):
                if args.max_samples > 0 and count >= args.max_samples:
                    return
                r0 = offset_h + tile_h * target_h
                c0 = offset_w + tile_w * target_w
                box = (c0, r0, c0 + target_w, r0 + target_h)
                planes = [np.asarray(img.crop(box), dtype=np.float32) for img in images]
                arr = np.stack(planes, axis=0)
                if float(arr.max() - arr.min()) < 10:
                    continue
                yield CanonicalSample(
                    dataset_id="s2c",
                    sample_id=f"{Path(paths[0]).stem}_bands{len(args.s2c_bands)}_t{tile_h:03d}x{tile_w:03d}",
                    kind="s2c",
                    array=arr,
                    layout="channel_height_width",
                    metadata={
                        "source_path": paths,
                        "source_format": "jp2",
                        "source_layout": "bands,H,W",
                        "cuszhi_pack_z": True,
                        "z_axis": "spectral_band",
                        "bands": list(args.s2c_bands),
                        "tile_row": int(tile_h),
                        "tile_col": int(tile_w),
                        "channels": int(arr.shape[0]),
                        "height": int(arr.shape[1]),
                        "width": int(arr.shape[2]),
                        "dtype": "float32",
                    },
                )
                count += 1
    finally:
        for img in images:
            img.close()
        for zf, _ in readers:
            if zf is not None:
                zf.close()


def iter_kodak_image_volume_samples(args: argparse.Namespace, resolution: tuple[int, int] | None) -> Iterator[CanonicalSample]:
    from PIL import Image

    adapter = KodakAdapter(args.data_root)
    paths = adapter.image_paths()
    if not paths:
        raise ValueError(f"No Kodak images found in {args.data_root}")
    if args.kodak_stack_images > 0:
        paths = paths[: args.kodak_stack_images]
    elif args.max_samples > 0:
        paths = paths[: args.max_samples]
    arrays = []
    min_h = None
    min_w = None
    for path in paths:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            hwc = np.asarray(rgb, dtype=np.float32)
        chw = np.transpose(hwc, (2, 0, 1))
        arrays.append(chw)
        min_h = chw.shape[1] if min_h is None else min(min_h, chw.shape[1])
        min_w = chw.shape[2] if min_w is None else min(min_w, chw.shape[2])
    target = resolution if resolution is not None else (int(min_h), int(min_w))
    stacked = [_crop_zhw(chw, target) for chw in arrays]
    arr = np.concatenate(stacked, axis=0)
    yield CanonicalSample(
        dataset_id="kodak",
        sample_id=f"images000-{len(paths)-1:03d}_rgbstack",
        kind="image",
        array=arr,
        layout="channel_height_width",
        metadata={
            "source_path": [str(p) for p in paths],
            "source_layout": "N,RGB,H,W",
            "cuszhi_pack_z": True,
            "z_axis": "image_rgb_channel",
            "image_count": int(len(paths)),
            "channels": int(arr.shape[0]),
            "height": int(arr.shape[1]),
            "width": int(arr.shape[2]),
            "dtype": "float32",
        },
    )


def iter_era5_npy_samples(
    path: str | Path,
    max_samples: int,
    max_channels: int,
    time_start: int,
    sample_mode: str,
    variable_index: int,
    resolution: tuple[int, int] | None,
) -> Iterator[CanonicalSample]:
    data = np.load(path, mmap_mode="r")
    if data.ndim != 4:
        raise ValueError(f"ERA5 npy must be [C,T,H,W], got {data.shape}")
    if sample_mode == "time":
        if variable_index < 0 or variable_index >= data.shape[0]:
            raise ValueError(f"variable_index {variable_index} out of range for C={data.shape[0]}")
        group_size = min(max_channels if max_channels > 0 else 3, data.shape[1] - time_start)
        if group_size <= 0:
            raise ValueError(f"No ERA5 time samples available from time_start={time_start}")
        count_available = data.shape[1] - time_start - group_size + 1
        count = count_available if max_samples <= 0 else min(max_samples, count_available)
        for offset in range(count):
            t0 = time_start + offset
            array = np.asarray(data[variable_index, t0 : t0 + group_size], dtype=np.float32)
            if resolution is not None:
                from compression_pipeline.adapters.era5 import center_crop_chw

                array = center_crop_chw(array, resolution)
            yield CanonicalSample(
                dataset_id="era5_npy",
                sample_id=f"era5_var{variable_index:03d}_t{t0:04d}-{t0 + group_size - 1:04d}",
                kind="scientific_field",
                array=array,
                layout="channel_height_width",
                metadata={
                    "source_path": str(path),
                    "source_layout": "C,T,H,W",
                    "era5_npy_3d_mode": "time",
                    "variable_index": int(variable_index),
                    "time_range": [int(t0), int(t0 + group_size)],
                    "dtype": "float32",
                    "height": int(array.shape[1]),
                    "width": int(array.shape[2]),
                    "channels": int(array.shape[0]),
                },
            )
        return
    channels = min(max_channels if max_channels > 0 else data.shape[0], data.shape[0])
    count = data.shape[1] - time_start if max_samples <= 0 else min(max_samples, data.shape[1] - time_start)
    for offset in range(count):
        t = time_start + offset
        array = np.asarray(data[:channels, t], dtype=np.float32)
        sample = CanonicalSample(
            dataset_id="era5_npy",
            sample_id=f"era5_t{t:04d}",
            kind="scientific_field",
            array=array,
            layout="channel_height_width",
            metadata={"source_path": str(path), "source_layout": "C,T,H,W", "time_index": t, "dtype": "float32"},
        )
        yield center_crop_sample(sample, resolution)


def center_crop_sample(sample: CanonicalSample, resolution: tuple[int, int] | None) -> CanonicalSample:
    if resolution is None:
        return sample
    target_h, target_w = resolution
    array = sample.array
    _, height, width = array.shape
    if target_h > height or target_w > width:
        raise ValueError(f"resolution {resolution} exceeds sample size {(height, width)}")
    h0 = (height - target_h) // 2
    w0 = (width - target_w) // 2
    cropped = array[:, h0 : h0 + target_h, w0 : w0 + target_w]
    metadata = dict(sample.metadata)
    metadata["crop_resolution"] = [target_h, target_w]
    return CanonicalSample(sample.dataset_id, sample.sample_id, sample.kind, cropped, sample.layout, metadata)


def prepare_codec_array(
    sample: CanonicalSample,
    args: argparse.Namespace,
    arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    if sample.dataset_id != "lysozyme" or args.lysozyme_invalid_policy == "raw":
        return arr, None, {}

    valid_mask = arr < float(args.lysozyme_invalid_threshold)
    invalid_count = int(arr.size - np.count_nonzero(valid_mask))
    if invalid_count == 0:
        return arr, valid_mask, {
            "mask_aware_evaluation": True,
            "invalid_value_policy": args.lysozyme_invalid_policy,
            "invalid_threshold": float(args.lysozyme_invalid_threshold),
            "invalid_voxel_count": 0,
            "valid_voxel_count": int(arr.size),
            "valid_fraction": 1.0,
        }

    valid_values = arr[valid_mask]
    if valid_values.size == 0:
        raise ValueError(f"{sample.sample_id} has no valid Lysozyme pixels below sentinel threshold")

    cleaned = np.array(arr, copy=True)
    if args.lysozyme_invalid_policy == "zero":
        fill_value = 0.0
    elif args.lysozyme_invalid_policy == "median":
        fill_value = float(np.median(valid_values))
    else:
        raise ValueError(f"Unsupported Lysozyme invalid policy: {args.lysozyme_invalid_policy}")
    cleaned[~valid_mask] = np.float32(fill_value)
    valid_range = float(np.max(valid_values) - np.min(valid_values))
    return np.ascontiguousarray(cleaned), valid_mask, {
        "mask_aware_evaluation": True,
        "invalid_value_policy": args.lysozyme_invalid_policy,
        "invalid_fill_value": fill_value,
        "invalid_threshold": float(args.lysozyme_invalid_threshold),
        "invalid_voxel_count": invalid_count,
        "valid_voxel_count": int(np.count_nonzero(valid_mask)),
        "valid_fraction": float(np.count_nonzero(valid_mask) / arr.size),
        "masked_data_min": float(np.min(valid_values)),
        "masked_data_max": float(np.max(valid_values)),
        "masked_data_range": valid_range,
    }


def run_cuszhi_sample(
    sample: CanonicalSample,
    args: argparse.Namespace,
    eb: float,
    output_dir: Path,
    mode_override: str | None = None,
) -> dict:
    exe = Path(args.cuszhi)
    if not exe.exists():
        raise FileNotFoundError(exe)
    arr = np.ascontiguousarray(sample.array.astype(np.float32, copy=False))
    arr, valid_mask, mask_metrics = prepare_codec_array(sample, args, arr)
    mode = mode_override or args.cuszhi_sample_mode
    if mode not in {"stack", "whole3d"}:
        raise ValueError(f"cuSZ-Hi only supports 3D stack mode, got {mode!r}")
    return run_cuszhi_stack_sample(
        sample,
        args,
        eb,
        output_dir,
        arr,
        exe,
        requested_mode="whole3d",
        valid_mask=valid_mask,
        mask_metrics=mask_metrics,
    )


def run_cuszhi_stack_sample(
    sample: CanonicalSample,
    args: argparse.Namespace,
    eb: float,
    output_dir: Path,
    arr: np.ndarray,
    exe: Path,
    requested_mode: str = "stack",
    valid_mask: np.ndarray | None = None,
    mask_metrics: dict | None = None,
    return_reconstruction: bool = False,
) -> dict:
    wall_start = time.perf_counter()
    channels, height, width = arr.shape
    with tempfile.TemporaryDirectory(prefix="cuszhi_", dir=output_dir) as tmp_dir:
        tmp = Path(tmp_dir)
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = prepend_ld_library_path(env, exe.parent)
        raw = tmp / f"{sample.sample_id}.f32"
        arr.tofile(raw)
        data_range = cuszhi_error_reference_range(arr, args, valid_mask)
        range_for_eb = data_range if data_range >= 1e-8 else 1.0
        abs_eb = max(float(eb) * range_for_eb, float(args.cuszhi_min_abs_eb))
        enc_cmd = [
            str(exe),
            "--report",
            "time,cr",
            "-z",
            "-t",
            "f32",
            "-m",
            "abs",
            "--dim3",
            f"{width}x{height}x{channels}",
            "-e",
            str(abs_eb),
            "--predictor",
            args.cuszhi_predictor,
            "-i",
            str(raw),
            "-s",
            args.cuszhi_scheme,
        ]
        enc_wall, enc_out, enc_peak_mb = run_command_profiled(enc_cmd, env=env)
        comp = raw.with_suffix(raw.suffix + ".cusza")
        if not comp.exists():
            raise RuntimeError(f"cuSZ-Hi did not write compressed file. Output:\n{enc_out[-2000:]}")
        dec_cmd = [str(exe), "--report", "time", "-x", "-i", str(comp), "--compare", str(raw)]
        dec_wall, dec_out, dec_peak_mb = run_command_profiled(dec_cmd, env=env)
        recon_path = raw.with_suffix(raw.suffix + ".cuszx")
        if not recon_path.exists():
            raise RuntimeError(f"cuSZ-Hi did not write decompressed file. Output:\n{dec_out[-2000:]}")
        recon = np.fromfile(recon_path, dtype=np.float32).reshape(arr.shape)
        wall_time = time.perf_counter() - wall_start
        error_values = np.abs((arr - recon)[valid_mask] if valid_mask is not None else (arr - recon))
        max_abs_error = float(np.max(error_values)) if error_values.size else 0.0
        metrics = base_metrics(
            arr,
            recon,
            comp.stat().st_size,
            (parse_cusz_time(enc_out) or enc_wall, parse_cusz_time(dec_out) or dec_wall),
            group_count=1,
            valid_mask=valid_mask,
            extra_metrics={"memory_usage_MB": max(enc_peak_mb, dec_peak_mb), "memory_reserved_MB": None},
        )
        add_lpips(metrics, args, arr, recon)
        metrics["codec_stdout_tail"] = ("\n" + enc_out + "\n" + dec_out)[-2000:]
        metrics["cuszhi_sample_mode"] = requested_mode
        metrics["cuszhi_whole3d"] = True
        metrics["cuszhi_min_abs_eb"] = float(args.cuszhi_min_abs_eb)
        metrics["cuszhi_eb_reference"] = args.cuszhi_eb_reference
        metrics["cuszhi_error_reference_range"] = data_range
        metrics["requested_abs_eb"] = abs_eb
        metrics["max_abs_error"] = max_abs_error
        metrics["error_bound_satisfied"] = bool(max_abs_error <= abs_eb * (1.0 + 1e-4) + 1e-12)
        metrics["sample_wall_time_total"] = wall_time
        metrics["sample_wall_throughput_MBps"] = metrics["original_bytes"] / wall_time / 1e6 if wall_time > 0 else None
        if mask_metrics:
            metrics.update(mask_metrics)
        if return_reconstruction:
            metrics["_reconstruction"] = recon
        return metrics


def cuszhi_error_reference_range(
    arr: np.ndarray,
    args: argparse.Namespace,
    valid_mask: np.ndarray | None = None,
) -> float:
    values = arr[valid_mask] if valid_mask is not None else arr
    if values.size == 0:
        raise ValueError("No valid values available for cuSZ-Hi error reference range")
    if args.cuszhi_eb_reference == "range":
        return float(np.max(values) - np.min(values))
    low = float(np.percentile(values, args.cuszhi_robust_low))
    high = float(np.percentile(values, args.cuszhi_robust_high))
    return float(high - low)


def run_visemz_sample(sample: CanonicalSample, args: argparse.Namespace, eb: float, output_dir: Path) -> dict:
    del eb
    exe = Path(args.visemz_bin)
    analysis = Path(args.visemz_analysis)
    synthesis = Path(args.visemz_synthesis)
    for path in (exe, analysis, synthesis):
        if not path.exists():
            raise FileNotFoundError(path)
    arr = np.ascontiguousarray(sample.array.astype(np.float32, copy=False))
    arr, valid_mask, mask_metrics = prepare_codec_array(sample, args, arr)
    codec_sample = CanonicalSample(sample.dataset_id, sample.sample_id, sample.kind, arr, sample.layout, sample.metadata)
    groups = build_image_groups(codec_sample)
    recon_groups = []
    bitstream_bytes = 0
    side_info_bytes = 0
    encode_time = 0.0
    decode_time = 0.0
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = prepend_ld_library_path(env, DEFAULT_TORCH_LIB, DEFAULT_NVTX_LIB, exe.parent)
    with tempfile.TemporaryDirectory(prefix="visemz_", dir=output_dir) as tmp_dir:
        tmp = Path(tmp_dir)
        for group in groups:
            tensor = np.clip(group.tensor[0], 0.0, 1.0)
            uint8 = np.rint(tensor * 255.0).astype(np.uint8)
            _, height, width = uint8.shape
            raw = tmp / f"{sample.sample_id}_g{group.group_index}.u8"
            comp = tmp / f"{sample.sample_id}_g{group.group_index}.visemz"
            out = tmp / f"{sample.sample_id}_g{group.group_index}.f32"
            uint8.tofile(raw)
            enc_cmd = [str(exe), "-m", str(analysis), "-c", "-i", str(raw), "-o", str(comp), "-x", str(height), "-y", str(width)]
            enc_wall, enc_out = run_command(enc_cmd, env=env)
            if not comp.exists():
                raise RuntimeError(f"visemz did not write compressed file. Output:\n{enc_out[-2000:]}")
            dec_cmd = [str(exe), "-m", str(synthesis), "-d", "-i", str(comp), "-o", str(out)]
            dec_wall, dec_out = run_command(dec_cmd, env=env)
            if not out.exists():
                raise RuntimeError(f"visemz did not write decompressed file. Output:\n{dec_out[-2000:]}")
            recon = np.fromfile(out, dtype=np.float32).reshape(3, height, width)
            recon_groups.append(np.clip(recon, 0.0, 1.0)[None])
            bitstream_bytes += comp.stat().st_size
            side_info_bytes += _normalization_side_info_bytes(group.normalization, group.actual_channels)
            encode_time += parse_visemz_time(enc_out) or enc_wall
            decode_time += parse_visemz_time(dec_out) or dec_wall
    reconstruction = reconstruct_from_groups(groups, recon_groups)
    metrics = base_metrics(
        arr,
        reconstruction,
        bitstream_bytes,
        (encode_time, decode_time),
        group_count=len(groups),
        side_info_bytes=side_info_bytes,
        valid_mask=valid_mask,
        extra_metrics={"memory_usage_MB": process_memory_usage_mb(), "memory_reserved_MB": None},
    )
    add_lpips(metrics, args, arr, reconstruction)
    if mask_metrics:
        metrics.update(mask_metrics)
    metrics["groups"] = len(groups)
    return metrics


def run_graphcomp_sample(sample: CanonicalSample, args: argparse.Namespace, eb: float, output_dir: Path) -> dict:
    lib_path = Path(args.graphcomp_lib_sz)
    if not lib_path.exists():
        raise FileNotFoundError(f"{lib_path} (build models/GraphComp/sz3_wrapper first)")

    arr = np.ascontiguousarray(sample.array.astype(np.float32, copy=False))
    arr, valid_mask, mask_metrics = prepare_codec_array(sample, args, arr)

    with tempfile.TemporaryDirectory(prefix="graphcomp_", dir=output_dir) as tmp_dir:
        tmp = Path(tmp_dir)
        start_encode = time.perf_counter()
        preds, side_info_path, graph_stats = graphcomp_predictor(arr, args, tmp / f"{sample.sample_id}.graphcomp_side.npz")
        side_info_bytes = side_info_path.stat().st_size
        comp = tmp / f"{sample.sample_id}.sz"
        lib = load_graphcomp_lib(lib_path)
        flat_data = np.ascontiguousarray(arr.reshape(-1))
        flat_preds = np.ascontiguousarray(preds.reshape(-1))
        quant = np.empty(flat_data.size, dtype=np.int32)
        data_range = float(np.ptp(flat_data))
        lib.compress(
            flat_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            flat_preds.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            float(eb),
            data_range,
            flat_data.size,
            float(np.mean(flat_data)),
            float(np.std(flat_data)),
            str(comp).encode("utf-8"),
            quant.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
        if not comp.exists():
            raise RuntimeError("GraphComp/SZ3 wrapper did not write compressed file")
        encode_time = time.perf_counter() - start_encode

        start_decode = time.perf_counter()
        decode_preds = load_graphcomp_preds(side_info_path, arr.shape)
        decode_flat_preds = np.ascontiguousarray(decode_preds.reshape(-1))
        recon_flat = np.zeros_like(flat_data)
        lib.decompress(
            str(comp).encode("utf-8"),
            decode_flat_preds.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            flat_data.size,
            recon_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            quant.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
        recon = recon_flat.reshape(arr.shape)
        decode_time = time.perf_counter() - start_decode

        max_error = float(np.max(np.abs(arr - recon)))
        bound = float(eb) * (data_range if data_range >= 1e-8 else 1.0)
        if max_error > bound + 1e-4:
            raise RuntimeError(f"GraphComp max error {max_error} exceeds bound {bound}")

        metrics = base_metrics(
            arr,
            recon,
            comp.stat().st_size,
            (encode_time, decode_time),
            group_count=1,
            side_info_bytes=side_info_bytes,
            valid_mask=valid_mask,
            extra_metrics={"memory_usage_MB": process_memory_usage_mb(), "memory_reserved_MB": None},
        )
        add_lpips(metrics, args, arr, recon)
        if mask_metrics:
            metrics.update(mask_metrics)
        metrics.update(
            {
                "graphcomp_side_info_policy": "segmentation_labels_int32_region_means_float32_npz_compressed",
                "graphcomp_residual_bitstream_bytes": comp.stat().st_size,
                "graphcomp_side_bitstream_bytes": side_info_bytes,
                "graphcomp_scale": float(args.graphcomp_scale),
                "graphcomp_sigma": float(args.graphcomp_sigma),
                "graphcomp_min_size": int(args.graphcomp_min_size),
                "graphcomp_region_count": graph_stats["region_count"],
                "graphcomp_max_error": max_error,
                "graphcomp_error_bound_abs": bound,
            }
        )
        return metrics


def add_lpips(metrics: dict, args: argparse.Namespace, original: np.ndarray, reconstruction: np.ndarray) -> None:
    lpips_fn = getattr(args, "lpips_fn", None)
    if lpips_fn is None:
        return
    value = lpips_fn(original, reconstruction)
    if value is not None:
        metrics["lpips"] = value


def graphcomp_predictor(arr: np.ndarray, args: argparse.Namespace, side_info_path: Path) -> tuple[np.ndarray, Path, dict[str, int]]:
    from skimage.segmentation import felzenszwalb

    preds = np.empty_like(arr, dtype=np.float32)
    labels_for_side = np.empty(arr.shape, dtype=np.int32)
    means_list = []
    offsets = [0]
    region_count = 0
    for channel in range(arr.shape[0]):
        plane = np.ascontiguousarray(arr[channel])
        labels = felzenszwalb(
            plane,
            scale=float(args.graphcomp_scale),
            sigma=float(args.graphcomp_sigma),
            min_size=int(args.graphcomp_min_size),
        )
        labels = np.asarray(labels, dtype=np.int32)
        unique_labels, inverse = np.unique(labels.reshape(-1), return_inverse=True)
        dense_labels = inverse.astype(np.int32).reshape(plane.shape)
        flat = plane.reshape(-1).astype(np.float64)
        sums = np.bincount(inverse, weights=flat)
        counts = np.bincount(inverse)
        means = (sums / np.maximum(counts, 1)).astype(np.float32)
        preds[channel] = means[inverse].reshape(plane.shape)
        labels_for_side[channel] = dense_labels
        means_list.append(means)
        offsets.append(offsets[-1] + int(means.size))
        region_count += int(unique_labels.size)
    np.savez_compressed(
        side_info_path,
        labels=labels_for_side,
        means=np.concatenate(means_list).astype(np.float32),
        offsets=np.asarray(offsets, dtype=np.int64),
    )
    return preds, side_info_path, {"region_count": region_count}


def load_graphcomp_preds(side_info_path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    side = np.load(side_info_path)
    labels = side["labels"]
    means = side["means"]
    offsets = side["offsets"]
    preds = np.empty(shape, dtype=np.float32)
    for channel in range(shape[0]):
        start = int(offsets[channel])
        end = int(offsets[channel + 1])
        preds[channel] = means[start:end][labels[channel]]
    return preds


def load_graphcomp_lib(path: Path):
    lib = ctypes.CDLL(str(path), use_last_error=True)
    lib.compress.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_double,
        ctypes.c_float,
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.compress.restype = ctypes.c_void_p
    lib.decompress.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.decompress.restype = ctypes.c_void_p
    return lib


def run_command(cmd: list[str], env: dict[str, str]) -> tuple[float, str]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout[-4000:]}")
    return elapsed, proc.stdout


def run_command_profiled(cmd: list[str], env: dict[str, str]) -> tuple[float, str, float]:
    start = time.perf_counter()
    proc, peak_mb = run_with_gpu_peak(
        cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout[-4000:]}")
    return elapsed, proc.stdout, float(peak_mb or 0.0)


def parse_visemz_time(output: str) -> float | None:
    match = re.search(r"Time taken for operation:\s*([0-9.]+)\s*microseconds", output)
    return float(match.group(1)) / 1e6 if match else None


def parse_cusz_time(output: str) -> float | None:
    matches = re.findall(r"time.*?([0-9]+(?:\.[0-9]+)?)\s*(?:ms|msec)", output, flags=re.IGNORECASE)
    if matches:
        return sum(float(value) for value in matches) / 1000.0
    return None


def prepend_ld_library_path(env: dict[str, str], *paths: Path) -> str:
    existing = env.get("LD_LIBRARY_PATH", "")
    prefixes = [str(path) for path in paths if path.exists()]
    return ":".join(prefixes + ([existing] if existing else []))


def normalize_model_request(name: str) -> tuple[str, str, str | None]:
    aliases = {
        "cuszhi": ("cuSZ-Hi", "cuSZ-Hi-3D", "whole3d"),
        "cusz-hi": ("cuSZ-Hi", "cuSZ-Hi-3D", "whole3d"),
        "cuSZ-Hi": ("cuSZ-Hi", "cuSZ-Hi-3D", "whole3d"),
        "cuszhi-3d": ("cuSZ-Hi", "cuSZ-Hi-3D", "whole3d"),
        "cusz-hi-3d": ("cuSZ-Hi", "cuSZ-Hi-3D", "whole3d"),
        "cuSZ-Hi-3D": ("cuSZ-Hi", "cuSZ-Hi-3D", "whole3d"),
        "cuszhi-whole3d": ("cuSZ-Hi", "cuSZ-Hi-3D", "whole3d"),
        "cuSZ-Hi-whole3d": ("cuSZ-Hi", "cuSZ-Hi-3D", "whole3d"),
        "visemz": ("visemz", "visemz", None),
        "GraphComp": ("GraphComp", "GraphComp", None),
        "graphcomp": ("GraphComp", "GraphComp", None),
    }
    if name not in aliases:
        raise ValueError(f"Unsupported external model: {name}")
    return aliases[name]


def graphcomp_unavailable_record(dataset_id: str, eb: float) -> dict:
    return {
        "model_name": "GraphComp",
        "model_id": f"GraphComp_eb{format_eb(eb)}",
        "metric": "error_bound",
        "dataset_id": dataset_id,
        "eb": eb,
        "error": (
            "GraphComp in models/GraphComp is not a runnable codec in this checkout: "
            "error_bounded.py hard-codes a missing lib_sz.so path and requires precomputed "
            "graph2grid reconstruction files for each dataset."
        ),
    }


def format_eb(value: float) -> str:
    text = f"{value:.6g}"
    return (
        text.replace("-", "m")
        .replace("+", "")
        .replace(".", "p")
        .replace("e", "e")
    )


def _load_summary(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def write_summary(path: Path, summary: list[dict]) -> None:
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
