import argparse
import fcntl
import gc
import json
import math
import os
import sys
from types import SimpleNamespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from compression_pipeline.adapters.era5 import ERA5Adapter
from compression_pipeline.adapters.e3sm_npz import E3SMNPZAdapter
from compression_pipeline.adapters.hurricane import HurricaneAdapter
from compression_pipeline.adapters.isotropic1024 import Isotropic1024Adapter
from compression_pipeline.adapters.kodak import KodakAdapter
from compression_pipeline.adapters.lysozyme import LysozymeAdapter
from compression_pipeline.adapters.nyx import NYXAdapter
from compression_pipeline.adapters.s2c import S2CAdapter
from compression_pipeline.adapters.shanghai_xray import ShanghaiXrayAdapter
from compression_pipeline.adapters.tomo_h5 import TomoH5Adapter
from compression_pipeline.adapters.turb_rot_npz import TurbRotNPZAdapter
from compression_pipeline.adapters.uvg import UVGAdapter
from compression_pipeline.caesar_runner import CAESAR_N_FRAMES, run_caesar_sequence
from compression_pipeline.canonical import CanonicalSample
from compression_pipeline.cra5_runner import run_cra5_sample
from compression_pipeline.metrics import make_lpips_fn, process_memory_usage_mb, reset_torch_peak_memory, torch_memory_usage_mb
from compression_pipeline.model_registry import image_model_jobs
from compression_pipeline.nvjpeg_codecs import run_nvjpeg_sample, run_nvjpeg2k_sample
from compression_pipeline.runner import run_image_grouped_sample
from compression_pipeline.torch_codecs import CompressAILikeCodec, DCVCRTCodec, DCMVCCodec, ForwardLikelihoodCodec


def parse_args():
    parser = argparse.ArgumentParser(description="Run supported codecs on ERA5 or Kodak through the shared data adapter.")
    parser.add_argument("--dataset", choices=["era5", "era5_npy", "kodak", "tomo", "uvg", "hurricane", "s2c", "nyx", "shanghai_xray", "isot1024", "lysozyme", "turb_rot_npz", "e3sm_npz"], required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--models", nargs="+", default=None, help="Subset: CRA5 DCAE WeConvene LIC_TCM LIC-HPCM RwkvCompress DCVC-RT DCMVC caesar_v caesar_d")
    parser.add_argument("--image_eval_mode", choices=["real", "forward"], default="real",
                        help="For image models, use true compress/decompress or forward+likelihood evaluation.")
    parser.add_argument("--max_model_jobs", type=int, default=-1, help="Limit number of checkpoint/model jobs after filtering; useful for smoke tests.")
    parser.add_argument("--gpu", default=None, help="Optional CUDA_VISIBLE_DEVICES override. Leave unset under Slurm to use the allocated GPU.")
    parser.add_argument("--max_samples", type=int, default=1)
    parser.add_argument("--max_channels", type=int, default=-1)
    parser.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--caesar_root", default=str(PROJECT_ROOT / "models" / "CAESAR"))
    parser.add_argument("--caesar_ckpt_dir", default=str(PROJECT_ROOT / "checkpoints" / "caesar"))
    parser.add_argument(
        "--caesar_norm_type",
        choices=["mean_range", "mean_range_hw"],
        default="mean_range",
        help="CAESAR ScientificDataset instance normalization used by this checkpoint.",
    )
    parser.add_argument("--caesar_start_index", type=int, default=0)
    parser.add_argument("--caesar_num_windows", type=int, default=1,
                        help="Number of contiguous CAESAR temporal windows to aggregate. Default keeps legacy single-window behavior.")
    parser.add_argument("--caesar_window_stride", type=int, default=0,
                        help="Stride between aggregated CAESAR windows. <=0 uses each model's n_frame.")
    parser.add_argument("--caesar_eb", type=float, nargs="+", default=[1e-4],
                        help="CAESAR error bound(s); pass multiple to sweep, e.g. --caesar_eb 1e-4 5e-4 1e-3")
    parser.add_argument(
        "--caesar_no_pca",
        action="store_true",
        help="Disable CAESAR PCA residual postprocessing and report only the learned model bitstream.",
    )
    parser.add_argument("--allow_cra5_adapted", action="store_true",
                        help="Allow experimental non-ERA5 CRA5 runs by resizing/replicating samples to 268x721x1440. Disabled by default for fair benchmarks.")
    parser.add_argument("--tomo_group_frames", type=int, default=1,
                        help="Stack N consecutive tomo frames as N-channel input (e.g. 3 for pseudo-RGB DCVC-RT/DCMVC evaluation).")
    parser.add_argument("--tile_size", type=int, default=None,
                        help="Tile large images into tile_size x tile_size blocks (for s2c dataset).")
    parser.add_argument("--turb_rot_section_index", type=int, default=0,
                        help="Section index used for Turb_Rot CAESAR sequence view.")
    parser.add_argument("--turb_rot_section_start", type=int, default=0,
                        help="First section index used for Turb_Rot 3-channel image samples.")
    parser.add_argument("--turb_rot_image_group_mode", choices=["auto", "variables", "sections"], default="auto",
                        help="For Turb_Rot image models, use velocity variables as channels when available or section slices.")
    parser.add_argument("--npz_image_channels", type=int, default=-1,
                        help="Limit variable channels used by turb_rot_npz/e3sm_npz in variables mode; <=0 keeps all variables.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_lpips", action="store_true", help="Disable optional LPIPS calculation.")
    parser.add_argument(
        "--nvjpeg_quality",
        type=int,
        nargs="+",
        default=[1, 5, 10, 25, 50, 75, 95],
        help="nvJPEG quality values used when --models includes nvjpeg.",
    )
    parser.add_argument(
        "--nvjpeg_binary",
        default=str(PROJECT_ROOT / "tools" / "nvjpeg" / "nvjpeg_roundtrip"),
        help="Path to the nvJPEG roundtrip helper binary.",
    )
    parser.add_argument("--nvjpeg_keep_tmp", action="store_true")
    parser.add_argument(
        "--nvjpeg2k_target_psnr",
        type=float,
        nargs="+",
        default=[40, 50, 60, 70, 80],
        help="nvJPEG2000 target PSNR values used when --models includes nvjpeg2k.",
    )
    parser.add_argument(
        "--nvjpeg2k_binary",
        default=str(PROJECT_ROOT / "tools" / "nvjpeg" / "nvjpeg2k_roundtrip"),
        help="Path to the nvJPEG2000 roundtrip helper binary.",
    )
    parser.add_argument(
        "--nvjpeg2k_keep_tmp",
        action="store_true",
        help="Keep temporary raw input/output files for debugging nvJPEG2000 runs.",
    )
    parser.add_argument(
        "--nvjpeg2k_sample_mode",
        choices=["adapter", "pack_z"],
        default="adapter",
        help="adapter uses the normal image-model samples; pack_z reuses the cuSZ-Hi packed-Z scientific samples.",
    )
    parser.add_argument("--nvjpeg2k_z_depth", type=int, default=0, help="Packed-Z depth for nvJPEG2000; 0 uses dataset defaults.")
    parser.add_argument("--nvjpeg2k_z_stride", type=int, default=0, help="Packed-Z stride for nvJPEG2000; 0 uses z_depth.")
    parser.add_argument("--nvjpeg2k_max_voxels", type=int, default=150_000_000)
    parser.add_argument("--npz_variable_index", type=int, default=0)
    parser.add_argument("--npz_time_start", type=int, default=0)
    parser.add_argument("--era5_time_start", type=int, default=0)
    parser.add_argument("--s2c_bands", nargs="+", default=["B02", "B03", "B04", "B08"])
    parser.add_argument("--kodak_stack_images", type=int, default=0)
    parser.add_argument(
        "--lysozyme_invalid_policy",
        choices=["zero", "median", "raw"],
        default="zero",
        help="Replace Lysozyme uint32 sentinel pixels before compression; raw disables mask-aware handling.",
    )
    parser.add_argument(
        "--lysozyme_invalid_threshold",
        type=float,
        default=4.294967e9,
        help="Values at or above this threshold are treated as Lysozyme detector sentinels.",
    )
    return parser.parse_args()


def iter_dataset_samples(args):
    if getattr(args, "nvjpeg2k_sample_mode", "adapter") == "pack_z":
        yield from iter_nvjpeg2k_packed_z_samples(args)
        return
    if args.dataset == "kodak":
        yield from KodakAdapter(args.data_root).iter_samples(max_samples=args.max_samples)
        return
    if args.dataset == "tomo":
        yield from TomoH5Adapter(args.data_root, group_frames=args.tomo_group_frames).iter_samples(
            max_samples=args.max_samples,
            resolution=tuple(args.resolution) if args.resolution else None,
        )
        return
    if args.dataset == "uvg":
        yield from UVGAdapter(args.data_root).iter_samples(max_samples=args.max_samples)
        return
    if args.dataset == "hurricane":
        yield from HurricaneAdapter(args.data_root).iter_samples(max_samples=args.max_samples)
        return
    if args.dataset == "era5_npy":
        yield from iter_era5_npy_samples(
            args.data_root,
            max_samples=args.max_samples,
            max_channels=args.max_channels,
            resolution=tuple(args.resolution) if args.resolution else None,
        )
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
    if args.dataset == "turb_rot_npz":
        yield from TurbRotNPZAdapter(
            args.data_root,
            section_index=args.turb_rot_section_index,
            section_start=args.turb_rot_section_start,
            image_group_mode=args.turb_rot_image_group_mode,
            image_channel_count=args.npz_image_channels,
        ).iter_samples(max_samples=args.max_samples)
        return
    if args.dataset == "e3sm_npz":
        yield from E3SMNPZAdapter(
            args.data_root,
            section_index=args.turb_rot_section_index,
            section_start=args.turb_rot_section_start,
            image_group_mode=args.turb_rot_image_group_mode,
            image_channel_count=args.npz_image_channels,
        ).iter_samples(max_samples=args.max_samples)
        return
    yield from ERA5Adapter(args.data_root).iter_samples(
        max_samples=args.max_samples,
        max_channels=args.max_channels,
        resolution=tuple(args.resolution) if args.resolution else None,
    )


def prepare_lysozyme_array(args, arr: np.ndarray):
    if args.dataset != "lysozyme" or args.lysozyme_invalid_policy == "raw":
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
        raise ValueError("Lysozyme sample has no valid pixels below sentinel threshold")
    cleaned = np.array(arr, copy=True)
    if args.lysozyme_invalid_policy == "zero":
        fill_value = 0.0
    elif args.lysozyme_invalid_policy == "median":
        fill_value = float(np.median(valid_values))
    else:
        raise ValueError(f"Unsupported Lysozyme invalid policy: {args.lysozyme_invalid_policy}")
    cleaned[~valid_mask] = np.float32(fill_value)
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
        "masked_data_range": float(np.max(valid_values) - np.min(valid_values)),
    }


def iter_nvjpeg2k_packed_z_samples(args):
    from scripts.run_external_scientific_codecs import iter_cuszhi_packed_z_samples

    resolution = tuple(args.resolution) if args.resolution else None
    packed_args = SimpleNamespace(**vars(args))
    packed_args.cuszhi_z_depth = int(args.nvjpeg2k_z_depth)
    packed_args.cuszhi_z_stride = int(args.nvjpeg2k_z_stride)
    packed_args.cuszhi_max_voxels = int(args.nvjpeg2k_max_voxels)
    packed_args.section_index = int(args.turb_rot_section_index)
    packed_args.section_start = int(args.turb_rot_section_start)
    packed_args.npz_image_mode = args.turb_rot_image_group_mode
    packed_args.era5_max_channels = args.max_channels
    yield from iter_cuszhi_packed_z_samples(packed_args, resolution)


def aggregate_caesar_results(results, sample_id, start_indices):
    if len(results) == 1:
        result = dict(results[0])
        result["sample_id"] = sample_id
        result["caesar_num_windows"] = 1
        result["caesar_window_start_indices"] = start_indices
        return result

    first = dict(results[0])
    voxel_counts = np.array([max(int(r.get("voxel_count", np.prod(r.get("shape", [0])))), 1) for r in results], dtype=np.float64)
    mse_values = np.array([float(r["mse"]) for r in results], dtype=np.float64)
    total_voxels = float(np.sum(voxel_counts))
    mse = float(np.sum(mse_values * voxel_counts) / total_voxels)
    data_min = min(float(r.get("original_min", 0.0)) for r in results)
    data_max = max(float(r.get("original_max", 1.0)) for r in results)
    data_range = data_max - data_min
    if data_range < 1e-8:
        data_range = 1.0
    psnr = float("inf") if mse < 1e-30 else float(10 * math.log10((data_range * data_range) / mse))

    bitstream_bytes = float(sum(float(r.get("bitstream_bytes", 0.0)) for r in results))
    side_info_bytes = float(sum(float(r.get("side_info_bytes", 0.0)) for r in results))
    total_bytes = float(sum(float(r.get("total_bytes_with_side_info", r.get("bitstream_bytes", 0.0))) for r in results))
    original_bytes = int(sum(int(r.get("original_bytes", 0)) for r in results))
    encode_time = float(sum(float(r.get("encode_time_avg", 0.0)) for r in results))
    decode_time = float(sum(float(r.get("decode_time_avg", 0.0)) for r in results))

    aggregate = first
    aggregate.update({
        "sample_id": sample_id,
        "start_index": start_indices[0],
        "caesar_num_windows": len(results),
        "caesar_window_start_indices": start_indices,
        "caesar_window_stride": start_indices[1] - start_indices[0] if len(start_indices) > 1 else None,
        "window_results": [
            {
                "start_index": r.get("start_index"),
                "shape": r.get("shape"),
                "bpp": r.get("bpp"),
                "psnr": r.get("psnr"),
                "mse": r.get("mse"),
                "encode_time_avg": r.get("encode_time_avg"),
                "decode_time_avg": r.get("decode_time_avg"),
            }
            for r in results
        ],
        "shape": [r.get("shape") for r in results],
        "original_min": data_min,
        "original_max": data_max,
        "voxel_count": int(total_voxels),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "psnr": psnr,
        "average_variable_psnr": float(np.mean([r["average_variable_psnr"] for r in results if r.get("average_variable_psnr") is not None]))
        if any(r.get("average_variable_psnr") is not None for r in results) else None,
        "average_frame_psnr": float(np.mean([r["average_frame_psnr"] for r in results if r.get("average_frame_psnr") is not None]))
        if any(r.get("average_frame_psnr") is not None for r in results) else None,
        "bpp": bitstream_bytes * 8.0 / total_voxels,
        "scientific_bpp": bitstream_bytes * 8.0 / total_voxels,
        "scientific_bpp_with_side_info": total_bytes * 8.0 / total_voxels,
        "bitstream_bytes": bitstream_bytes,
        "side_info_bytes": side_info_bytes,
        "total_bytes_with_side_info": total_bytes,
        "original_bytes": original_bytes,
        "compression_ratio": original_bytes / total_bytes if total_bytes > 0 else float("inf"),
        "encode_time_avg": encode_time,
        "decode_time_avg": decode_time,
        "encode_throughput": original_bytes / encode_time if encode_time > 0 else None,
        "decode_throughput": original_bytes / decode_time if decode_time > 0 else None,
        "lpips": float(np.mean([r["lpips"] for r in results if r.get("lpips") is not None]))
        if any(r.get("lpips") is not None for r in results) else None,
        "memory_usage_MB": max((r.get("memory_usage_MB") or 0.0) for r in results),
        "memory_reserved_MB": max((r.get("memory_reserved_MB") or 0.0) for r in results),
    })
    return aggregate


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "summary.json"
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    print(f"torch.cuda.is_available()={torch.cuda.is_available()} device={device}", flush=True)
    lpips_fn = None if args.no_lpips else make_lpips_fn(device)

    # Load existing results (resume-safe) so concurrent jobs don't overwrite each other
    existing_keys: set[tuple] = set()
    summary = []
    if summary_file.exists():
        try:
            existing = json.loads(summary_file.read_text(encoding="utf-8"))
            for r in existing:
                if "error" not in r:
                    key_value = r.get("target_jpeg2000_psnr", r.get("quality", r.get("eb", 0)))
                    key = (r.get("model_id", ""), r.get("sample_id", ""), key_value, r.get("start_index", 0))
                    existing_keys.add(key)
            summary = existing
        except Exception:
            pass

    def _safe_write_summary():
        """Atomically write summary with file locking to prevent concurrent overwrites."""
        with open(summary_file, "w", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            json.dump(summary, f, indent=2)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    requested_models = set(args.models) if args.models else None
    caesar_models = requested_caesar_models(requested_models)
    nvjpeg_requested = requested_nvjpeg(requested_models)
    nvjpeg2k_requested = requested_nvjpeg2k(requested_models)
    special_models = {
        "CAESAR",
        "caesar_v",
        "caesar_d",
        "CAESAR-V",
        "CAESAR-D",
        "nvjpeg",
        "nvJPEG",
        "nvjpeg2k",
        "nvJPEG2K",
        "nvJPEG2000",
        "nvjpeg2000",
    }
    non_caesar_models = None if requested_models is None else requested_models - special_models

    if caesar_models:
        if args.dataset == "era5":
            sequence, timestamps = ERA5Adapter(args.data_root).load_sequence(
                max_samples=args.max_samples,
                max_channels=args.max_channels,
                resolution=tuple(args.resolution) if args.resolution else None,
            )
        elif args.dataset == "era5_npy":
            sequence, timestamps = load_era5_npy_sequence(
                args.data_root,
                max_samples=args.max_samples,
                max_channels=args.max_channels,
                resolution=tuple(args.resolution) if args.resolution else None,
            )
        elif args.dataset == "kodak":
            sequence, timestamps = KodakAdapter(args.data_root).load_sequence(
                max_samples=args.max_samples,
                resolution=tuple(args.resolution) if args.resolution else None,
            )
        elif args.dataset == "tomo":
            sequence, timestamps = TomoH5Adapter(args.data_root).load_sequence(
                max_samples=args.max_samples,
                resolution=tuple(args.resolution) if args.resolution else None,
            )
        elif args.dataset == "hurricane":
            sequence, timestamps = HurricaneAdapter(args.data_root).load_sequence(
                max_samples=args.max_samples,
                resolution=tuple(args.resolution) if args.resolution else None,
            )
        elif args.dataset == "nyx":
            sequence, timestamps = NYXAdapter(args.data_root).load_sequence(
                max_samples=args.max_samples,
                resolution=tuple(args.resolution) if args.resolution else None,
            )
        elif args.dataset == "isot1024":
            sequence, timestamps = Isotropic1024Adapter(args.data_root).load_sequence(
                max_samples=args.max_samples,
                resolution=tuple(args.resolution) if args.resolution else None,
            )
        elif args.dataset == "uvg":
            sequence, timestamps = UVGAdapter(args.data_root).load_sequence(
                max_samples=args.max_samples,
                resolution=tuple(args.resolution) if args.resolution else None,
            )
        elif args.dataset == "s2c":
            from compression_pipeline.adapters.s2c import S2CAdapter
            adapter = S2CAdapter(args.data_root, tile_size=args.tile_size)
            sequence, timestamps = adapter.load_sequence(
                max_samples=args.max_samples,
                resolution=tuple(args.resolution) if args.resolution else None,
            )
        elif args.dataset == "lysozyme":
            sequence, timestamps = LysozymeAdapter(args.data_root).load_sequence(
                max_samples=args.max_samples,
                resolution=tuple(args.resolution) if args.resolution else None,
            )
        elif args.dataset == "turb_rot_npz":
            sequence, timestamps = TurbRotNPZAdapter(
                args.data_root,
                section_index=args.turb_rot_section_index,
                section_start=args.turb_rot_section_start,
                image_group_mode=args.turb_rot_image_group_mode,
                image_channel_count=args.npz_image_channels,
            ).load_sequence(
                max_samples=args.max_samples,
                resolution=tuple(args.resolution) if args.resolution else None,
            )
        elif args.dataset == "e3sm_npz":
            sequence, timestamps = E3SMNPZAdapter(
                args.data_root,
                section_index=args.turb_rot_section_index,
                section_start=args.turb_rot_section_start,
                image_group_mode=args.turb_rot_image_group_mode,
                image_channel_count=args.npz_image_channels,
            ).load_sequence(
                max_samples=args.max_samples,
                resolution=tuple(args.resolution) if args.resolution else None,
            )
        else:
            raise SystemExit("CAESAR requires a dataset with sequential structure")
        sequence, valid_mask_vthw, caesar_mask_metrics = prepare_lysozyme_array(args, sequence)
        max_n_frame = max(CAESAR_N_FRAMES[name] for name in caesar_models)
        if args.caesar_num_windows <= 0:
            raise SystemExit(f"--caesar_num_windows must be positive, got {args.caesar_num_windows}")
        max_requested_end = args.caesar_start_index + max_n_frame
        if args.caesar_num_windows > 1:
            max_stride = args.caesar_window_stride if args.caesar_window_stride > 0 else max_n_frame
            max_requested_end = args.caesar_start_index + (args.caesar_num_windows - 1) * max_stride + max_n_frame
        if sequence.shape[1] < max_requested_end:
            raise SystemExit(
                f"CAESAR requires at least {max_requested_end} contiguous samples for "
                f"{args.caesar_num_windows} window(s), got sequence T={sequence.shape[1]}"
            )
        if args.max_samples > 0 and args.max_samples < max_requested_end:
            raise SystemExit(
                f"CAESAR requires at least {max_requested_end} contiguous samples, "
                f"got --max_samples {args.max_samples}"
            )
        for model_name in caesar_models:
            result_model_id = f"{model_name}_no_pca" if args.caesar_no_pca else model_name
            stride = args.caesar_window_stride if args.caesar_window_stride > 0 else CAESAR_N_FRAMES[model_name]
            start_indices = [args.caesar_start_index + i * stride for i in range(args.caesar_num_windows)]
            for eb in args.caesar_eb:
                # Skip if already completed (resume-safe, cross-job dedup)
                sample_id = f"{args.dataset}_{result_model_id}"
                key = (result_model_id, sample_id, eb, args.caesar_start_index)
                if key in existing_keys:
                    print(f"[model] skip {result_model_id} eb={eb} — already in {summary_file.name}", flush=True)
                    continue
                try:
                    print(
                        f"[model] running {result_model_id} eb={eb} on {args.dataset} "
                        f"sequence windows={start_indices}",
                        flush=True,
                    )
                    window_results = []
                    for start_index in start_indices:
                        window_sample_id = sample_id if len(start_indices) == 1 else f"{sample_id}_t{start_index}"
                        window_results.append(run_caesar_sequence(
                            sequence,
                            timestamps,
                            model_name=model_name,
                            caesar_root=args.caesar_root,
                            ckpt_dir=args.caesar_ckpt_dir,
                            output_dir=output_dir,
                            device=device,
                            batch_size=args.batch_size,
                            eb=eb,
                            start_index=start_index,
                            sample_id=window_sample_id,
                            collect_lpips=not args.no_lpips,
                            use_pca_postprocess=not args.caesar_no_pca,
                            valid_mask_vthw=valid_mask_vthw,
                            metric_extras=caesar_mask_metrics,
                            norm_type=args.caesar_norm_type,
                        ))
                    result = aggregate_caesar_results(window_results, sample_id, start_indices)
                    result["eb"] = eb
                    summary.append(result)
                    existing_keys.add(key)
                    print(json.dumps(result, indent=2), flush=True)
                except Exception as exc:
                    summary.append({"model_name": "CAESAR", "model_id": model_name, "metric": "mse", "eb": eb, "error": str(exc)})
                    print(f"[error] {model_name} eb={eb}: {exc}", flush=True)
                finally:
                    _safe_write_summary()

    if nvjpeg_requested:
        if args.dataset != "kodak":
            print("[warn] nvJPEG is an 8-bit RGB JPEG codec; running it on non-Kodak data is usually not meaningful.", flush=True)
        samples = list(iter_dataset_samples(args))
        if not samples:
            raise SystemExit(f"No samples found in {args.data_root}")
        for sample in samples:
            for quality in args.nvjpeg_quality:
                model_id = f"nvjpeg_q{quality}"
                key = (model_id, sample.sample_id, quality, 0)
                if key in existing_keys:
                    print(f"[model] skip {model_id} {sample.sample_id} — already in {summary_file.name}", flush=True)
                    continue
                try:
                    print(f"[model] running {model_id} on {sample.sample_id}", flush=True)
                    result = run_nvjpeg_sample(
                        sample,
                        quality=int(quality),
                        output_dir=output_dir,
                        lpips_fn=lpips_fn,
                        memory_fn=lambda: {"memory_usage_MB": process_memory_usage_mb(), "memory_reserved_MB": None},
                        binary=Path(args.nvjpeg_binary),
                        keep_tmp=args.nvjpeg_keep_tmp,
                    )
                    summary.append(result)
                    existing_keys.add(key)
                    print(json.dumps(result, indent=2), flush=True)
                except Exception as exc:
                    summary.append({
                        "model_name": "nvJPEG",
                        "model_id": model_id,
                        "metric": "quality",
                        "quality": int(quality),
                        "sample_id": sample.sample_id,
                        "error": str(exc),
                    })
                    print(f"[error] {model_id} {sample.sample_id}: {exc}", flush=True)
                finally:
                    _safe_write_summary()

    if nvjpeg2k_requested:
        samples = list(iter_dataset_samples(args))
        if not samples:
            raise SystemExit(f"No samples found in {args.data_root}")
        for sample in samples:
            sample_array, valid_mask, mask_metrics = prepare_lysozyme_array(args, sample.array)
            if sample_array is not sample.array:
                sample = CanonicalSample(
                    sample.dataset_id,
                    sample.sample_id,
                    sample.kind,
                    sample_array,
                    sample.layout,
                    sample.metadata,
                )
            for target_psnr in args.nvjpeg2k_target_psnr:
                model_id = f"nvjpeg2k_psnr{target_psnr:g}"
                key = (model_id, sample.sample_id, target_psnr, 0)
                if key in existing_keys:
                    print(f"[model] skip {model_id} {sample.sample_id} — already in {summary_file.name}", flush=True)
                    continue
                try:
                    print(f"[model] running {model_id} on {sample.sample_id}", flush=True)
                    reset_torch_peak_memory(device)
                    result = run_nvjpeg2k_sample(
                        sample,
                        target_psnr=float(target_psnr),
                        output_dir=output_dir,
                        lpips_fn=lpips_fn,
                        memory_fn=lambda: {"memory_usage_MB": process_memory_usage_mb(), "memory_reserved_MB": None},
                        valid_mask=valid_mask,
                        metric_extras=mask_metrics,
                        binary=Path(args.nvjpeg2k_binary),
                        keep_tmp=args.nvjpeg2k_keep_tmp,
                    )
                    summary.append(result)
                    existing_keys.add(key)
                    print(json.dumps(result, indent=2), flush=True)
                except Exception as exc:
                    summary.append({
                        "model_name": "nvJPEG2000",
                        "model_id": model_id,
                        "metric": "target_psnr",
                        "target_jpeg2000_psnr": float(target_psnr),
                        "sample_id": sample.sample_id,
                        "error": str(exc),
                    })
                    print(f"[error] {model_id} {sample.sample_id}: {exc}", flush=True)
                finally:
                    _safe_write_summary()

    jobs = list(image_model_jobs(args.project_root, non_caesar_models))
    if non_caesar_models:
        missing_models = sorted(non_caesar_models - {job.model_name for job in jobs})
        if missing_models:
            expected = PROJECT_ROOT / "checkpoints"
            raise SystemExit(
                f"No runnable model job/checkpoint found for: {', '.join(missing_models)}. "
                f"Check the model-specific directories under {expected}."
            )
    if args.max_model_jobs > 0:
        jobs = jobs[:args.max_model_jobs]
    if non_caesar_models != set() and jobs:
        samples = list(iter_dataset_samples(args))
        if not samples:
            raise SystemExit(f"No samples found in {args.data_root}")
    else:
        samples = []

    for job in jobs:
        model = None
        try:
            print(f"[model] loading {job.model_id}", flush=True)
            model = job.loader(device)
            params = sum(p.numel() for p in model.parameters())
            codec = None
            if job.model_name != "CRA5":
                if job.codec_cls is not None:
                    codec_cls = job.codec_cls
                else:
                    codec_cls = CompressAILikeCodec if args.image_eval_mode == "real" else ForwardLikelihoodCodec
                codec = codec_cls(model, device=device, divisor=job.divisor, **job.codec_kwargs)
            for sample in samples:
                print(f"[sample] {job.model_id} {sample.sample_id}", flush=True)
                sample_array, valid_mask, mask_metrics = prepare_lysozyme_array(args, sample.array)
                if sample_array is not sample.array:
                    sample = CanonicalSample(
                        sample.dataset_id,
                        sample.sample_id,
                        sample.kind,
                        sample_array,
                        sample.layout,
                        sample.metadata,
                    )
                if job.model_name == "CRA5":
                    reset_torch_peak_memory(device)
                    result = run_cra5_sample(
                        sample,
                        model,
                        device=device,
                        allow_adapted=args.allow_cra5_adapted,
                        valid_mask=valid_mask,
                    )
                    result.update(torch_memory_usage_mb(device))
                    if mask_metrics:
                        result.update(mask_metrics)
                else:
                    reset_torch_peak_memory(device)
                    result = run_image_grouped_sample(
                        sample,
                        codec,
                        lpips_fn=lpips_fn,
                        memory_fn=lambda d=device: torch_memory_usage_mb(d),
                        valid_mask=valid_mask,
                        metric_extras=mask_metrics,
                    )
                result.update({
                    "model_name": job.model_name,
                    "model_id": job.model_id,
                    "metric": job.metric,
                    "checkpoint": job.checkpoint,
                    "params": params,
                    "image_eval_mode": args.image_eval_mode if job.model_name != "CRA5" else "native",
                })
                summary.append(result)
                print(json.dumps(result, indent=2), flush=True)
        except Exception as exc:
            summary.append({
                "model_name": job.model_name,
                "model_id": job.model_id,
                "metric": job.metric,
                "checkpoint": job.checkpoint,
                "error": str(exc),
            })
            print(f"[error] {job.model_id}: {exc}", flush=True)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Results: {summary_file}", flush=True)


def requested_caesar_models(requested_models):
    if requested_models is None:
        return []
    if "CAESAR" in requested_models:
        return ["caesar_v", "caesar_d"]
    models = []
    aliases = {
        "caesar_v": "caesar_v",
        "CAESAR-V": "caesar_v",
        "caesar_d": "caesar_d",
        "CAESAR-D": "caesar_d",
    }
    for requested in requested_models:
        if requested in aliases:
            models.append(aliases[requested])
    return models


def requested_nvjpeg(requested_models):
    if requested_models is None:
        return False
    return bool({"nvjpeg", "nvJPEG"} & set(requested_models))


def requested_nvjpeg2k(requested_models):
    if requested_models is None:
        return False
    return bool({"nvjpeg2k", "nvJPEG2K", "nvJPEG2000", "nvjpeg2000"} & set(requested_models))


def iter_era5_npy_samples(path, max_samples: int, max_channels: int, resolution):
    data = np.load(path, mmap_mode="r")
    if data.ndim != 4:
        raise ValueError(f"ERA5 npy must be [C,T,H,W], got {data.shape}")
    channels = min(max_channels if max_channels > 0 else data.shape[0], data.shape[0])
    count = data.shape[1] if max_samples <= 0 else min(max_samples, data.shape[1])
    for t in range(count):
        array = np.asarray(data[:channels, t], dtype=np.float32)
        if resolution is not None:
            from compression_pipeline.adapters.era5 import center_crop_chw

            array = center_crop_chw(array, resolution)
        yield CanonicalSample(
            dataset_id="era5_npy",
            sample_id=f"era5_t{t:04d}",
            kind="scientific_field",
            array=array,
            layout="channel_height_width",
            metadata={
                "source_path": str(path),
                "source_layout": "C,T,H,W",
                "time_index": int(t),
                "dtype": "float32",
                "height": int(array.shape[1]),
                "width": int(array.shape[2]),
                "channels": int(array.shape[0]),
            },
        )


def load_era5_npy_sequence(path, max_samples: int, max_channels: int, resolution):
    samples = list(
        iter_era5_npy_samples(
            path,
            max_samples=max_samples,
            max_channels=max_channels,
            resolution=resolution,
        )
    )
    if not samples:
        raise ValueError(f"No ERA5 npy samples found in {path}")
    tchw = np.stack([sample.array for sample in samples], axis=0)
    timestamps = [f"era5_t{i:04d}" for i in range(len(samples))]
    return np.transpose(tchw, (1, 0, 2, 3)), timestamps


if __name__ == "__main__":
    main()
