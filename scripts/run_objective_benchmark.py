#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compression_pipeline.canonical import CanonicalSample
from compression_pipeline.metrics import base_metrics, make_lpips_fn, torch_memory_usage_mb
from compression_pipeline.objective_data import checksum, load_normalization, load_objective_samples
from compression_pipeline.objective_stacking import (
    crop_corpus_depth,
    pack_objective_corpus,
    pad_corpus_depth,
    unpack_objective_corpus,
)
from compression_pipeline.torch_codecs import CompressAILikeCodec
from compression_pipeline.views import build_image_groups
from scripts.run_matched_codec_validation import DEFAULT_DATASETS, aggregate_rows, row_weight


PROTOCOL_ID = "aifc-objective-v1"
SCIENTIFIC_DATASETS = {"e3sm_npz", "era5_npy", "hurricane", "nyx", "turb_rot_npz", "tomo", "lysozyme", "s2c"}
GENERAL_DATASETS = {"kodak", "uvg_twilight_1080p"}
DEFAULT_CUSZ_EBS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
DEFAULT_CAESAR_EBS = [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
DEFAULT_J2K_PSNR = [20, 30, 40, 50, 60, 70, 80]
DEFAULT_JPEG_QUALITY = [1, 5, 10, 25, 50, 75, 95]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run objective-v1 non-temporal codec tracks.")
    parser.add_argument("--dataset", choices=DEFAULT_DATASETS, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("unified_results/objective_v1"))
    parser.add_argument("--input-root", type=Path, default=None, help="Prepared manifest root; defaults to --output-root.")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--model-id-contains", nargs="+", default=None)
    parser.add_argument("--sample-id-contains", nargs="+", default=None)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument(
        "--caesar-batch-size",
        type=int,
        default=64,
        help="CAESAR inference batch size; 64 matches the authors' eval_caesar.ipynb.",
    )
    parser.add_argument(
        "--caesar-checkpoint-root",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/caesar",
        help="Directory containing CAESAR-V/D checkpoints.",
    )
    parser.add_argument(
        "--caesar-variant",
        default="original",
        help="Result-curve suffix for a checkpoint variant, e.g. turb_tuned.",
    )
    parser.add_argument(
        "--caesar-norm-type",
        choices=["mean_range", "mean_range_hw"],
        default="mean_range",
        help="CAESAR ScientificDataset normalization expected by the checkpoint.",
    )
    parser.add_argument(
        "--caesar-interpo-rate",
        type=int,
        default=3,
        help="CAESAR-D condition-frame interval; ignored by CAESAR-V.",
    )
    parser.add_argument(
        "--caesar-diffusion-ensemble-size",
        type=int,
        default=1,
        help=(
            "Number of deterministic-sequence diffusion samples decoded and "
            "averaged in pixel space for CAESAR-D."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--cusz-eb", type=float, nargs="+", default=DEFAULT_CUSZ_EBS)
    parser.add_argument("--caesar-eb", type=float, nargs="+", default=DEFAULT_CAESAR_EBS)
    parser.add_argument("--j2k-psnr", type=float, nargs="+", default=DEFAULT_J2K_PSNR)
    parser.add_argument("--jpeg-quality", type=int, nargs="+", default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-lpips", action="store_true", help="Skip LPIPS for endpoint probes only.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def hardware_manifest(gpu: str) -> dict[str, Any]:
    import torch

    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_physical_index": str(gpu),
        "gpu_name": props.name,
        "gpu_total_memory_bytes": int(props.total_memory),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def repeated_roundtrip(
    call: Callable[[], dict[str, Any]],
    warmups: int,
    repeats: int,
    seed: int = 20260722,
    metric_call: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    def run_once() -> dict[str, Any]:
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return call()

    for _ in range(warmups):
        run_once()
    metric_only_probe = metric_call is not None and warmups == 0 and repeats == 1
    measured = [metric_call()] if metric_only_probe else [run_once() for _ in range(repeats)]
    if not measured:
        raise ValueError("At least one measured repetition is required")
    result = dict(measured[0])
    if metric_call is not None and not metric_only_probe:
        metric_result = metric_call()
        if metric_result is not None:
            for field in ("lpips", "ms_ssim"):
                if field in metric_result:
                    result[field] = metric_result[field]
    timing = []
    for row in measured:
        timing.append({
            "encode_seconds": float(row.get("encode_time_avg", 0.0)),
            "decode_seconds": float(row.get("decode_time_avg", 0.0)),
            "roundtrip_seconds": float(row["sample_wall_time_total"]),
            "scientific_bpp_with_side_info": row.get("scientific_bpp_with_side_info"),
            "psnr": row.get("psnr"),
        })
    walls = [item["roundtrip_seconds"] for item in timing]
    encodes = [item["encode_seconds"] for item in timing]
    decodes = [item["decode_seconds"] for item in timing]
    result.update({
        "timing_repetitions": timing,
        "sample_wall_time_total": statistics.median(walls),
        "encode_time_avg": statistics.median(encodes),
        "decode_time_avg": statistics.median(decodes),
        "sample_wall_time_p10": float(np.percentile(walls, 10)),
        "sample_wall_time_p90": float(np.percentile(walls, 90)),
        "deterministic_seed": int(seed),
    })
    original_bytes = int(result.get("original_bytes", 0))
    result["sample_wall_throughput_MBps"] = original_bytes / result["sample_wall_time_total"] / 1e6
    return result


def objective_fields(
    manifest: dict[str, Any],
    normalization: dict[str, Any],
    hardware: dict[str, Any],
    track_id: str,
) -> dict[str, Any]:
    return {
        **manifest,
        "protocol_id": PROTOCOL_ID,
        "track_id": track_id,
        "external_input_manifest": normalization,
        "metric_protocol": PROTOCOL_ID,
        "timing_protocol": PROTOCOL_ID,
        "hardware_manifest": hardware,
        "rate_denominator": "canonical_grid_symbols",
    }


def corpus_objective_fields(
    manifests: list[dict[str, Any]],
    normalization: dict[str, Any],
    hardware: dict[str, Any],
    track_id: str,
    packing: dict[str, Any],
) -> dict[str, Any]:
    def combined_hash(field: str) -> str:
        digest = hashlib.sha256()
        for manifest in manifests:
            digest.update(str(manifest[field]).encode("ascii"))
        return digest.hexdigest()

    fields = objective_fields(
        {
            "dataset_id": manifests[0]["dataset_id"],
            "canonical_sample_id": "__objective_corpus__",
            "canonical_sha256": combined_hash("canonical_sha256"),
            "normalized_canonical_sha256": combined_hash("normalized_canonical_sha256"),
            "canonical_shape": [manifest["canonical_shape"] for manifest in manifests],
            "canonical_symbol_count": sum(int(manifest["canonical_symbol_count"]) for manifest in manifests),
            "canonical_valid_symbol_count": sum(int(manifest["canonical_valid_symbol_count"]) for manifest in manifests),
        },
        normalization,
        hardware,
        track_id,
    )
    fields.update({
        "covered_canonical_sample_ids": [manifest["canonical_sample_id"] for manifest in manifests],
        "covered_canonical_sha256": {
            manifest["canonical_sample_id"]: manifest["canonical_sha256"] for manifest in manifests
        },
        "covered_normalized_canonical_sha256": {
            manifest["canonical_sample_id"]: manifest["normalized_canonical_sha256"] for manifest in manifests
        },
        "corpus_packing_manifest": packing,
    })
    return fields


def finalize_row(row: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
    output = {**row, **fields}
    output["payload_bytes"] = output.get("bitstream_bytes")
    output["normalized_mse"] = output.get("mse")
    normalized_mse = output.get("normalized_mse")
    output["normalized_psnr"] = (
        -10.0 * math.log10(max(float(normalized_mse), 1e-30))
        if isinstance(normalized_mse, (int, float)) and math.isfinite(float(normalized_mse))
        else None
    )
    output["fixed_scale_data_range"] = 1.0
    output["canonical_symbol_count"] = fields["canonical_symbol_count"]
    output["canonical_valid_symbol_count"] = fields["canonical_valid_symbol_count"]
    return output


def tag_caesar_variant(row: dict[str, Any], args: argparse.Namespace, model_name: str, eb: float) -> dict[str, Any]:
    variant = str(args.caesar_variant).strip().lower().replace("-", "_")
    suffix = "" if variant == "original" else f"_{variant}"
    output = dict(row)
    output.update({
        "model_id": f"{model_name}{suffix}-objective-eb{eb:g}",
        "checkpoint_variant": variant,
        "checkpoint_root": str(args.caesar_checkpoint_root.resolve()),
        "caesar_norm_type": args.caesar_norm_type,
        "caesar_interpo_rate": args.caesar_interpo_rate if model_name == "caesar_d" else None,
        "caesar_diffusion_ensemble_size": (
            args.caesar_diffusion_ensemble_size if model_name == "caesar_d" else None
        ),
    })
    return output


def image_point(
    codec: Any,
    dataset_id: str,
    sample_id: str,
    normalized: np.ndarray,
    mask: np.ndarray | None,
    batch_size: int,
    lpips_fn: Callable | None,
    return_reconstruction: bool = False,
) -> dict[str, Any]:
    wall_start = time.perf_counter()
    reconstruction = np.empty_like(normalized, dtype=np.float32)
    tasks: list[tuple[np.ndarray, int, int, int]] = []
    if dataset_id in GENERAL_DATASETS:
        for time_index in range(normalized.shape[1]):
            tasks.append((normalized[:, time_index], -1, time_index, normalized.shape[0]))
    elif dataset_id == "s2c":
        flattened = normalized[:, 0]
        for start in range(0, flattened.shape[0], 3):
            chunk = flattened[start : start + 3]
            actual = chunk.shape[0]
            if actual < 3:
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], 3 - actual, axis=0)], axis=0)
            tasks.append((chunk, -2, start, actual))
    else:
        for variable in range(normalized.shape[0]):
            sample = CanonicalSample(
                dataset_id, f"{sample_id}_v{variable:03d}", "scientific_field", normalized[variable],
                "channel_height_width", {"dtype": "float32", "external_normalized": True, "normalization_id": dataset_id},
            )
            for group in build_image_groups(sample):
                tasks.append((group.tensor[0], variable, group.source_channel_start, group.actual_channels))

    bitstream_bytes = 0
    encode_time = 0.0
    decode_time = 0.0
    for offset in range(0, len(tasks), batch_size):
        batch_tasks = tasks[offset : offset + batch_size]
        batch = np.stack([item[0] for item in batch_tasks], axis=0).astype(np.float32, copy=False)
        result = codec.roundtrip(batch)
        bitstream_bytes += int(result.bitstream_bytes)
        encode_time += float(result.encode_time)
        decode_time += float(result.decode_time)
        for index, (_, variable, start, actual) in enumerate(batch_tasks):
            restored = result.reconstruction[index, :actual]
            if variable == -1:
                reconstruction[:, start] = restored
            elif variable == -2:
                reconstruction[start : start + actual, 0] = restored
            else:
                reconstruction[variable, start : start + actual] = restored

    wall = time.perf_counter() - wall_start
    lpips_value = None
    if lpips_fn is not None:
        values = (
            [lpips_fn(normalized[:, index], reconstruction[:, index]) for index in range(normalized.shape[1])]
            if dataset_id in GENERAL_DATASETS else [lpips_fn(normalized, reconstruction)]
        )
        values = [value for value in values if isinstance(value, (int, float))]
        lpips_value = float(np.mean(values)) if values else None
    extras = {"lpips": lpips_value}
    extras.update(torch_memory_usage_mb("cuda"))
    metrics = base_metrics(
        normalized, reconstruction, bitstream_bytes, (encode_time, decode_time),
        group_count=len(tasks), side_info_bytes=0, valid_mask=mask, extra_metrics=extras,
    )
    metrics["sample_wall_time_total"] = wall
    metrics["sample_wall_throughput_MBps"] = metrics["original_bytes"] / wall / 1e6
    metrics["partition_count"] = len(tasks)
    metrics["partition_policy"] = "RGB frames" if dataset_id in GENERAL_DATASETS else "fixed-normalized 2D groups"
    if return_reconstruction:
        metrics["_reconstruction"] = reconstruction
    return metrics


def run_image_models(args, samples, normalization, manifests, append) -> None:
    import torch
    from compression_pipeline.model_registry import image_model_jobs

    requested = set(args.models)
    registry_names = set()
    if "DCAE" in requested:
        registry_names.add("DCAE")
    if "HPCM" in requested or "LIC-HPCM" in requested:
        registry_names.add("LIC-HPCM")
    if "DCMVC-I" in requested:
        registry_names.add("DCMVC")
    if "DCVC-RT-I" in requested:
        registry_names.add("DCVC-RT")
    jobs = list(image_model_jobs(PROJECT_ROOT, registry_names))
    if args.model_id_contains:
        jobs = [job for job in jobs if any(token in job.model_id for token in args.model_id_contains)]
    if args.smoke:
        selected = []
        for name in registry_names:
            group = [job for job in jobs if job.model_name == name]
            if group:
                selected.append(group[len(group) // 2])
        jobs = selected
    lpips_fn = make_lpips_fn("cuda", max_image_pairs=32 if args.dataset in SCIENTIFIC_DATASETS else None) if not args.no_lpips else None

    for job in jobs:
        print(f"[load] {job.model_id}", flush=True)
        model = job.loader("cuda")
        codec_cls = job.codec_cls or CompressAILikeCodec
        codec = codec_cls(model, device="cuda", divisor=job.divisor, **job.codec_kwargs)
        try:
            for sample, normalized, manifest in zip(samples, normalization, manifests):
                fields = objective_fields(manifest, manifest["external_input_manifest"], args.hardware, "rgb_intra" if args.dataset in GENERAL_DATASETS else "scientific_numeric")
                metric_call = None
                if lpips_fn is not None:
                    metric_call = lambda s=sample, n=normalized: image_point(
                        codec, args.dataset, s.sample_id, n, s.mask, 1, lpips_fn,
                    )
                row = repeated_roundtrip(
                    lambda s=sample, n=normalized: image_point(
                        codec, args.dataset, s.sample_id, n, s.mask,
                        1,
                        None,
                    ),
                    args.warmups, args.repeats,
                    metric_call=metric_call,
                )
                row.update({"model_name": job.model_name, "model_id": job.model_id, "control": Path(job.checkpoint).name if job.checkpoint else job.model_id})
                append(finalize_row(row, fields))
        finally:
            del codec, model
            gc.collect()
            torch.cuda.empty_cache()


def cusz_point(args, sample, normalized, output_dir, eb, lpips_fn=None) -> dict[str, Any]:
    from scripts.run_external_scientific_codecs import run_cuszhi_stack_sample

    codec_args = SimpleNamespace(
        cuszhi=str(PROJECT_ROOT / "models/cuSZ-Hi/build/cuszhi"), cuszhi_scheme="huffman",
        cuszhi_predictor="lorenzo", cuszhi_min_abs_eb=1e-20, cuszhi_eb_reference="range",
        cuszhi_robust_low=0.1, cuszhi_robust_high=99.9, lpips_fn=lpips_fn,
    )
    rows = []
    if args.dataset == "s2c":
        partitions = [(np.ascontiguousarray(normalized[:, 0]), sample.mask[:, 0] if sample.mask is not None else None)]
    else:
        partitions = [
            (np.ascontiguousarray(normalized[variable]), sample.mask[variable] if sample.mask is not None else None)
            for variable in range(normalized.shape[0])
        ]
    for index, (array, part_mask) in enumerate(partitions):
        canonical = CanonicalSample(args.dataset, f"{sample.sample_id}_p{index:03d}", "scientific_field", array, "channel_height_width", {})
        result = run_cuszhi_stack_sample(
            canonical, codec_args, float(eb), output_dir, array, Path(codec_args.cuszhi),
            requested_mode="whole3d", valid_mask=part_mask,
        )
        rows.append(row_weight(result, array.size, int(part_mask.sum()) if part_mask is not None else array.size))
    return aggregate_rows(rows, normalized, sample.mask, {
        "model_name": "cuSZ-Hi", "model_id": f"cuSZ-Hi-objective-eb{eb:g}", "control": float(eb), "eb": float(eb),
        "partition_policy": "spectral band stack" if args.dataset == "s2c" else "one complete 3D volume per variable",
    })


def corpus_lpips(dataset_id: str, packed, reconstruction: np.ndarray, lpips_fn: Callable | None) -> float | None:
    if lpips_fn is None:
        return None
    values = []
    if dataset_id == "kodak":
        originals = unpack_objective_corpus(dataset_id, packed, packed.volume)
        reconstructions = unpack_objective_corpus(dataset_id, packed, reconstruction)
        for original, restored in zip(originals, reconstructions):
            values.append(lpips_fn(original[:, 0], restored[:, 0]))
    elif dataset_id == "uvg_twilight_1080p":
        for index in range(packed.volume.shape[1]):
            values.append(lpips_fn(packed.volume[:, index], reconstruction[:, index]))
    else:
        values.append(lpips_fn(packed.volume, reconstruction))
    values = [value for value in values if isinstance(value, (int, float))]
    return float(np.mean(values)) if values else None


def cusz_corpus_point(args, packed, output_dir, eb, lpips_fn: Callable | None) -> dict[str, Any]:
    from scripts.run_external_scientific_codecs import run_cuszhi_stack_sample

    codec_args = SimpleNamespace(
        cuszhi=str(PROJECT_ROOT / "models/cuSZ-Hi/build/cuszhi"), cuszhi_scheme="huffman",
        cuszhi_predictor="lorenzo", cuszhi_min_abs_eb=1e-20, cuszhi_eb_reference="range",
        cuszhi_robust_low=0.1, cuszhi_robust_high=99.9, lpips_fn=None,
    )
    rows = []
    reconstructions = []
    for variable in range(packed.volume.shape[0]):
        array = np.ascontiguousarray(packed.volume[variable])
        part_mask = packed.mask[variable] if packed.mask is not None else None
        canonical = CanonicalSample(
            args.dataset, f"objective_corpus_v{variable:03d}", "scientific_field", array,
            "channel_height_width", {},
        )
        result = run_cuszhi_stack_sample(
            canonical, codec_args, float(eb), output_dir, array, Path(codec_args.cuszhi),
            requested_mode="whole3d", valid_mask=part_mask, return_reconstruction=True,
        )
        reconstructions.append(result.pop("_reconstruction"))
        rows.append(row_weight(result, array.size, int(part_mask.sum()) if part_mask is not None else array.size))
    reconstruction = np.stack(reconstructions, axis=0)
    row = aggregate_rows(rows, packed.volume, packed.mask, {
        "model_name": "cuSZ-Hi", "model_id": f"cuSZ-Hi-objective-eb{eb:g}",
        "control": float(eb), "eb": float(eb), "partition_policy": packed.metadata["packing"],
    })
    row["lpips"] = corpus_lpips(args.dataset, packed, reconstruction, lpips_fn)
    row["corpus_packing_manifest"] = packed.metadata
    return row


def run_cusz(args, samples, normalized_samples, manifests, output_dir, append) -> None:
    if "cuSZ-Hi" not in args.models:
        return
    controls = [args.cusz_eb[len(args.cusz_eb) // 2]] if args.smoke else args.cusz_eb
    if args.dataset in {"s2c", "kodak", "uvg_twilight_1080p"}:
        packed = pack_objective_corpus(args.dataset, normalized_samples, [sample.mask for sample in samples])
        fields = corpus_objective_fields(
            manifests, manifests[0]["external_input_manifest"], args.hardware,
            "video_temporal" if args.dataset == "uvg_twilight_1080p" else (
                "rgb_intra" if args.dataset == "kodak" else "error_bounded"
            ),
            packed.metadata,
        )
        lpips_fn = make_lpips_fn("cuda", max_image_pairs=32) if not args.no_lpips else None
        for eb in controls:
            try:
                row = repeated_roundtrip(
                    lambda e=eb: cusz_corpus_point(args, packed, output_dir, e, None),
                    args.warmups, args.repeats,
                    metric_call=(
                        (lambda e=eb: cusz_corpus_point(args, packed, output_dir, e, lpips_fn))
                        if lpips_fn is not None else None
                    ),
                )
                append(finalize_row(row, fields))
            except Exception as exc:
                append({
                    **fields, "model_name": "cuSZ-Hi", "model_id": f"cuSZ-Hi-objective-eb{eb:g}",
                    "control": float(eb), "eb": float(eb), "error": str(exc),
                })
        return
    if args.dataset not in SCIENTIFIC_DATASETS:
        return
    lpips_fn = make_lpips_fn("cuda", max_image_pairs=32 if args.dataset in SCIENTIFIC_DATASETS else None) if not args.no_lpips else None
    for eb in controls:
        for sample, normalized, manifest in zip(samples, normalized_samples, manifests):
            fields = objective_fields(manifest, manifest["external_input_manifest"], args.hardware, "error_bounded")
            try:
                row = repeated_roundtrip(
                    lambda s=sample, n=normalized: cusz_point(args, s, n, output_dir, eb),
                    args.warmups, args.repeats,
                    metric_call=(
                        lambda s=sample, n=normalized: cusz_point(args, s, n, output_dir, eb, lpips_fn)
                    ) if lpips_fn is not None else None,
                )
                append(finalize_row(row, fields))
            except Exception as exc:
                append({**fields, "model_name": "cuSZ-Hi", "model_id": f"cuSZ-Hi-objective-eb{eb:g}", "control": float(eb), "eb": float(eb), "error": str(exc)})


def j2k_point(sample, normalized, output_dir, target, lpips_fn: Callable | None = None) -> dict[str, Any]:
    from compression_pipeline.nvjpeg_codecs import run_nvjpeg2k_sample

    rows = []
    if sample.dataset_id in GENERAL_DATASETS:
        partitions = [
            (np.ascontiguousarray(normalized[:, index]), None, f"f{index:03d}")
            for index in range(normalized.shape[1])
        ]
    else:
        partitions = [
            (
                np.ascontiguousarray(normalized[variable]),
                sample.mask[variable] if sample.mask is not None else None,
                f"v{variable:03d}",
            )
            for variable in range(normalized.shape[0])
        ]
    for array, part_mask, suffix in partitions:
        canonical = CanonicalSample(sample.dataset_id, f"{sample.sample_id}_{suffix}", "scientific_field", array, "channel_height_width", {})
        result = run_nvjpeg2k_sample(
            canonical, float(target), output_dir, valid_mask=part_mask, fixed_unit_range=True, lpips_fn=lpips_fn,
        )
        rows.append(row_weight(result, array.size, int(part_mask.sum()) if part_mask is not None else array.size))
    row = aggregate_rows(rows, normalized, sample.mask, {
        "model_name": "nvJPEG2000", "model_id": f"nvJPEG2000-objective-psnr{target:g}", "control": float(target),
        "partition_policy": "fixed-unit-range uint16 planes per variable",
    })
    values = [item.get("lpips") for item in rows if isinstance(item.get("lpips"), (int, float))]
    row["lpips"] = float(np.mean(values)) if values else None
    return row


def run_j2k(args, samples, normalized_samples, manifests, output_dir, append) -> None:
    if "nvJPEG2000" not in args.models:
        return
    controls = [args.j2k_psnr[len(args.j2k_psnr) // 2]] if args.smoke else args.j2k_psnr
    lpips_fn = make_lpips_fn("cuda", max_image_pairs=32 if args.dataset in SCIENTIFIC_DATASETS else None) if not args.no_lpips else None
    for target in controls:
        for sample, normalized, manifest in zip(samples, normalized_samples, manifests):
            fields = objective_fields(manifest, manifest["external_input_manifest"], args.hardware, "rgb_intra" if args.dataset in GENERAL_DATASETS else "scientific_numeric")
            metric_call = None
            if lpips_fn is not None:
                metric_call = lambda s=sample, n=normalized: j2k_point(
                    s, n, output_dir, target, lpips_fn=lpips_fn,
                )
            row = repeated_roundtrip(
                lambda s=sample, n=normalized: j2k_point(s, n, output_dir, target, lpips_fn=None),
                args.warmups, args.repeats,
                metric_call=metric_call,
            )
            append(finalize_row(row, fields))


def jpeg_point(sample, normalized, output_dir, quality, lpips_fn: Callable | None = None) -> dict[str, Any]:
    from compression_pipeline.nvjpeg_codecs import run_nvjpeg_sample

    rows = []
    for time_index in range(normalized.shape[1]):
        array = np.ascontiguousarray(normalized[:, time_index])
        canonical = CanonicalSample(sample.dataset_id, f"{sample.sample_id}_f{time_index:03d}", "image", array, "channel_height_width", {})
        result = run_nvjpeg_sample(canonical, int(quality), output_dir, lpips_fn=lpips_fn, fixed_unit_range=True)
        rows.append(row_weight(result, array.size, array.size))
    row = aggregate_rows(rows, normalized, sample.mask, {
        "model_name": "nvJPEG", "model_id": f"nvJPEG-objective-q{quality}", "control": int(quality),
        "partition_policy": "RGB frame",
    })
    values = [item.get("lpips") for item in rows if isinstance(item.get("lpips"), (int, float))]
    row["lpips"] = float(np.mean(values)) if values else None
    return row


def run_jpeg(args, samples, normalized_samples, manifests, output_dir, append) -> None:
    if "nvJPEG" not in args.models or args.dataset not in GENERAL_DATASETS:
        return
    controls = [args.jpeg_quality[len(args.jpeg_quality) // 2]] if args.smoke else args.jpeg_quality
    lpips_fn = None if args.no_lpips else make_lpips_fn("cuda")
    for quality in controls:
        for sample, normalized, manifest in zip(samples, normalized_samples, manifests):
            fields = objective_fields(manifest, manifest["external_input_manifest"], args.hardware, "rgb_intra")
            row = repeated_roundtrip(
                lambda s=sample, n=normalized: jpeg_point(s, n, output_dir, quality, lpips_fn=None),
                args.warmups, args.repeats,
                metric_call=lambda s=sample, n=normalized: jpeg_point(
                    s, n, output_dir, quality, lpips_fn=lpips_fn
                ),
            )
            append(finalize_row(row, fields))


def caesar_stacked_point(
    args, compressor, prepared, packed, padded, model_name, eb, canonical_symbol_count, lpips_fn
) -> dict[str, Any]:
    from compression_pipeline.caesar_runner import _count_caesar_params
    from compression_pipeline.objective_caesar import caesar_corpus_raw_roundtrip

    recon_padded, compressed_bytes, encode_seconds, decode_seconds, wall_seconds = caesar_corpus_raw_roundtrip(
        compressor, prepared, float(eb)
    )
    recon_padded = np.ascontiguousarray(
        recon_padded.reshape(recon_padded.shape[0], -1, *recon_padded.shape[-2:])
    )
    reconstruction = crop_corpus_depth(padded, recon_padded)
    metrics = base_metrics(
        packed.volume,
        reconstruction,
        compressed_bytes,
        (encode_seconds, decode_seconds),
        group_count=len(prepared.dataset),
        side_info_bytes=0,
        valid_mask=packed.mask,
        extra_metrics={
            **torch_memory_usage_mb("cuda"),
            "lpips": corpus_lpips(args.dataset, packed, reconstruction, lpips_fn),
        },
    )
    metrics.update({
        "model_name": "CAESAR",
        "model_id": f"{model_name}-objective-eb{eb:g}",
        "control": float(eb),
        "eb": float(eb),
        "params": _count_caesar_params(compressor, model_name),
        "caesar_postprocess": "pca",
        "caesar_inference_batch_size": int(prepared.loader.batch_size),
        "partition_policy": packed.metadata["packing"],
        "corpus_packing_manifest": padded.metadata,
        "sample_wall_time_total": wall_seconds,
        "sample_wall_throughput_MBps": canonical_symbol_count * 4 / wall_seconds / 1e6,
    })
    return metrics


def run_caesar(args, samples, normalized_samples, manifests, output_dir, append) -> None:
    requested = set(args.models)
    if not ({"CAESAR-V", "CAESAR-D"} & requested):
        return
    from compression_pipeline.objective_caesar import (
        caesar_corpus_roundtrip,
        load_caesar_compressor,
        prepare_caesar_corpus,
    )

    controls = [args.caesar_eb[len(args.caesar_eb) // 2]] if args.smoke else args.caesar_eb
    lpips_fn = make_lpips_fn("cuda") if not args.no_lpips else None
    if args.dataset in {"s2c", "kodak", "uvg_twilight_1080p"}:
        packed = pack_objective_corpus(args.dataset, normalized_samples, [sample.mask for sample in samples])
        fields = corpus_objective_fields(
            manifests, manifests[0]["external_input_manifest"], args.hardware,
            "video_temporal" if args.dataset == "uvg_twilight_1080p" else (
                "rgb_intra" if args.dataset == "kodak" else "scientific_numeric"
            ),
            packed.metadata,
        )
        for cli_name, model_name in [("CAESAR-V", "caesar_v"), ("CAESAR-D", "caesar_d")]:
            if cli_name not in requested:
                continue
            print(f"[load] {cli_name}", flush=True)
            compressor = load_caesar_compressor(
                model_name,
                PROJECT_ROOT / "models/CAESAR",
                args.caesar_checkpoint_root,
                "cuda",
                args.caesar_interpo_rate,
                args.caesar_diffusion_ensemble_size,
            )
            padded = pad_corpus_depth(packed, 8 if model_name == "caesar_v" else 16)
            prepared = prepare_caesar_corpus(
                padded.volume, padded.mask, model_name, PROJECT_ROOT / "models/CAESAR", output_dir,
                "objective_corpus", batch_size=args.caesar_batch_size,
                norm_type=args.caesar_norm_type,
            )
            try:
                for eb in controls:
                    try:
                        row = repeated_roundtrip(
                            lambda e=eb: caesar_stacked_point(
                                args, compressor, prepared, packed, padded, model_name, e,
                                fields["canonical_symbol_count"], None,
                            ),
                            args.warmups, args.repeats,
                            metric_call=(
                                (lambda e=eb: caesar_stacked_point(
                                    args, compressor, prepared, packed, padded, model_name, e,
                                    fields["canonical_symbol_count"], lpips_fn,
                                )) if lpips_fn is not None else None
                            ),
                        )
                        row = tag_caesar_variant(row, args, model_name, eb)
                        append(finalize_row(row, fields))
                    except Exception as exc:
                        append({
                            **fields, **tag_caesar_variant({}, args, model_name, eb), "model_name": "CAESAR",
                            "control": float(eb), "eb": float(eb), "error": str(exc),
                        })
            finally:
                prepared.close()
                del compressor
                gc.collect()
                import torch
                torch.cuda.empty_cache()
        return
    for cli_name, model_name in [("CAESAR-V", "caesar_v"), ("CAESAR-D", "caesar_d")]:
        if cli_name not in requested:
            continue
        print(f"[load] {cli_name}", flush=True)
        compressor = load_caesar_compressor(
            model_name,
            PROJECT_ROOT / "models/CAESAR",
            args.caesar_checkpoint_root,
            "cuda",
            args.caesar_interpo_rate,
            args.caesar_diffusion_ensemble_size,
        )
        try:
            for sample, normalized, manifest in zip(samples, normalized_samples, manifests):
                corpus = prepare_caesar_corpus(
                    normalized, sample.mask, model_name, PROJECT_ROOT / "models/CAESAR", output_dir,
                    sample.sample_id, batch_size=args.caesar_batch_size,
                    norm_type=args.caesar_norm_type,
                )
                try:
                    fields = objective_fields(manifest, manifest["external_input_manifest"], args.hardware, "scientific_numeric")
                    for eb in controls:
                        try:
                            row = repeated_roundtrip(
                                lambda e=eb, c=corpus: caesar_corpus_roundtrip(
                                    compressor, c, model_name, e, manifest["canonical_symbol_count"]
                                ),
                                args.warmups, args.repeats,
                                metric_call=(
                                    lambda e=eb, c=corpus: caesar_corpus_roundtrip(
                                        compressor, c, model_name, e, manifest["canonical_symbol_count"], lpips_fn
                                    )
                                ) if lpips_fn is not None else None,
                            )
                            row = tag_caesar_variant(row, args, model_name, eb)
                            append(finalize_row(row, fields))
                        except Exception as exc:
                            append({
                                **fields, **tag_caesar_variant({}, args, model_name, eb), "model_name": "CAESAR",
                                "control": float(eb), "eb": float(eb), "error": str(exc),
                            })
                finally:
                    corpus.close()
        finally:
            del compressor
            gc.collect()
            import torch
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(f"CUDA unavailable on physical GPU {args.gpu}")
    if args.smoke:
        args.warmups = 0
        args.repeats = 1
    if args.models is None:
        if args.dataset in SCIENTIFIC_DATASETS:
            args.models = [
                "DCAE", "HPCM", "CAESAR-V", "CAESAR-D", "cuSZ-Hi", "nvJPEG2000",
                "DCMVC-I", "DCVC-RT-I",
            ]
        elif args.dataset == "kodak":
            args.models = [
                "DCAE", "HPCM", "CAESAR-V", "CAESAR-D", "cuSZ-Hi", "nvJPEG",
                "DCMVC-I", "DCVC-RT-I",
            ]
        else:
            # UVG P-frame codecs are run by run_objective_video.py.
            args.models = ["DCAE", "HPCM", "CAESAR-V", "CAESAR-D", "cuSZ-Hi", "nvJPEG"]
    args.hardware = hardware_manifest(args.gpu)
    output_dir = args.output_root / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = (args.input_root or args.output_root) / args.dataset
    sample_manifest_path = input_dir / "samples.json"
    normalization_path = input_dir / "normalization.json"
    if not sample_manifest_path.exists() or not normalization_path.exists():
        raise FileNotFoundError(f"Run prepare_objective_inputs.py first for {args.dataset}")
    manifests = json.loads(sample_manifest_path.read_text(encoding="utf-8"))
    samples = load_objective_samples(args.dataset)
    if args.sample_id_contains:
        samples = [
            sample for sample in samples
            if any(token in sample.sample_id for token in args.sample_id_contains)
        ]
        if not samples:
            raise ValueError(f"No objective samples match {args.sample_id_contains}")
    normalization_spec = load_normalization(normalization_path)
    normalized_samples = [normalization_spec.normalize(sample.raw) for sample in samples]
    manifest_by_id = {item["canonical_sample_id"]: item for item in manifests}
    manifests = [manifest_by_id[sample.sample_id] for sample in samples]
    for sample, normalized, manifest in zip(samples, normalized_samples, manifests):
        if checksum(sample.raw, sample.mask) != manifest["canonical_sha256"]:
            raise ValueError(f"Raw checksum changed for {sample.sample_id}")
        if checksum(normalized, sample.mask) != manifest["normalized_canonical_sha256"]:
            raise ValueError(f"Normalized checksum changed for {sample.sample_id}")

    summary_path = output_dir / "summary.json"
    rows = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
    requested = set(args.models)
    if args.force:
        def keep(row):
            family = str(row.get("model_name"))
            aliases = {"LIC-HPCM": "LIC-HPCM", "DCMVC": "DCMVC-I", "DCVC-RT": "DCVC-RT-I"}
            if family == "CAESAR" and ({"CAESAR-V", "CAESAR-D"} & requested):
                return False
            requested_aliases = {"LIC-HPCM" if value == "HPCM" else value for value in requested}
            return aliases.get(family, family) not in requested_aliases
        rows = [row for row in rows if keep(row)]
        summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    done = {(row.get("model_id"), row.get("canonical_sample_id"), str(row.get("control"))) for row in rows if "error" not in row}

    def append(row: dict[str, Any]) -> None:
        key = (row.get("model_id"), row.get("canonical_sample_id"), str(row.get("control")))
        if key in done:
            print(f"[skip] {key}", flush=True)
            return
        row["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rows.append(row)
        if "error" not in row:
            done.add(key)
        summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        if "error" in row:
            print(f"[error] {row.get('model_id')} {row.get('canonical_sample_id')}: {row['error'][-300:]}", flush=True)
        else:
            print(
                f"[result] {row['model_id']} {row['canonical_sample_id']} "
                f"bpp={row['scientific_bpp_with_side_info']:.5g} psnr={row['psnr']:.3f} "
                f"wall={row['sample_wall_throughput_MBps']:.3f} MB/s",
                flush=True,
            )

    run_image_models(args, samples, normalized_samples, manifests, append)
    run_caesar(args, samples, normalized_samples, manifests, output_dir, append)
    run_cusz(args, samples, normalized_samples, manifests, output_dir, append)
    run_j2k(args, samples, normalized_samples, manifests, output_dir, append)
    run_jpeg(args, samples, normalized_samples, manifests, output_dir, append)
    print(summary_path)


if __name__ == "__main__":
    main()
