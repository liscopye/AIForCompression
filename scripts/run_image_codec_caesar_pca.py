#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compression_pipeline.canonical import CanonicalSample
from compression_pipeline.metrics import base_metrics, make_lpips_fn, reset_torch_peak_memory, torch_memory_usage_mb
from compression_pipeline.model_registry import image_model_jobs
from compression_pipeline.runner import _normalization_side_info_bytes
from compression_pipeline.torch_codecs import CompressAILikeCodec, ForwardLikelihoodCodec
from compression_pipeline.views import build_image_groups, reconstruct_from_groups
from scripts.run_dataset_compression import iter_dataset_samples, prepare_lysozyme_array


DEFAULT_EBS = [1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one image-codec checkpoint and add CAESAR PCA residual postprocessing."
    )
    parser.add_argument("--dataset", choices=[
        "era5", "era5_npy", "kodak", "tomo", "uvg", "hurricane", "s2c", "nyx",
        "shanghai_xray", "isot1024", "lysozyme", "turb_rot_npz", "e3sm_npz",
    ], required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    parser.add_argument("--caesar_root", default=str(PROJECT_ROOT / "models" / "CAESAR"))
    parser.add_argument("--models", nargs="+", default=["DCAE", "HPCM-base", "HPCM-large"])
    parser.add_argument("--checkpoint_index", type=int, default=3, help="1-based checkpoint index inside each family.")
    parser.add_argument("--eb", type=float, nargs="+", default=DEFAULT_EBS)
    parser.add_argument("--auto_select_eb", action="store_true",
                        help="Run all probe EB values, then keep seven points spaced evenly in log-bpp.")
    parser.add_argument("--selected_points", type=int, default=7)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--image_eval_mode", choices=["real", "forward"], default="real")
    parser.add_argument("--max_samples", type=int, default=1)
    parser.add_argument("--max_channels", type=int, default=-1)
    parser.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_lpips", action="store_true")
    parser.add_argument("--tomo_group_frames", type=int, default=1)
    parser.add_argument("--tile_size", type=int, default=None)
    parser.add_argument("--turb_rot_section_index", type=int, default=0)
    parser.add_argument("--turb_rot_section_start", type=int, default=0)
    parser.add_argument("--turb_rot_image_group_mode", choices=["auto", "variables", "sections"], default="auto")
    parser.add_argument("--npz_image_channels", type=int, default=-1)
    parser.add_argument("--npz_variable_index", type=int, default=0)
    parser.add_argument("--npz_time_start", type=int, default=0)
    parser.add_argument("--era5_time_start", type=int, default=0)
    parser.add_argument("--s2c_bands", nargs="+", default=["B02", "B03", "B04", "B08"])
    parser.add_argument("--kodak_stack_images", type=int, default=0)
    parser.add_argument("--lysozyme_invalid_policy", choices=["zero", "median", "raw"], default="zero")
    parser.add_argument("--lysozyme_invalid_threshold", type=float, default=4.294967e9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} device={device}", flush=True)

    lpips_fn = None if args.no_lpips else make_lpips_fn(device)
    samples = list(iter_dataset_samples(args))
    if not samples:
        raise SystemExit(f"No samples found in {args.data_root}")

    selected_jobs = select_jobs(args.project_root, args.models, args.checkpoint_index)
    if not selected_jobs:
        raise SystemExit(f"No image model jobs selected for {args.models}")

    all_rows: list[dict] = []
    for job in selected_jobs:
        model = None
        try:
            print(f"[model] loading {job.model_id}", flush=True)
            model = job.loader(device)
            params = sum(p.numel() for p in model.parameters())
            codec_cls = job.codec_cls or (CompressAILikeCodec if args.image_eval_mode == "real" else ForwardLikelihoodCodec)
            codec = codec_cls(model, device=device, divisor=job.divisor, **job.codec_kwargs)
            sample_runs = []
            for sample in samples:
                sample_array, valid_mask, mask_metrics = prepare_lysozyme_array(args, sample.array)
                if sample_array is not sample.array:
                    sample = CanonicalSample(
                        sample.dataset_id, sample.sample_id, sample.kind, sample_array, sample.layout, sample.metadata
                    )
                print(f"[sample] {job.model_id} {sample.sample_id}", flush=True)
                reset_torch_peak_memory(device)
                sample_runs.append(run_base_image_codec(sample, codec, lpips_fn, valid_mask, mask_metrics))

            rows = []
            base_rows = []
            for run in sample_runs:
                base = dict(run["base_metrics"])
                base.update(common_job_fields(job, params, args.image_eval_mode))
                base["pca_postprocess"] = "none"
                base["label"] = pca_curve_label(job, enabled=False)
                base_rows.append(base)

            base_agg = aggregate_rows(base_rows, f"{job.model_id}_base")
            base_agg.update(common_job_fields(job, params, args.image_eval_mode))
            base_agg["model_id"] = f"{job.model_id}_base"
            base_agg["pca_postprocess"] = "none"
            base_agg["label"] = pca_curve_label(job, enabled=False)
            rows.append(base_agg)

            candidate_rows = []
            for eb in args.eb:
                agg = run_pca_eb(sample_runs, job, params, args.image_eval_mode, eb, device, Path(args.caesar_root), lpips_fn)
                candidate_rows.append(agg)
                if "bpp" in agg and "psnr" in agg:
                    print(
                        f"[pca] {job.model_id} eb={eb:g} bpp={agg['bpp']:.4g} psnr={agg['psnr']:.3f}",
                        flush=True,
                    )
                else:
                    print(f"[pca-error] {job.model_id} eb={eb:g}: {agg.get('error')}", flush=True)

            selected_rows = candidate_rows
            if args.auto_select_eb:
                selected_rows = select_even_log_bpp(candidate_rows, args.selected_points)
                selected = ", ".join(f"{r.get('pca_eb'):g}@{r.get('bpp'):.4g}" for r in selected_rows)
                print(f"[select] {job.model_id}: {selected}", flush=True)

            rows.extend(selected_rows)
            for agg in selected_rows:
                eb = float(agg["pca_eb"])
                agg.update(common_job_fields(job, params, args.image_eval_mode))
                agg["model_name"] = f"{job.model_name}+CAESAR-PCA"
                agg["model_id"] = f"{job.model_id}_caesar_pca_eb{eb:g}"
                agg["label"] = pca_curve_label(job, enabled=True)
                agg["pca_eb"] = float(eb)
                agg["pca_postprocess"] = "caesar_pca"

            all_rows.extend(rows)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"Results: {summary_file}", flush=True)


def select_jobs(project_root: str, requested: list[str], checkpoint_index: int):
    jobs = list(image_model_jobs(project_root, {"DCAE", "LIC-HPCM"}))
    selected = []
    requested_set = set(requested)
    groups = {
        "DCAE": [j for j in jobs if j.model_name == "DCAE"],
        "HPCM-base": [j for j in jobs if j.model_name == "LIC-HPCM" and "-base_" in j.model_id],
        "HPCM-large": [j for j in jobs if j.model_name == "LIC-HPCM" and "-large_" in j.model_id],
    }
    for name, group in groups.items():
        if name not in requested_set and ("LIC-HPCM" not in requested_set or not name.startswith("HPCM-")):
            continue
        if checkpoint_index < 1 or checkpoint_index > len(group):
            raise ValueError(f"{name} has {len(group)} checkpoints; cannot select index {checkpoint_index}")
        selected.append(group[checkpoint_index - 1])
    return selected


def run_pca_eb(sample_runs, job, params, image_eval_mode, eb, device, caesar_root, lpips_fn):
    eb_rows = []
    errors = []
    for run in sample_runs:
        try:
            pca_row = apply_caesar_pca_to_run(
                run,
                eb=float(eb),
                device=device,
                caesar_root=caesar_root,
                lpips_fn=lpips_fn,
            )
        except Exception as exc:
            errors.append({"sample_id": run["sample"].sample_id, "error": str(exc)})
            continue
        pca_row.update(common_job_fields(job, params, image_eval_mode))
        pca_row["model_name"] = f"{job.model_name}+CAESAR-PCA"
        pca_row["model_id"] = f"{job.model_id}_caesar_pca_eb{eb:g}"
        pca_row["label"] = pca_curve_label(job, enabled=True)
        eb_rows.append(pca_row)
    if not eb_rows:
        return {
            **common_job_fields(job, params, image_eval_mode),
            "model_name": f"{job.model_name}+CAESAR-PCA",
            "model_id": f"{job.model_id}_caesar_pca_eb{eb:g}",
            "label": pca_curve_label(job, enabled=True),
            "pca_eb": float(eb),
            "pca_postprocess": "caesar_pca",
            "error": f"all samples failed for eb={eb:g}",
            "sample_errors": errors,
        }
    agg = aggregate_rows(eb_rows, f"{job.model_id}_caesar_pca_eb{eb:g}")
    agg.update(common_job_fields(job, params, image_eval_mode))
    agg["model_name"] = f"{job.model_name}+CAESAR-PCA"
    agg["model_id"] = f"{job.model_id}_caesar_pca_eb{eb:g}"
    agg["label"] = pca_curve_label(job, enabled=True)
    agg["pca_eb"] = float(eb)
    agg["pca_postprocess"] = "caesar_pca"
    agg["success_count"] = len(eb_rows)
    agg["error_count"] = len(errors)
    if errors:
        agg["sample_errors"] = errors[:10]
    return agg


def select_even_log_bpp(rows: list[dict], n_points: int) -> list[dict]:
    valid = [r for r in rows if isinstance(r.get("bpp"), (int, float)) and r["bpp"] > 0]
    valid = sorted(valid, key=lambda r: (float(r["bpp"]), float(r.get("pca_eb", 0.0))))
    unique: list[dict] = []
    for row in valid:
        bpp_value = float(row["bpp"])
        if unique and abs(bpp_value - float(unique[-1]["bpp"])) <= max(1e-9, 1e-4 * bpp_value):
            continue
        unique.append(row)
    valid = unique
    valid = keep_right_edge_of_left_plateau(valid)
    if len(valid) <= n_points:
        return valid
    log_bpps = [float(np.log(float(r["bpp"]))) for r in valid]
    selected = {0, len(valid) - 1}
    while len(selected) < n_points:
        ordered = sorted(selected)
        best_idx = None
        best_score = -1.0
        for left, right in zip(ordered, ordered[1:]):
            if right - left <= 1:
                continue
            midpoint = 0.5 * (log_bpps[left] + log_bpps[right])
            candidates = range(left + 1, right)
            idx = min(candidates, key=lambda i: abs(log_bpps[i] - midpoint))
            score = log_bpps[right] - log_bpps[left]
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected.add(best_idx)
    return [valid[i] for i in sorted(selected, key=lambda idx: float(valid[idx]["bpp"]))]


def keep_right_edge_of_left_plateau(rows: list[dict], psnr_eps: float = 0.25) -> list[dict]:
    """Drop the no-op left plateau while keeping its right edge.

    Large EB values often add zero PCA bytes, producing several nearly identical
    low-bpp points.  Keeping the right edge preserves where PCA starts to matter
    and frees slots for smaller-EB/high-bpp points.
    """
    if len(rows) <= 1 or not isinstance(rows[0].get("psnr"), (int, float)):
        return rows
    base_psnr = float(rows[0]["psnr"])
    right_edge = 0
    for idx, row in enumerate(rows):
        psnr = row.get("psnr")
        if not isinstance(psnr, (int, float)):
            break
        if float(psnr) <= base_psnr + psnr_eps:
            right_edge = idx
            continue
        break
    if right_edge <= 0:
        return rows
    return [rows[right_edge], *rows[right_edge + 1:]]


def run_base_image_codec(sample, codec, lpips_fn, valid_mask, mask_metrics):
    groups = build_image_groups(sample)
    t0 = time.time()
    results = [codec.roundtrip(group.tensor) for group in groups]
    wall = time.time() - t0
    reconstruction = reconstruct_from_groups(groups, [result.reconstruction for result in results])
    bitstream_bytes = int(sum(result.bitstream_bytes for result in results))
    side_info_bytes = int(sum(_normalization_side_info_bytes(group.normalization, group.actual_channels) for group in groups))
    encode_time = float(sum(result.encode_time for result in results))
    decode_time = float(sum(result.decode_time for result in results))
    extras = dict(mask_metrics or {})
    if lpips_fn is not None:
        extras["lpips"] = lpips_fn(sample.array, reconstruction)
    extras.update(torch_memory_usage_mb())
    metrics = base_metrics(
        sample.array,
        reconstruction,
        bitstream_bytes,
        (encode_time, decode_time),
        group_count=len(groups),
        side_info_bytes=side_info_bytes,
        valid_mask=valid_mask,
        extra_metrics=extras,
    )
    metrics.update({
        "dataset_id": sample.dataset_id,
        "sample_id": sample.sample_id,
        "sample_kind": sample.kind,
        "groups": len(groups),
        "shape": list(sample.array.shape),
        "voxel_count": int(sample.array.size),
        "original_min": float(np.min(sample.array)),
        "original_max": float(np.max(sample.array)),
        "sample_wall_time_total": wall,
    })
    return {
        "sample": sample,
        "valid_mask": valid_mask,
        "base_reconstruction": reconstruction,
        "base_bitstream_bytes": bitstream_bytes,
        "base_side_info_bytes": side_info_bytes,
        "base_encode_time": encode_time,
        "base_decode_time": decode_time,
        "groups": len(groups),
        "base_metrics": metrics,
    }


def apply_caesar_pca_to_run(run, eb: float, device: str, caesar_root: Path, lpips_fn):
    sample = run["sample"]
    pca_recon, pca_bytes, pca_encode, pca_decode = caesar_pca_postprocess(
        sample.array,
        run["base_reconstruction"],
        eb=eb,
        device=device,
        caesar_root=caesar_root,
    )
    bitstream_bytes = int(run["base_bitstream_bytes"] + pca_bytes)
    extras = {
        "base_bitstream_bytes": int(run["base_bitstream_bytes"]),
        "base_scientific_bpp": float(run["base_bitstream_bytes"] * 8.0 / sample.array.size),
        "pca_bytes": int(pca_bytes),
        "pca_eb": float(eb),
        "pca_encode_time": float(pca_encode),
        "pca_decode_time": float(pca_decode),
        "pca_postprocess": "caesar_pca",
    }
    if lpips_fn is not None:
        extras["lpips"] = lpips_fn(sample.array, pca_recon)
    metrics = base_metrics(
        sample.array,
        pca_recon,
        bitstream_bytes,
        (run["base_encode_time"] + pca_encode, run["base_decode_time"] + pca_decode),
        group_count=run["groups"],
        side_info_bytes=run["base_side_info_bytes"],
        valid_mask=run["valid_mask"],
        extra_metrics=extras,
    )
    metrics.update({
        "dataset_id": sample.dataset_id,
        "sample_id": sample.sample_id,
        "sample_kind": sample.kind,
        "groups": run["groups"],
        "shape": list(sample.array.shape),
        "voxel_count": int(sample.array.size),
        "original_min": float(np.min(sample.array)),
        "original_max": float(np.max(sample.array)),
    })
    return metrics


def caesar_pca_postprocess(original, reconstruction, eb: float, device: str, caesar_root: Path):
    if str(caesar_root) not in sys.path:
        sys.path.insert(0, str(caesar_root))
    from CAESAR.models.run_gae_cuda import PCACompressor

    original_np = np.asarray(original, dtype=np.float32)
    recon_np = np.asarray(reconstruction, dtype=np.float32)
    padded_original, crop = pad_last2_to_multiple(original_np, 8)
    padded_recon, _ = pad_last2_to_multiple(recon_np, 8)

    original_t = torch.from_numpy(np.ascontiguousarray(padded_original)).to(device)
    recon_t = torch.from_numpy(np.ascontiguousarray(padded_recon)).to(device)
    offset = original_t.mean()
    scale = original_t.max() - original_t.min()
    if float(scale.detach().cpu()) < 1e-8:
        scale = torch.ones_like(scale)
    original_norm = (original_t - offset) / scale
    recon_norm = (recon_t - offset) / scale

    compressor = PCACompressor(eb, 2, codec_algorithm="Zstd", device=device)
    t0 = time.time()
    try:
        meta, compressed, pca_bytes = compressor.compress(original_norm, recon_norm)
    except torch._C._LinAlgError:
        torch.cuda.empty_cache() if str(device).startswith("cuda") else None
        meta, compressed, pca_bytes = compressor.compress(original_norm.double(), recon_norm.double())
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.time()
    if pca_bytes > 0:
        recon_norm = compressor.decompress(recon_norm, meta, compressed, to_np=False)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    t2 = time.time()
    scale_cpu = scale.detach().cpu()
    offset_cpu = offset.detach().cpu()
    restored = (recon_norm.detach().cpu() * scale_cpu + offset_cpu).numpy()
    restored = crop_last2(restored, crop)
    return restored.astype(np.float32, copy=False), int(pca_bytes), t1 - t0, t2 - t1


def pad_last2_to_multiple(array: np.ndarray, multiple: int):
    h, w = array.shape[-2], array.shape[-1]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return array, (h, w)
    pad_width = [(0, 0)] * array.ndim
    pad_width[-2] = (0, pad_h)
    pad_width[-1] = (0, pad_w)
    return np.pad(array, pad_width, mode="edge"), (h, w)


def crop_last2(array: np.ndarray, crop: tuple[int, int]) -> np.ndarray:
    h, w = crop
    return array[..., :h, :w]


def aggregate_rows(rows: list[dict], sample_id: str) -> dict:
    if len(rows) == 1:
        row = dict(rows[0])
        row["sample_id"] = rows[0].get("sample_id")
        row["aggregated_samples"] = 1
        return row
    first = dict(rows[0])
    voxel_counts = np.array([int(r.get("voxel_count", np.prod(r.get("shape", [1])))) for r in rows], dtype=np.float64)
    voxel_counts = np.maximum(voxel_counts, 1)
    total_voxels = float(np.sum(voxel_counts))
    mse = float(np.sum([float(r["mse"]) * v for r, v in zip(rows, voxel_counts)]) / total_voxels)
    data_min = min(float(r.get("original_min", 0.0)) for r in rows)
    data_max = max(float(r.get("original_max", 1.0)) for r in rows)
    data_range = max(data_max - data_min, 1e-8)
    bitstream_bytes = int(sum(int(r.get("bitstream_bytes", 0)) for r in rows))
    side_info_bytes = int(sum(int(r.get("side_info_bytes", 0)) for r in rows))
    total_bytes = bitstream_bytes + side_info_bytes
    original_bytes = int(sum(int(r.get("original_bytes", 0)) for r in rows))
    encode_time = float(sum(float(r.get("encode_time_avg", 0.0)) for r in rows))
    decode_time = float(sum(float(r.get("decode_time_avg", 0.0)) for r in rows))
    first.update({
        "sample_id": sample_id,
        "aggregated_samples": len(rows),
        "sample_ids": [r.get("sample_id") for r in rows],
        "shape": [r.get("shape") for r in rows],
        "voxel_count": int(total_voxels),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "psnr": float("inf") if mse < 1e-30 else float(10 * math.log10((data_range * data_range) / mse)),
        "average_variable_psnr": mean_present(rows, "average_variable_psnr"),
        "average_frame_psnr": mean_present(rows, "average_frame_psnr"),
        "lpips": mean_present(rows, "lpips"),
        "bitstream_bytes": bitstream_bytes,
        "side_info_bytes": side_info_bytes,
        "total_bytes_with_side_info": total_bytes,
        "original_bytes": original_bytes,
        "bpp": bitstream_bytes * 8.0 / total_voxels,
        "scientific_bpp": bitstream_bytes * 8.0 / total_voxels,
        "scientific_bpp_with_side_info": total_bytes * 8.0 / total_voxels,
        "compression_ratio": original_bytes / total_bytes if total_bytes > 0 else float("inf"),
        "encode_time_avg": encode_time,
        "decode_time_avg": decode_time,
        "encode_time_total": encode_time,
        "decode_time_total": decode_time,
        "encode_throughput": original_bytes / encode_time if encode_time > 0 else None,
        "decode_throughput": original_bytes / decode_time if decode_time > 0 else None,
        "memory_usage_MB": max((r.get("memory_usage_MB") or 0.0) for r in rows),
        "memory_reserved_MB": max((r.get("memory_reserved_MB") or 0.0) for r in rows),
        "pca_bytes": int(sum(int(r.get("pca_bytes", 0)) for r in rows)),
        "base_bitstream_bytes": int(sum(int(r.get("base_bitstream_bytes", 0)) for r in rows)),
    })
    return first


def mean_present(rows: list[dict], key: str):
    values = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float)) and math.isfinite(float(r[key]))]
    return float(np.mean(values)) if values else None


def common_job_fields(job, params: int, image_eval_mode: str) -> dict:
    return {
        "model_name": job.model_name,
        "model_id": job.model_id,
        "metric": job.metric,
        "checkpoint": job.checkpoint,
        "params": params,
        "image_eval_mode": image_eval_mode,
    }


def pca_curve_label(job, enabled: bool) -> str:
    if job.model_name == "DCAE":
        base = "DCAE"
    elif "large" in job.model_id:
        base = "HPCM-large"
    else:
        base = "HPCM-base"
    return f"{base}+CAESAR-PCA" if enabled else f"{base} third ckpt"


if __name__ == "__main__":
    main()
