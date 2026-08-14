#!/usr/bin/env python3
"""Run Objective-v1 DCAE/HPCM + CAESAR-PCA ablations."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression_pipeline.metrics import base_metrics, make_lpips_fn, torch_memory_usage_mb
from compression_pipeline.model_registry import image_model_jobs
from compression_pipeline.objective_data import checksum, load_normalization, load_objective_samples
from compression_pipeline.torch_codecs import CompressAILikeCodec
from scripts.run_objective_benchmark import (
    finalize_row,
    hardware_manifest,
    image_point,
    objective_fields,
    repeated_roundtrip,
)


DEFAULT_EBS = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--input-root", type=Path, default=Path("unified_results/objective_all_to_all_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/tmp/aifc_objective_pca_hybrids"))
    parser.add_argument("--models", nargs="+", default=["DCAE", "HPCM-base", "HPCM-large"])
    parser.add_argument("--sample-id-contains", nargs="+", default=None)
    parser.add_argument("--checkpoint-index", type=int, default=3)
    parser.add_argument("--pca-eb", type=float, nargs="+", default=DEFAULT_EBS)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--image-batch-size", type=int, default=1)
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def select_jobs(requested: list[str], checkpoint_index: int):
    jobs = list(image_model_jobs(ROOT, {"DCAE", "LIC-HPCM"}))
    groups = {
        "DCAE": [job for job in jobs if job.model_name == "DCAE"],
        "HPCM-base": [job for job in jobs if job.model_name == "LIC-HPCM" and "-base_" in job.model_id],
        "HPCM-large": [job for job in jobs if job.model_name == "LIC-HPCM" and "-large_" in job.model_id],
    }
    selected = []
    for name in requested:
        group = groups[name]
        if not 1 <= checkpoint_index <= len(group):
            raise ValueError(f"{name} has {len(group)} checkpoints, index {checkpoint_index} is invalid")
        selected.append((name, group[checkpoint_index - 1]))
    return selected


def pad8(array: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = array.shape[-2:]
    pad_h = (-height) % 8
    pad_w = (-width) % 8
    if not pad_h and not pad_w:
        return array, (height, width)
    pads = [(0, 0)] * array.ndim
    pads[-2] = (0, pad_h)
    pads[-1] = (0, pad_w)
    return np.pad(array, pads, mode="edge"), (height, width)


def pca_roundtrip(original: np.ndarray, reconstruction: np.ndarray, eb: float):
    import torch

    caesar_root = ROOT / "models/CAESAR"
    if str(caesar_root) not in sys.path:
        sys.path.insert(0, str(caesar_root))
    from CAESAR.models.run_gae_cuda import PCACompressor

    original_pad, crop = pad8(np.asarray(original, dtype=np.float32))
    recon_pad, _ = pad8(np.asarray(reconstruction, dtype=np.float32))
    original_t = torch.from_numpy(np.ascontiguousarray(original_pad)).cuda()
    recon_t = torch.from_numpy(np.ascontiguousarray(recon_pad)).cuda()
    offset = original_t.mean()
    scale = original_t.max() - original_t.min()
    if float(scale) < 1e-8:
        scale = torch.ones_like(scale)
    original_norm = (original_t - offset) / scale
    recon_norm = (recon_t - offset) / scale
    compressor = PCACompressor(float(eb), 2, codec_algorithm="Zstd", device="cuda")
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        meta, compressed, payload_bytes = compressor.compress(original_norm, recon_norm)
    except torch._C._LinAlgError:
        torch.cuda.empty_cache()
        meta, compressed, payload_bytes = compressor.compress(original_norm.double(), recon_norm.double())
    torch.cuda.synchronize()
    encoded = time.perf_counter() - started
    started = time.perf_counter()
    if payload_bytes:
        recon_norm = compressor.decompress(recon_norm, meta, compressed, to_np=False)
    torch.cuda.synchronize()
    decoded = time.perf_counter() - started
    restored = (recon_norm.detach().cpu() * scale.detach().cpu() + offset.detach().cpu()).numpy()
    height, width = crop
    # Two float32 values are required to decode the PCA residual normalization.
    side_bytes = 8
    return restored[..., :height, :width].astype(np.float32, copy=False), int(payload_bytes), side_bytes, encoded, decoded


def hybrid_point(original, mask, base, eb: float, lpips_fn=None):
    started = time.perf_counter()
    reconstruction, pca_bytes, pca_side_bytes, pca_encode, pca_decode = pca_roundtrip(
        original, base["reconstruction"], eb
    )
    wall = base["wall"] + (time.perf_counter() - started)
    extras = {
        "lpips": lpips_fn(original, reconstruction) if lpips_fn is not None else None,
        "base_bitstream_bytes": base["bitstream_bytes"],
        "pca_bytes": pca_bytes,
        "pca_side_info_bytes": pca_side_bytes,
        "pca_eb": float(eb),
        "pca_encode_time": pca_encode,
        "pca_decode_time": pca_decode,
        "pca_postprocess": "caesar_pca",
        **torch_memory_usage_mb("cuda"),
    }
    row = base_metrics(
        original,
        reconstruction,
        base["bitstream_bytes"] + pca_bytes,
        (base["encode_time"] + pca_encode, base["decode_time"] + pca_decode),
        group_count=base["partition_count"],
        side_info_bytes=pca_side_bytes,
        valid_mask=mask,
        extra_metrics=extras,
    )
    row.update({
        "sample_wall_time_total": wall,
        "sample_wall_throughput_MBps": row["original_bytes"] / wall / 1e6,
        "partition_count": base["partition_count"],
        "partition_policy": base["partition_policy"],
    })
    return row


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(f"CUDA unavailable on physical GPU {args.gpu}")
    if args.smoke:
        args.warmups, args.repeats = 0, 1
        args.pca_eb = [args.pca_eb[len(args.pca_eb) // 2]]
        args.models = args.models[:1]

    input_dir = args.input_root / args.dataset
    manifests = json.loads((input_dir / "samples.json").read_text())
    manifest_by_id = {item["canonical_sample_id"]: item for item in manifests}
    normalization = load_normalization(input_dir / "normalization.json")
    samples = load_objective_samples(args.dataset)
    if args.sample_id_contains:
        samples = [
            sample for sample in samples
            if any(token in sample.sample_id for token in args.sample_id_contains)
        ]
        if not samples:
            raise ValueError(f"No samples match {args.sample_id_contains}")
    normalized_samples = [normalization.normalize(sample.raw) for sample in samples]
    manifests = [manifest_by_id[sample.sample_id] for sample in samples]
    for sample, normalized, manifest in zip(samples, normalized_samples, manifests):
        if checksum(sample.raw, sample.mask) != manifest["canonical_sha256"]:
            raise ValueError(f"Raw checksum changed for {sample.sample_id}")
        if checksum(normalized, sample.mask) != manifest["normalized_canonical_sha256"]:
            raise ValueError(f"Normalized checksum changed for {sample.sample_id}")

    output_dir = args.output_root / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    rows = [] if args.force or not summary_path.exists() else json.loads(summary_path.read_text())
    done = {(row.get("model_id"), row.get("canonical_sample_id"), row.get("control")) for row in rows if "error" not in row}
    lpips_fn = None if args.no_lpips else make_lpips_fn("cuda", max_image_pairs=32)
    hardware = hardware_manifest(args.gpu)

    for label, job in select_jobs(args.models, args.checkpoint_index):
        expected_keys = {
            (f"{job.model_id}_caesar_pca_eb{eb:g}", sample.sample_id, float(eb))
            for sample in samples for eb in args.pca_eb
        }
        if expected_keys <= done:
            print(f"[skip-model] {label}: all requested samples and EB points already exist", flush=True)
            continue
        print(f"[load] {label}: {job.model_id}", flush=True)
        model = job.loader("cuda")
        codec_cls = job.codec_cls or CompressAILikeCodec
        codec = codec_cls(model, device="cuda", divisor=job.divisor, **job.codec_kwargs)
        try:
            for sample, normalized, manifest in zip(samples, normalized_samples, manifests):
                base_row = image_point(
                    codec, args.dataset, sample.sample_id, normalized, sample.mask,
                    args.image_batch_size, None, return_reconstruction=True,
                )
                reconstruction = base_row.pop("_reconstruction")
                base = {
                    "reconstruction": reconstruction,
                    "bitstream_bytes": int(base_row["bitstream_bytes"]),
                    "encode_time": float(base_row["encode_time_total"]),
                    "decode_time": float(base_row["decode_time_total"]),
                    "wall": float(base_row["sample_wall_time_total"]),
                    "partition_count": int(base_row["partition_count"]),
                    "partition_policy": base_row["partition_policy"],
                }
                fields = objective_fields(manifest, manifest["external_input_manifest"], hardware, "pca_hybrid_ablation")
                if args.dataset == "lysozyme":
                    external = dict(fields["external_input_manifest"])
                    external.update({
                        "validity_mask_policy": "shared_benchmark_metadata",
                        "validity_mask_rate_bytes": 0,
                    })
                    fields["external_input_manifest"] = external
                for eb in args.pca_eb:
                    model_id = f"{job.model_id}_caesar_pca_eb{eb:g}"
                    key = (model_id, sample.sample_id, float(eb))
                    if key in done:
                        print(f"[skip] {key}", flush=True)
                        continue
                    row = repeated_roundtrip(
                        lambda e=eb: hybrid_point(normalized, sample.mask, base, e),
                        args.warmups,
                        args.repeats,
                        metric_call=(
                            lambda e=eb: hybrid_point(normalized, sample.mask, base, e, lpips_fn)
                        ) if lpips_fn is not None else None,
                    )
                    row.update({
                        "model_name": f"{label}+CAESAR-PCA",
                        "model_id": model_id,
                        "control": float(eb),
                        "pca_eb": float(eb),
                        "checkpoint": job.checkpoint,
                        "params": sum(parameter.numel() for parameter in model.parameters()),
                        "checkpoint_index": args.checkpoint_index,
                        "ablation_track": "image_codec_plus_caesar_pca",
                    })
                    row = finalize_row(row, fields)
                    row["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    rows.append(row)
                    done.add(key)
                    summary_path.write_text(json.dumps(rows, indent=2))
                    print(f"[result] {model_id} {sample.sample_id} bpp={row['scientific_bpp_with_side_info']:.5g} psnr={row['psnr']:.3f}", flush=True)
        finally:
            del codec, model
            gc.collect()
            torch.cuda.empty_cache()
    print(summary_path)


if __name__ == "__main__":
    main()
