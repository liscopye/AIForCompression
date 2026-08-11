#!/usr/bin/env python3
"""Evaluate the early ERA5 fine-tune trajectory for CAESAR-V and CAESAR-D.

For CAESAR-D stage 1, VAE-only checkpoints are paired with the original
pre-tune diffusion weights, matching the model available before stage 2 starts.
"""

import argparse
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

try:
    from sweep_caesar_era5_checkpoints import (
        CKPT_DIR, DEFAULT_EBS, ROOT, decorate_metrics, eb_label, run_one,
    )
except ModuleNotFoundError:
    from scripts.sweep_caesar_era5_checkpoints import (
        CKPT_DIR, DEFAULT_EBS, ROOT, decorate_metrics, eb_label, run_one,
    )


PRIOR_RESULTS = ROOT / "results" / "caesar_era5_checkpoint_sweep"


def v_specs() -> list[tuple[str, Path, str | None]]:
    return [
        ("original", CKPT_DIR / "caesar_v.pt", "original"),
        ("update10k", CKPT_DIR / "caesar_v_tuning_era5_update10000.pt", None),
        ("update20k", CKPT_DIR / "caesar_v_tuning_era5_update20000.pt", "update20k"),
    ]


def d_specs(derived_dir: Path) -> list[tuple[str, Path, str | None]]:
    specs = [("original", CKPT_DIR / "caesar_d.pt", "original")]
    original = torch.load(CKPT_DIR / "caesar_d.pt", map_location="cpu", weights_only=False)
    vae_sources = [
        (f"s1_vae_{step}k", CKPT_DIR / f"caesar_d_tuning_era5_vae_update{step}000.pt")
        for step in range(10, 101, 10)
    ] + [("s1_vae_best", CKPT_DIR / "caesar_d_tuning_era5_vae.pt")]
    for label, vae_path in vae_sources:
        full_path = derived_dir / f"{label}.pt"
        vae = torch.load(vae_path, map_location="cpu", weights_only=False)
        torch.save({"vae": vae, "diffusion": original["diffusion"]}, full_path)
        specs.append((label, full_path, None))
    for step in range(10, 41, 10):
        reuse = "update40k" if step == 40 else None
        specs.append((f"s2_diff_{step}k", CKPT_DIR / f"caesar_d_tuning_era5_update{step}000.pt", reuse))
    return specs


def reuse_prior(model_type: str, old_label: str, new_label: str, checkpoint: str,
                eb: float, output_dir: Path) -> dict | None:
    src = (PRIOR_RESULTS / model_type / f"eb_{eb_label(eb)}" /
           f"CAESAR-{model_type}_era5_{old_label}.json")
    if not src.exists():
        return None
    metrics = decorate_metrics(json.loads(src.read_text()), model_type, new_label, Path(checkpoint), eb)
    out_dir = output_dir / model_type / f"eb_{eb_label(eb)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"CAESAR-{model_type}_era5_{new_label}.json").write_text(
        json.dumps(metrics, indent=2)
    )
    return metrics


def run_job(job: tuple, gpu: str, output_dir: Path, args: argparse.Namespace) -> dict:
    model_type, label, checkpoint, old_label, eb, source_checkpoint = job
    if old_label:
        reused = reuse_prior(model_type, old_label, label, source_checkpoint, eb, output_dir)
        if reused is not None:
            return reused
    max_blocks = args.max_blocks_v if model_type == "V" else args.max_blocks_d
    batch_size = args.batch_size_v if model_type == "V" else args.batch_size_d
    metrics = run_one(model_type, label, checkpoint, eb, output_dir, gpu,
                      max_blocks, batch_size, reuse_existing=True)
    metrics["checkpoint"] = source_checkpoint
    if label.startswith("s1_vae_"):
        metrics["paired_diffusion"] = str(CKPT_DIR / "caesar_d.pt")
    return metrics


def run_queue(queue: list[tuple], gpu: str, output_dir: Path, args: argparse.Namespace) -> list[dict]:
    results = []
    for job in queue:
        model_type, label, _, old_label, eb, _ = job
        state = "reuse" if old_label else "run"
        print(f"[GPU {gpu}] {state} CAESAR-{model_type} {label} eb={eb:g}", flush=True)
        results.append(run_job(job, gpu, output_dir, args))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep CAESAR ERA5 early fine-tune trajectory.")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--output_dir", type=Path,
                        default=ROOT / "results" / "caesar_era5_early_transition")
    parser.add_argument("--eb", type=float, nargs="+", default=DEFAULT_EBS)
    parser.add_argument("--max_blocks_v", type=int, default=50)
    parser.add_argument("--max_blocks_d", type=int, default=20)
    parser.add_argument("--batch_size_v", type=int, default=32)
    parser.add_argument("--batch_size_d", type=int, default=8)
    args = parser.parse_args()
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="caesar_d_s1_", dir=args.output_dir) as tmp:
        derived_dir = Path(tmp)
        specifications = {"V": v_specs(), "D": d_specs(derived_dir)}
        manifest = {"V": [], "D": []}
        jobs = []
        for model_type, specs in specifications.items():
            for label, checkpoint, old_label in specs:
                source = str(checkpoint)
                if label.startswith("s1_vae_"):
                    filename = label.replace("s1_vae_", "")
                    source = str(
                        CKPT_DIR / ("caesar_d_tuning_era5_vae.pt" if filename == "best"
                                    else f"caesar_d_tuning_era5_vae_update{filename[:-1]}000.pt")
                    )
                manifest[model_type].append({
                    "label": label,
                    "checkpoint": source,
                    "paired_diffusion": str(CKPT_DIR / "caesar_d.pt")
                    if label.startswith("s1_vae_") else None,
                })
                for eb in args.eb:
                    jobs.append((model_type, label, checkpoint, old_label, eb, source))
        (args.output_dir / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2))
        queues = [jobs[i::len(gpus)] for i in range(len(gpus))]
        all_results = []
        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures = [
                executor.submit(run_queue, queue, gpu, args.output_dir, args)
                for gpu, queue in zip(gpus, queues) if queue
            ]
            for future in futures:
                all_results.extend(future.result())

    order = {
        mt: {label: idx for idx, (label, _, _) in enumerate(specs)}
        for mt, specs in specifications.items()
    }
    all_results.sort(key=lambda r: ("VD".index(r["model_type"]),
                                    order[r["model_type"]][r["checkpoint_label"]], r["eb"]))
    for mt in "VD":
        subset = [r for r in all_results if r["model_type"] == mt]
        path = args.output_dir / mt / "sweep_results.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(subset, indent=2))
        print(f"Saved {len(subset)} results: {path}")


if __name__ == "__main__":
    main()
