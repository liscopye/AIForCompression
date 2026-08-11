#!/usr/bin/env python3
"""Evaluate selected ERA5-finetuned CAESAR checkpoints across error bounds.

The default checkpoint set samples the completed CAESAR-V run and the
CAESAR-D run that continued after update 140000.
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "scripts" / "eval_caesar_lysozyme.py"
CKPT_DIR = ROOT / "checkpoints" / "caesar"
DEFAULT_EBS = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

CHECKPOINTS = {
    "V": [
        ("original", "caesar_v.pt"),
        ("update20k", "caesar_v_tuning_era5_update20000.pt"),
        ("update40k", "caesar_v_tuning_era5_update40000.pt"),
        ("update60k", "caesar_v_tuning_era5_update60000.pt"),
        ("update80k", "caesar_v_tuning_era5_update80000.pt"),
        ("update100k", "caesar_v_tuning_era5_update100000.pt"),
        ("best", "caesar_v_tuning_era5.pt"),
    ],
    "D": [
        ("original", "caesar_d.pt"),
        ("update40k", "caesar_d_tuning_era5_update40000.pt"),
        ("update80k", "caesar_d_tuning_era5_update80000.pt"),
        ("update120k", "caesar_d_tuning_era5_update120000.pt"),
        ("update160k_resume", "caesar_d_tuning_era5_resume_from140000_update20000.pt"),
        ("update200k_resume", "caesar_d_tuning_era5_resume_from140000_update60000.pt"),
        ("best", "caesar_d_tuning_era5.pt"),
    ],
}


def eb_label(eb: float) -> str:
    return f"{eb:.0e}".replace("e-0", "e-").replace("e-", "em")


def decorate_metrics(metrics: dict, model_type: str, label: str, checkpoint: Path, eb: float) -> dict:
    metrics.update({
        "model_id": f"caesar_{model_type.lower()}",
        "model_type": model_type,
        "checkpoint_label": label,
        "checkpoint": str(checkpoint),
        "eb": eb,
    })
    return metrics


def run_one(model_type: str, label: str, checkpoint: Path, eb: float, output_root: Path,
            gpu: str, max_blocks: int, batch_size: int, reuse_existing: bool) -> dict:
    out_dir = output_root / model_type / f"eb_{eb_label(eb)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    test_data = Path("/workspace/Data/ERA5/finetune_processed/test_blocks") / (
        "era5_test_V_nf8.npz" if model_type == "V" else "era5_test_D_nf16.npz"
    )
    variant_label = f"era5_{label}"
    json_path = out_dir / f"CAESAR-{model_type}_{variant_label}.json"
    log_path = out_dir / f"CAESAR-{model_type}_{variant_label}.log"
    if reuse_existing and json_path.exists():
        return decorate_metrics(json.loads(json_path.read_text()), model_type, label, checkpoint, eb)

    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--model_type", model_type,
        "--checkpoint", str(checkpoint),
        "--variant_name", variant_label,
        "--test_data", str(test_data),
        "--device", "cuda:0",
        "--output_dir", str(out_dir),
        "--eb", str(eb),
        "--batch_size", str(batch_size),
        "--max_blocks", str(max_blocks),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    log_path.write_text(result.stdout + "\nSTDERR:\n" + result.stderr)
    if result.returncode != 0 or not json_path.exists():
        raise RuntimeError(f"{model_type} {label} eb={eb:g} failed; see {log_path}")

    return decorate_metrics(json.loads(json_path.read_text()), model_type, label, checkpoint, eb)


def run_gpu_queue(gpu: str, jobs: list[tuple], output_root: Path, args: argparse.Namespace) -> list[dict]:
    results = []
    for model_type, label, checkpoint, eb in jobs:
        max_blocks = args.max_blocks_v if model_type == "V" else args.max_blocks_d
        batch_size = args.batch_size_v if model_type == "V" else args.batch_size_d
        print(f"[GPU {gpu}] CAESAR-{model_type} {label} eb={eb:g}", flush=True)
        results.append(run_one(model_type, label, checkpoint, eb, output_root,
                               gpu, max_blocks, batch_size, args.reuse_existing))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep CAESAR ERA5 intermediate checkpoints.")
    parser.add_argument("--model_type", choices=["V", "D", "both"], default="both")
    parser.add_argument("--gpus", default="0", help="Comma-separated CUDA_VISIBLE_DEVICES ids.")
    parser.add_argument("--output_dir", type=Path,
                        default=ROOT / "results" / "caesar_era5_checkpoint_sweep")
    parser.add_argument("--eb", type=float, nargs="+", default=DEFAULT_EBS)
    parser.add_argument("--max_blocks_v", type=int, default=50)
    parser.add_argument("--max_blocks_d", type=int, default=20)
    parser.add_argument("--batch_size_v", type=int, default=32)
    parser.add_argument("--batch_size_d", type=int, default=8)
    parser.add_argument("--reuse_existing", action="store_true",
                        help="Load existing per-checkpoint result JSONs instead of rerunning them.")
    args = parser.parse_args()

    model_types = ["V", "D"] if args.model_type == "both" else [args.model_type]
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id.")

    jobs = []
    manifest = {}
    for model_type in model_types:
        manifest[model_type] = []
        for label, filename in CHECKPOINTS[model_type]:
            checkpoint = CKPT_DIR / filename
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            manifest[model_type].append({"label": label, "checkpoint": str(checkpoint)})
            for eb in args.eb:
                jobs.append((model_type, label, checkpoint, eb))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2))
    queues = [jobs[i::len(gpus)] for i in range(len(gpus))]
    all_results = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(run_gpu_queue, gpu, queue, args.output_dir, args)
            for gpu, queue in zip(gpus, queues) if queue
        ]
        for future in futures:
            all_results.extend(future.result())

    label_order = {
        model_type: {label: idx for idx, (label, _) in enumerate(CHECKPOINTS[model_type])}
        for model_type in model_types
    }
    all_results.sort(
        key=lambda r: (
            model_types.index(r["model_type"]),
            label_order[r["model_type"]][r["checkpoint_label"]],
            r["eb"],
        )
    )
    for model_type in model_types:
        subset = [r for r in all_results if r["model_type"] == model_type]
        out = args.output_dir / model_type / "sweep_results.json"
        out.write_text(json.dumps(subset, indent=2))
        print(f"Saved {len(subset)} results: {out}")
    (args.output_dir / "sweep_results.json").write_text(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
