#!/usr/bin/env python3
"""Select CAESAR hourly fine-tuning checkpoints from held-out RD sweeps."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("unified_results/caesar_era5_hourly_pilot_eval"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/caesar_era5_hourly_pilot"),
    )
    parser.add_argument(
        "--original-dir",
        type=Path,
        default=Path("checkpoints/caesar"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("unified_results/caesar_era5_hourly_selection"),
    )
    parser.add_argument(
        "--selected-checkpoint-dir",
        type=Path,
        default=Path("checkpoints/caesar_era5_hourly_selected"),
    )
    return parser.parse_args()


def load_curve(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    curve = [
        row
        for row in rows
        if not row.get("error")
        and math.isfinite(float(row["scientific_bpp"]))
        and float(row["scientific_bpp"]) > 0
        and math.isfinite(float(row["psnr"]))
    ]
    if len(curve) < 3:
        raise ValueError(f"Need at least three valid RD points in {path}, got {len(curve)}")
    curve.sort(key=lambda row: float(row["psnr"]))
    rates = [float(row["scientific_bpp"]) for row in curve]
    qualities = [float(row["psnr"]) for row in curve]
    if any(b <= a for a, b in zip(rates, rates[1:])):
        raise ValueError(f"Non-monotonic rate curve in {path}: {rates}")
    if any(b <= a for a, b in zip(qualities, qualities[1:])):
        raise ValueError(f"Non-monotonic PSNR curve in {path}: {qualities}")
    return curve


def piecewise_log_bd_rate(reference: list[dict], candidate: list[dict]) -> float:
    ref_quality = np.asarray([float(row["psnr"]) for row in reference])
    ref_log_rate = np.log(
        np.asarray([float(row["scientific_bpp"]) for row in reference])
    )
    candidate_quality = np.asarray([float(row["psnr"]) for row in candidate])
    candidate_log_rate = np.log(
        np.asarray([float(row["scientific_bpp"]) for row in candidate])
    )
    low = max(ref_quality[0], candidate_quality[0])
    high = min(ref_quality[-1], candidate_quality[-1])
    if high <= low:
        raise ValueError("RD curves have no overlapping PSNR interval")
    quality = np.linspace(low, high, 10_000)
    delta = np.interp(quality, candidate_quality, candidate_log_rate) - np.interp(
        quality, ref_quality, ref_log_rate
    )
    mean_delta = np.trapezoid(delta, quality) / (high - low)
    return float((np.exp(mean_delta) - 1.0) * 100.0)


def checkpoint_path(
    model_type: str,
    candidate_name: str,
    checkpoint_dir: Path,
    original_dir: Path,
) -> Path:
    if candidate_name == f"original_{model_type.lower()}":
        return original_dir / f"caesar_{model_type.lower()}.pt"
    if model_type == "V":
        return checkpoint_dir / f"{candidate_name}.pt"
    return checkpoint_dir / "packaged_d" / f"{candidate_name}.pt"


def candidate_norm_type(candidate_name: str) -> str:
    return "mean_range_hw" if "_hw" in candidate_name else "mean_range"


def rank_model(
    model_type: str,
    eval_dir: Path,
    checkpoint_dir: Path,
    original_dir: Path,
) -> tuple[list[dict], dict]:
    prefix = model_type.lower() + "_"
    baseline_name = f"original_{model_type.lower()}"
    baseline = load_curve(eval_dir / baseline_name / "summary.json")
    ranking: list[dict] = []
    rejected: list[dict] = []

    for summary_path in sorted(eval_dir.glob(f"{prefix}*/summary.json")):
        name = summary_path.parent.name
        try:
            curve = load_curve(summary_path)
            bd_rate = piecewise_log_bd_rate(baseline, curve)
            checkpoint = checkpoint_path(
                model_type, name, checkpoint_dir, original_dir
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            ranking.append(
                {
                    "name": name,
                    "checkpoint": str(checkpoint.resolve()),
                    "norm_type": candidate_norm_type(name),
                    "bd_rate_percent": bd_rate,
                    "curve": curve,
                }
            )
        except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
            rejected.append({"name": name, "reason": str(exc)})

    baseline_checkpoint = checkpoint_path(
        model_type, baseline_name, checkpoint_dir, original_dir
    )
    ranking.append(
        {
            "name": baseline_name,
            "checkpoint": str(baseline_checkpoint.resolve()),
            "norm_type": "mean_range",
            "bd_rate_percent": 0.0,
            "curve": baseline,
        }
    )
    ranking.sort(key=lambda item: item["bd_rate_percent"])
    selected = dict(ranking[0])
    selected["rejected_candidates"] = rejected
    return ranking, selected


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.selected_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "selection_metric": "piecewise log-rate BD-rate over held-out 3-point RD curve",
        "selection_role": "checkpoint screening only; final claim requires objective-v1",
        "models": {},
    }
    for model_type in ("V", "D"):
        ranking, selected = rank_model(
            model_type,
            args.eval_dir,
            args.checkpoint_dir,
            args.original_dir,
        )
        target = args.selected_checkpoint_dir / f"caesar_{model_type.lower()}.pt"
        shutil.copy2(selected["checkpoint"], target)
        output["models"][model_type] = {
            "selected": selected,
            "ranking": ranking,
            "materialized_checkpoint": str(target.resolve()),
        }

    selection_path = args.output_dir / "selection.json"
    selection_path.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    (args.selected_checkpoint_dir / "selection.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(selection_path)
    for model_type, result in output["models"].items():
        selected = result["selected"]
        print(
            f"CAESAR-{model_type}: {selected['name']} "
            f"BD-rate={selected['bd_rate_percent']:.3f}%"
        )


if __name__ == "__main__":
    main()
