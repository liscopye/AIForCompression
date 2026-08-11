#!/usr/bin/env python3
"""Summarize the formal ERA5 RD comparison for stable CAESAR fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("unified_results/objective_all_to_all_v1/era5_npy/summary.json"),
    )
    parser.add_argument(
        "--tuned-v",
        type=Path,
        default=Path(
            "unified_results/objective_v1_era5_tuned_stable_v1000/"
            "era5_npy/summary.json"
        ),
    )
    parser.add_argument(
        "--tuned-d",
        type=Path,
        default=Path(
            "unified_results/objective_v1_era5_tuned_stable_d100/"
            "era5_npy/summary.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("unified_results/caesar_era5_stable_tuning_20260723"),
    )
    parser.add_argument(
        "--selection-json",
        type=Path,
        default=None,
        help="Optional hourly checkpoint selection manifest.",
    )
    parser.add_argument(
        "--require-improvement",
        action="store_true",
        help="Exit non-zero unless both tuned V and D have negative BD-rate.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [row for row in rows if not row.get("error")]


def select_baseline(rows: list[dict], variant: str) -> list[dict]:
    prefix = f"caesar_{variant}-"
    selected = [
        row
        for row in rows
        if row.get("model_name") == "CAESAR"
        and str(row.get("model_id", "")).startswith(prefix)
    ]
    if len(selected) != 7:
        raise ValueError(f"Expected 7 baseline {variant.upper()} rows, got {len(selected)}")
    return selected


def sorted_curve(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: float(row["scientific_bpp"]))


def piecewise_log_bd_rate(reference: list[dict], candidate: list[dict]) -> float:
    reference = sorted(reference, key=lambda row: float(row["psnr"]))
    candidate = sorted(candidate, key=lambda row: float(row["psnr"]))
    low = max(float(reference[0]["psnr"]), float(candidate[0]["psnr"]))
    high = min(float(reference[-1]["psnr"]), float(candidate[-1]["psnr"]))
    if high <= low:
        raise ValueError("RD curves have no shared PSNR interval")

    psnr = np.linspace(low, high, 10_000)
    reference_log_rate = np.interp(
        psnr,
        [float(row["psnr"]) for row in reference],
        np.log([float(row["scientific_bpp"]) for row in reference]),
    )
    candidate_log_rate = np.interp(
        psnr,
        [float(row["psnr"]) for row in candidate],
        np.log([float(row["scientific_bpp"]) for row in candidate]),
    )
    mean_log_delta = np.trapezoid(candidate_log_rate - reference_log_rate, psnr) / (
        high - low
    )
    return float((np.exp(mean_log_delta) - 1.0) * 100.0)


def compact_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "eb": float(row["eb"]),
            "scientific_bpp": float(row["scientific_bpp"]),
            "psnr": float(row["psnr"]),
            "encode_time_avg": float(row["encode_time_avg"]),
            "decode_time_avg": float(row["decode_time_avg"]),
            "sample_wall_time_median": float(row["sample_wall_time_total"]),
        }
        for row in sorted(rows, key=lambda row: float(row["eb"]), reverse=True)
    ]


def improvement_failures(v_bd_rate: float, d_bd_rate: float) -> list[str]:
    failures = []
    if not np.isfinite(v_bd_rate) or v_bd_rate >= 0:
        failures.append(f"CAESAR-V BD-rate did not improve: {v_bd_rate}")
    if not np.isfinite(d_bd_rate) or d_bd_rate >= 0:
        failures.append(f"CAESAR-D BD-rate did not improve: {d_bd_rate}")
    return failures


def validate_curve_norm(rows: list[dict], expected: str, label: str) -> None:
    observed = {row.get("caesar_norm_type") for row in rows}
    if observed != {expected}:
        raise ValueError(
            f"{label} expected caesar_norm_type={expected}, observed={observed}"
        )


def plot_panel(
    axis: plt.Axes,
    baseline: list[dict],
    tuned: list[dict],
    title: str,
    tuned_label: str,
    color: str,
    bd_rate: float,
) -> None:
    baseline = sorted_curve(baseline)
    tuned = sorted_curve(tuned)
    axis.plot(
        [row["scientific_bpp"] for row in baseline],
        [row["psnr"] for row in baseline],
        color="#4B5563",
        marker="o",
        linewidth=2.0,
        markersize=5,
        label="Original",
    )
    axis.plot(
        [row["scientific_bpp"] for row in tuned],
        [row["psnr"] for row in tuned],
        color=color,
        marker="s",
        linewidth=2.2,
        markersize=5,
        label=tuned_label,
    )
    axis.set_xscale("log")
    axis.set_title(f"{title}\nPiecewise log-rate BD-rate: {bd_rate:.2f}%")
    axis.set_xlabel("Scientific BPP")
    axis.grid(True, which="major", color="#D1D5DB", linewidth=0.7)
    axis.grid(True, which="minor", color="#E5E7EB", linewidth=0.45, alpha=0.7)
    axis.legend(frameon=False)


def main() -> None:
    args = parse_args()
    selected_v = (
        "checkpoints/caesar_era5_stability_20260723/"
        "v_mr_lam1e4_anchor0_update1000.pt"
    )
    selected_d = (
        "checkpoints/caesar_era5_stability_20260723/packaged_d/"
        "d_mr_lam1e4_anchor0_update100.pt"
    )
    tuned_v_label = "ERA5 tuned (update 1000)"
    tuned_d_label = "ERA5 tuned (update 100)"
    selected_v_norm = "mean_range"
    selected_d_norm = "mean_range"
    if args.selection_json is not None:
        selection = json.loads(args.selection_json.read_text(encoding="utf-8"))
        selected_v_row = selection["models"]["V"]["selected"]
        selected_d_row = selection["models"]["D"]["selected"]
        selected_v = selected_v_row["checkpoint"]
        selected_d = selected_d_row["checkpoint"]
        selected_v_norm = selected_v_row["norm_type"]
        selected_d_norm = selected_d_row["norm_type"]
        tuned_v_label = f"Hourly tuned ({selected_v_row['name']})"
        tuned_d_label = f"Hourly tuned ({selected_d_row['name']})"

    baseline_rows = load_rows(args.baseline)
    original_v = select_baseline(baseline_rows, "v")
    original_d = select_baseline(baseline_rows, "d")
    tuned_v = load_rows(args.tuned_v)
    tuned_d = load_rows(args.tuned_d)
    if len(tuned_v) != 7 or len(tuned_d) != 7:
        raise ValueError(
            f"Expected 7 tuned rows per model, got V={len(tuned_v)}, D={len(tuned_d)}"
        )
    if args.selection_json is not None:
        validate_curve_norm(tuned_v, selected_v_norm, "CAESAR-V tuned curve")
        validate_curve_norm(tuned_d, selected_d_norm, "CAESAR-D tuned curve")
        validate_curve_norm(original_v, "mean_range", "CAESAR-V original curve")
        validate_curve_norm(original_d, "mean_range", "CAESAR-D original curve")

    v_bd_rate = piecewise_log_bd_rate(original_v, tuned_v)
    d_bd_rate = piecewise_log_bd_rate(original_d, tuned_d)
    gate_failures = improvement_failures(v_bd_rate, d_bd_rate)
    output = {
        "protocol_id": "aifc-objective-v1",
        "dataset_id": "era5_npy",
        "sample_id": "vars000-267_t000-015_crop240",
        "timing": {"warmups": 2, "repeats": 5, "scope": "end_to_end_no_io"},
        "fine_tuning_success_gate": {
            "require_both_negative_bd_rate": bool(args.require_improvement),
            "passed": not gate_failures,
            "failures": gate_failures,
        },
        "caesar_v": {
            "selected_checkpoint": selected_v,
            "norm_type": selected_v_norm,
            "piecewise_log_rate_bd_rate_percent": v_bd_rate,
            "original": compact_rows(original_v),
            "tuned": compact_rows(tuned_v),
        },
        "caesar_d": {
            "selected_checkpoint": selected_d,
            "norm_type": selected_d_norm,
            "piecewise_log_rate_bd_rate_percent": d_bd_rate,
            "original": compact_rows(original_d),
            "tuned": compact_rows(tuned_d),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "figure.dpi": 150,
            "savefig.dpi": 220,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True)
    plot_panel(
        axes[0], original_v, tuned_v, "CAESAR-V", tuned_v_label,
        "#D97706", v_bd_rate,
    )
    plot_panel(
        axes[1], original_d, tuned_d, "CAESAR-D", tuned_d_label,
        "#2563EB", d_bd_rate,
    )
    axes[0].set_ylabel("PSNR (dB)")
    figure.suptitle("ERA5 Stable Fine-tuning: Original vs. Tuned", fontsize=14)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            args.output_dir / f"caesar_era5_stable_rd.{suffix}",
            bbox_inches="tight",
        )
    plt.close(figure)
    print(args.output_dir / "summary.json")
    if args.require_improvement and gate_failures:
        raise RuntimeError("; ".join(gate_failures))


if __name__ == "__main__":
    main()
