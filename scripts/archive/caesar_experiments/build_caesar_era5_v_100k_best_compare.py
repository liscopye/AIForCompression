#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_EBS = [0.1, 0.01, 0.003, 0.001, 0.0001, 3e-6, 1e-9]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--new-variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a row list in {path}")
    return rows


def valid(row: dict[str, Any]) -> bool:
    bpp = row.get("scientific_bpp_with_side_info")
    psnr = row.get("normalized_psnr")
    return (
        "error" not in row
        and isinstance(bpp, (int, float))
        and isinstance(psnr, (int, float))
        and math.isfinite(float(bpp))
        and math.isfinite(float(psnr))
        and float(bpp) > 0
    )


def caesar_v_rows(rows: list[dict[str, Any]], variant: str | None) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        model_id = str(row.get("model_id", "")).lower()
        if not model_id.startswith("caesar_v"):
            continue
        actual_variant = str(row.get("checkpoint_variant", "original")).lower()
        if variant is None:
            if actual_variant not in {"", "original", "none"}:
                continue
        elif actual_variant != variant.lower():
            continue
        if valid(row):
            selected.append(dict(row))
    selected.sort(key=lambda row: float(row["scientific_bpp_with_side_info"]))
    found = sorted(float(row["eb"]) for row in selected)
    expected = sorted(EXPECTED_EBS)
    if len(found) != len(expected) or any(
        abs(a - b) > max(1e-12, abs(b) * 1e-8) for a, b in zip(found, expected)
    ):
        raise ValueError(f"Variant {variant or 'original'} has EB values {found}, expected {expected}")
    return selected


def save_plot(curves: dict[str, list[dict[str, Any]]], output: Path) -> None:
    styles = {
        "Original": {"color": "#555555", "marker": "o", "linestyle": "--"},
        "Previous FT 100k": {"color": "#D17A22", "marker": "s", "linestyle": "-"},
        "New low-rate FT 100k": {"color": "#007C91", "marker": "D", "linestyle": "-"},
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.linewidth": 0.9,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    for ax in axes:
        for label, rows in curves.items():
            style = styles[label]
            ax.plot(
                [row["scientific_bpp_with_side_info"] for row in rows],
                [row["normalized_psnr"] for row in rows],
                label=label,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.2,
                markersize=5.8,
                markeredgecolor="white",
                markeredgewidth=0.65,
            )
        ax.set_xscale("log")
        ax.set_xlabel("BPP including required side information")
        ax.grid(True, which="major", color="#D1D5DB", linewidth=0.7, alpha=0.8)
        ax.grid(True, which="minor", color="#E5E7EB", linewidth=0.45, alpha=0.65)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Dataset-normalized PSNR (dB)")
    axes[0].set_title("Full RD range")
    axes[1].set_xlim(0.07, 2.5)
    axes[1].set_ylim(30, 72)
    axes[1].set_title("Low-BPP detail")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        handlelength=2.8,
    )
    fig.suptitle("ERA5 Objective-v1: CAESAR-V Fine-tuning Comparison", fontsize=14)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.84, bottom=0.20, wspace=0.17)
    fig.savefig(output / "caesar_v_era5_original_previous_new_100k.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "caesar_v_era5_original_previous_new_100k.pdf", bbox_inches="tight")
    plt.close(fig)


def interpolated_bpp(rows: list[dict[str, Any]], psnr: float) -> float:
    points = sorted(
        (float(row["normalized_psnr"]), math.log(float(row["scientific_bpp_with_side_info"])))
        for row in rows
    )
    quality = np.asarray([point[0] for point in points])
    log_rate = np.asarray([point[1] for point in points])
    return math.exp(float(np.interp(psnr, quality, log_rate)))


def main() -> None:
    args = parse_args()
    curves = {
        "Original": caesar_v_rows(load_rows(args.baseline), None),
        "Previous FT 100k": caesar_v_rows(load_rows(args.previous), "finetuned_100k"),
        "New low-rate FT 100k": caesar_v_rows(load_rows(args.new), args.new_variant),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    combined = []
    for label, rows in curves.items():
        for row in rows:
            row["comparison_curve"] = label
            combined.append(row)
    (args.output / "summary.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )
    comparison_psnr = min(
        float(curves["Previous FT 100k"][0]["normalized_psnr"]),
        float(curves["New low-rate FT 100k"][-1]["normalized_psnr"]),
    )
    common_quality_rates = {
        label: interpolated_bpp(rows, comparison_psnr)
        for label, rows in curves.items()
    }
    manifest = {
        "protocol_id": "aifc-objective-v1",
        "dataset": "era5_npy",
        "curves": {
            "original": str(args.baseline.resolve()),
            "previous_finetuned_100k": str(args.previous.resolve()),
            "new_finetuned_100k": str(args.new.resolve()),
        },
        "new_checkpoint_variant": args.new_variant,
        "new_checkpoint": curves["New low-rate FT 100k"][0].get("checkpoint_root"),
        "previous_checkpoint": curves["Previous FT 100k"][0].get("checkpoint_root"),
        "selection_scope": (
            "The new lambda=1e-3 checkpoint is selected as the low-rate CAESAR-V result. "
            "It is not the best member of the new sweep at every PSNR; lambda=1e-4 is "
            "slightly better in parts of the high-quality range."
        ),
        "common_quality_comparison": {
            "normalized_psnr_db": comparison_psnr,
            "interpolated_bpp": common_quality_rates,
            "new_vs_previous_rate_change_pct": (
                common_quality_rates["New low-rate FT 100k"]
                / common_quality_rates["Previous FT 100k"]
                - 1.0
            )
            * 100.0,
            "new_vs_original_rate_change_pct": (
                common_quality_rates["New low-rate FT 100k"]
                / common_quality_rates["Original"]
                - 1.0
            )
            * 100.0,
        },
        "eb_values": EXPECTED_EBS,
        "timing_status": "RD-only: 0 warmups and 1 measured repetition; do not use for throughput ranking",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    save_plot(curves, args.output)


if __name__ == "__main__":
    main()
