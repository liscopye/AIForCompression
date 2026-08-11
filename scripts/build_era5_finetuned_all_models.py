#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_objective_benchmark import model_family


EXPECTED_EBS = [0.1, 0.01, 0.003, 0.001, 0.0001, 3e-6, 1e-9]

STYLES = {
    "DCAE": ("#168AAD", "o", "-"),
    "LIC-HPCM-base": ("#52B788", "D", "-"),
    "LIC-HPCM-large": ("#087F5B", "D", "--"),
    "CAESAR-D original": ("#E9A23B", "s", "--"),
    "CAESAR-V original": ("#B86B17", "o", "--"),
    "CAESAR-V fine-tuned 100k": ("#C83E4D", "s", "-"),
    "cuSZ-Hi": ("#161616", "X", "-"),
    "nvJPEG2000": ("#7656A3", "P", "-"),
    "DCMVC-I": ("#2978A0", "v", "-"),
    "DCVC-RT-I": ("#D1495B", "^", "-"),
}

ORDER = {name: index for index, name in enumerate(STYLES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}")
    return payload


def curve_name(row: dict[str, Any]) -> str:
    family = model_family(row)
    model_id = str(row.get("model_id", "")).lower()
    if family == "LIC-HPCM":
        return "LIC-HPCM-large" if "large" in model_id else "LIC-HPCM-base"
    if family == "CAESAR-D":
        return "CAESAR-D original"
    if family == "CAESAR-V":
        if str(row.get("checkpoint_variant", "")).lower() == "finetuned_100k":
            return "CAESAR-V fine-tuned 100k"
        return "CAESAR-V original"
    return family


def finite_point(row: dict[str, Any]) -> bool:
    return (
        "error" not in row
        and isinstance(row.get("scientific_bpp_with_side_info"), (int, float))
        and isinstance(row.get("normalized_psnr"), (int, float))
        and float(row["scientific_bpp_with_side_info"]) > 0
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def plot(rows: list[dict[str, Any]], output: Path) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if finite_point(row):
            groups[curve_name(row)].append(row)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.9,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 5.4),
        gridspec_kw={"width_ratios": [1.05, 0.95]},
    )
    for ax in axes:
        for name in sorted(groups, key=lambda value: (ORDER.get(value, 999), value)):
            points = sorted(
                groups[name], key=lambda row: float(row["scientific_bpp_with_side_info"])
            )
            color, marker, linestyle = STYLES.get(name, ("#6B7280", "o", "-"))
            emphasis = name == "CAESAR-V fine-tuned 100k"
            ax.plot(
                [row["scientific_bpp_with_side_info"] for row in points],
                [row["normalized_psnr"] for row in points],
                label=name,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=2.5 if emphasis else 1.7,
                markersize=6.2 if emphasis else 4.8,
                markeredgecolor="white",
                markeredgewidth=0.6,
                zorder=6 if emphasis else 3,
            )
        ax.set_xlabel("BPP including required side information")
        ax.grid(True, which="major", color="#D1D5DB", linewidth=0.7, alpha=0.75)
        ax.grid(True, which="minor", color="#E5E7EB", linewidth=0.5, alpha=0.55)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_xscale("log")
    axes[0].set_ylabel("Dataset-normalized PSNR (dB)")
    axes[0].set_title("Full rate-distortion range")

    axes[1].set_xscale("log")
    axes[1].set_xlim(0.004, 2.5)
    axes[1].set_ylim(20, 72)
    axes[1].set_title("Low-BPP detail")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=5,
        frameon=False,
        columnspacing=1.0,
        handlelength=2.5,
    )

    fig.suptitle("ERA5 Objective-v1: CAESAR Fine-tuning and All Model Families", fontsize=14)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.25, wspace=0.18)
    save_figure(fig, output / "era5_objective_all_models_with_caesar_v_finetuned_100k.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    baseline = load_rows(args.baseline)
    fine_tuned = []
    for path in sorted(args.shard_root.glob("gpu*/era5_npy/summary.json")):
        fine_tuned.extend(
            row
            for row in load_rows(path)
            if str(row.get("checkpoint_variant", "")).lower() == "finetuned_100k"
        )
    fine_tuned = [row for row in fine_tuned if finite_point(row)]
    found_ebs = sorted(float(row["eb"]) for row in fine_tuned)
    if len(fine_tuned) != len(EXPECTED_EBS) or any(
        abs(found - expected) > max(1e-12, abs(expected) * 1e-8)
        for found, expected in zip(found_ebs, sorted(EXPECTED_EBS))
    ):
        raise ValueError(f"Expected fine-tuned EBs {EXPECTED_EBS}, found {found_ebs}")

    args.output.mkdir(parents=True, exist_ok=True)
    combined = baseline + fine_tuned
    (args.output / "summary.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )
    manifest = {
        "protocol_id": "aifc-objective-v1",
        "dataset": "era5_npy",
        "baseline_summary": str(args.baseline.resolve()),
        "fine_tuned_checkpoint_variant": "finetuned_100k",
        "fine_tuned_eb_values": EXPECTED_EBS,
        "fine_tuned_timing_status": "RD-only: 0 warmups and 1 measured repetition; do not use for throughput ranking",
        "comparison_scope": (
            "The fine-tuned curve supports comparison against CAESAR-V original only. "
            "It does not establish superiority over DCAE or LIC-HPCM because their "
            "measured BPP ranges do not overlap the fine-tuned CAESAR-V range."
        ),
        "baseline_rows": len(baseline),
        "fine_tuned_rows": len(fine_tuned),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    plot(combined, args.output)


if __name__ == "__main__":
    main()
