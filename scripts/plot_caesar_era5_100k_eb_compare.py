#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--original-no-pca", type=Path, required=True)
    parser.add_argument("--finetuned-no-pca", type=Path, required=True)
    return parser.parse_args()


def load_one(path: Path) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8"))
    valid = [row for row in rows if "error" not in row]
    if len(valid) != 1:
        raise ValueError(f"Expected one valid result in {path}, found {len(valid)}")
    return valid[0]


def main() -> None:
    args = parse_args()
    root = args.result_root
    raw_root = root / "raw"
    series = {}
    for variant, baseline_path in (
        ("Original", args.original_no_pca),
        ("Fine-tuned 100k", args.finetuned_no_pca),
    ):
        prefix = "original" if variant == "Original" else "finetuned"
        eb_results = [
            load_one(path)
            for path in sorted(
                raw_root.glob(f"{prefix}_eb*/summary.json"),
                key=lambda path: float(
                    path.parent.name.split("_eb", 1)[1].replace("p", ".")
                ),
                reverse=True,
            )
        ]
        if len(eb_results) != 7:
            raise ValueError(f"Expected 7 EB results for {variant}, found {len(eb_results)}")
        series[variant] = {
            "no_pca": load_one(baseline_path),
            "eb_results": eb_results,
        }

    comparison = {
        "protocol": {
            "dataset": "ERA5 2024-06-01 through 2024-06-16 at 00:00 UTC",
            "variables": 268,
            "resolution": [240, 240],
            "windows": 2,
            "frames_per_window": 8,
            "metric": "average-variable PSNR",
            "eb_values": [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001],
        },
        "series": series,
    }
    (root / "comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.linewidth": 0.9,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.4),
        gridspec_kw={"width_ratios": [1.08, 0.92]},
        constrained_layout=True,
    )
    styles = {
        "Original": {
            "color": "#4B5563",
            "marker": "o",
            "linestyle": "--",
        },
        "Fine-tuned 100k": {
            "color": "#C43D3D",
            "marker": "s",
            "linestyle": "-",
        },
    }

    for ax in axes:
        for label, values in series.items():
            points = sorted(values["eb_results"], key=lambda row: row["bpp"])
            style = styles[label]
            ax.plot(
                [row["bpp"] for row in points],
                [row["average_variable_psnr"] for row in points],
                label=f"{label} + PCA",
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                markersize=5.5,
                markeredgecolor="white",
                markeredgewidth=0.7,
                zorder=3,
            )
            baseline = values["no_pca"]
            ax.scatter(
                baseline["bpp"],
                baseline["average_variable_psnr"],
                label=f"{label} no PCA",
                color=style["color"],
                marker="*",
                s=115,
                edgecolors="white",
                linewidths=0.8,
                zorder=4,
            )
        ax.set_xlabel("Scientific BPP")
        ax.grid(True, which="major", color="#D1D5DB", linewidth=0.7, alpha=0.75)
        ax.grid(True, which="minor", color="#E5E7EB", linewidth=0.5, alpha=0.55)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_xscale("log")
    axes[0].xaxis.set_major_formatter(ScalarFormatter())
    axes[0].set_ylabel("Average-variable PSNR (dB)")
    axes[0].set_title("Full rate-distortion range")

    axes[1].set_xlim(0.22, 1.15)
    axes[1].set_ylim(43, 54)
    axes[1].set_title("Low-BPP detail")
    axes[1].legend(loc="lower right", frameon=True, framealpha=0.95)

    fig.suptitle("CAESAR-V on ERA5: Original vs Fine-tuned 100k", fontsize=14)
    fig.savefig(root / "caesar_era5_original_vs_finetuned_100k_eb.png")
    fig.savefig(root / "caesar_era5_original_vs_finetuned_100k_eb.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
