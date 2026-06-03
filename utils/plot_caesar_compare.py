#!/usr/bin/env python3
"""Compare two CAESAR summary.json files (e.g. original vs finetuned).

Usage:
  python utils/plot_caesar_compare.py \
    --orig unified_results/era5_caesar/summary.json \
    --tuned unified_results/era5_caesar_tuned/summary.json \
    --output unified_results/era5_caesar_tuned/comparison.png \
    --title "ERA5 24ch: CAESAR Tuned vs Original"
"""

import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── colour palette ──────────────────────────────────────────────
C = {
    "v_orig":   "#2196F3",   # blue  — CAESAR-V original (solid)
    "v_tuned":  "#0D47A1",   # dark blue — CAESAR-V tuned (dashed, unfilled)
    "d_orig":   "#FF9800",   # orange — CAESAR-D original (solid)
    "d_tuned":  "#BF360C",   # dark orange — CAESAR-D tuned (dashed, unfilled)
}

MARKERS  = {"caesar_v": "o", "caesar_d": "s"}
LABELS   = {"caesar_v": "CAESAR-V", "caesar_d": "CAESAR-D"}


def load(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def sort_by_eb(data: list[dict], model: str) -> list[dict]:
    return sorted(
        [r for r in data if r.get("model_id") == model and "error" not in r],
        key=lambda x: x["eb"],
    )


def plot_one_pair(ax, items_orig, items_tuned, model):
    label = LABELS[model]
    mk = MARKERS[model]
    c_orig = C["v_orig"] if model == "caesar_v" else C["d_orig"]
    c_tuned = C["v_tuned"] if model == "caesar_v" else C["d_tuned"]

    # original (solid lines, filled markers)
    bpp_o = [r["bpp"] for r in items_orig]
    psnr_o = [r["psnr"] for r in items_orig]
    ebs_o = [r["eb"] for r in items_orig]
    ax.plot(bpp_o, psnr_o, marker=mk, color=c_orig, ls="-",
            label=f"{label} original", markersize=9, linewidth=1.8)
    for x, y, e in zip(bpp_o, psnr_o, ebs_o):
        ax.annotate(f"{e:.0e}", (x, y), textcoords="offset points",
                     xytext=(9, 2), fontsize=6.5, color=c_orig, alpha=0.85)

    # tuned (dashed lines, unfilled markers)
    bpp_t = [r["bpp"] for r in items_tuned]
    psnr_t = [r["psnr"] for r in items_tuned]
    ebs_t = [r["eb"] for r in items_tuned]
    ax.plot(bpp_t, psnr_t, marker=mk, color=c_tuned, ls="--",
            label=f"{label} tuned", markersize=9, linewidth=1.8,
            markerfacecolor="none")
    for x, y, e in zip(bpp_t, psnr_t, ebs_t):
        ax.annotate(f"{e:.0e}", (x, y), textcoords="offset points",
                     xytext=(9, -14), fontsize=6.5, color=c_tuned, alpha=0.85)


def plot_throughput(ax, items_orig, items_tuned, model):
    label = LABELS[model]
    mk = MARKERS[model]
    c_orig = C["v_orig"] if model == "caesar_v" else C["d_orig"]
    c_tuned = C["v_tuned"] if model == "caesar_v" else C["d_tuned"]

    bpp_o = [r["bpp"] for r in items_orig]
    bpp_t = [r["bpp"] for r in items_tuned]

    def _plot(x, y, ls, c, lbl, mk, mfc):
        ax.plot(x, y, marker=mk, color=c, ls=ls, label=lbl, markersize=8, linewidth=1.5, markerfacecolor=mfc)

    _plot(bpp_o, [r["encode_throughput"] / 1e6 for r in items_orig], "-",  c_orig, f"{label} enc orig", mk, c_orig)
    _plot(bpp_o, [r["decode_throughput"] / 1e6 for r in items_orig], "-.", c_orig, f"{label} dec orig", mk, c_orig)
    _plot(bpp_t, [r["encode_throughput"] / 1e6 for r in items_tuned], "--", c_tuned, f"{label} enc tuned", mk, "none")
    _plot(bpp_t, [r["decode_throughput"] / 1e6 for r in items_tuned], ":",  c_tuned, f"{label} dec tuned", mk, "none")

    # Annotate eb on tuned encode points
    for x, y, e in zip(bpp_t, [r["encode_throughput"] / 1e6 for r in items_tuned], [r["eb"] for r in items_tuned]):
        ax.annotate(f"{e:.0e}", (x, y), textcoords="offset points",
                     xytext=(5, -12), fontsize=6, color=c_tuned, alpha=0.8)


def main():
    parser = argparse.ArgumentParser(description="Compare original vs tuned CAESAR results")
    parser.add_argument("--orig", required=True, help="Original summary.json")
    parser.add_argument("--tuned", required=True, help="Tuned summary.json")
    parser.add_argument("--output", default="caesar_comparison.png", help="Output png path")
    parser.add_argument("--title", default="CAESAR: Tuned vs Original", help="Plot title")
    args = parser.parse_args()

    original = load(args.orig)
    tuned = load(args.tuned)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # ── PSNR vs BPP ──
    ax = axes[0]
    for model in ["caesar_v", "caesar_d"]:
        plot_one_pair(ax, sort_by_eb(original, model), sort_by_eb(tuned, model), model)
    ax.set_xlabel("BPP (per element)", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title(args.title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle="--")

    # ── Throughput vs BPP ──
    ax = axes[1]
    for model in ["caesar_v", "caesar_d"]:
        plot_throughput(ax, sort_by_eb(original, model), sort_by_eb(tuned, model), model)
    ax.set_xlabel("BPP (per element)", fontsize=12)
    ax.set_ylabel("Throughput (MB/s)", fontsize=12)
    ax.set_title("Encode / Decode Throughput", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle="--")

    plt.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
