#!/usr/bin/env python3
"""Plot CAESAR lysozyme EB sweep: PSNR vs BPP + Encode/Decode time vs EB.

Usage:
  python utils/plot_caesar_lysozyme_eb_sweep.py --model_type V
  python utils/plot_caesar_lysozyme_eb_sweep.py --model_type D
  python utils/plot_caesar_lysozyme_eb_sweep.py --model_type both
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STYLES = {
    "CAESAR-V_original":  {"color": "#3498db", "marker": "o", "label": "CAESAR-V (original)"},
    "CAESAR-V_finetuned": {"color": "#e74c3c", "marker": "s", "label": "CAESAR-V (finetuned)"},
    "CAESAR-D_original":  {"color": "#2ecc71", "marker": "o", "label": "CAESAR-D (original)"},
    "CAESAR-D_finetuned": {"color": "#e67e22", "marker": "s", "label": "CAESAR-D (finetuned)"},
}


def load_results(sweep_path: str) -> dict:
    with open(sweep_path) as f:
        data = json.load(f)
    groups = {}
    for r in data:
        name = r["variant"]
        groups.setdefault(name, []).append(r)
    for v in groups.values():
        v.sort(key=lambda x: x["eb"])
    return groups


def plot_psnr_vs_bpp(groups: dict, out_dir: Path, model_tag: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, items in sorted(groups.items()):
        if name not in STYLES:
            continue
        s = STYLES[name]
        x = [r["bpp"] for r in items]
        y = [r["psnr"] for r in items]
        for xi, yi, r in zip(x, y, items):
            ax.annotate(f"{r['eb']:.0e}", (xi, yi), textcoords="offset points",
                        xytext=(8, 4), fontsize=7, alpha=0.7)
        ax.plot(x, y, color=s["color"], marker=s["marker"],
                linewidth=2.2, markersize=9, label=s["label"])

    ax.set_xlabel("BPP (bits per pixel)", fontsize=14)
    ax.set_ylabel("PSNR (dB)", fontsize=14)
    ax.set_title(f"CAESAR-{model_tag} Lysozyme — PSNR vs BPP (EB sweep)", fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.legend(fontsize=12, framealpha=0.85)
    fig.tight_layout()
    out = out_dir / "psnr_vs_bpp.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_time_vs_eb(groups: dict, out_dir: Path, model_tag: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for name, items in sorted(groups.items()):
        if name not in STYLES:
            continue
        s = STYLES[name]
        eb = [r["eb"] for r in items]
        enc = [r["encode_time_total"] for r in items]
        dec = [r["decode_time_total"] for r in items]

        ax1.semilogx(eb, enc, color=s["color"], marker=s["marker"],
                     linewidth=2.2, markersize=9, label=s["label"])
        ax2.semilogx(eb, dec, color=s["color"], marker=s["marker"],
                     linewidth=2.2, markersize=9, label=s["label"])

    ax1.set_xlabel("Error Bound (EB)", fontsize=13)
    ax1.set_ylabel("Encode Time (s)", fontsize=13)
    ax1.set_title("Encode Time vs EB", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3, linewidth=0.6)
    ax1.legend(fontsize=11, framealpha=0.85)

    ax2.set_xlabel("Error Bound (EB)", fontsize=13)
    ax2.set_ylabel("Decode Time (s)", fontsize=13)
    ax2.set_title("Decode Time vs EB", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, linewidth=0.6)
    ax2.legend(fontsize=11, framealpha=0.85)

    fig.suptitle(f"CAESAR-{model_tag} Lysozyme — Encode / Decode Time vs EB",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    out = out_dir / "time_vs_eb.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def make_sweep_comparison_plot(groups_v: dict, groups_d: dict, out_base: Path):
    """Combined PSNR vs BPP plot with both V and D on same axes."""
    fig, ax = plt.subplots(figsize=(12, 7))
    all_groups = {}
    if groups_v:
        all_groups.update(groups_v)
    if groups_d:
        all_groups.update(groups_d)

    for name, items in sorted(all_groups.items()):
        if name not in STYLES:
            continue
        s = STYLES[name]
        x = [r["bpp"] for r in items]
        y = [r["psnr"] for r in items]
        for xi, yi, r in zip(x, y, items):
            ax.annotate(f"{r['eb']:.0e}", (xi, yi), textcoords="offset points",
                        xytext=(8, 4), fontsize=7, alpha=0.6)
        ax.plot(x, y, color=s["color"], marker=s["marker"],
                linewidth=2.2, markersize=9, label=s["label"])

    ax.set_xlabel("BPP (bits per pixel)", fontsize=14)
    ax.set_ylabel("PSNR (dB)", fontsize=14)
    ax.set_title("CAESAR Lysozyme — PSNR vs BPP (V & D)", fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.legend(fontsize=12, framealpha=0.85)
    fig.tight_layout()
    out = out_base / "psnr_vs_bpp_combined.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    p = argparse.ArgumentParser(description="Plot CAESAR lysozyme EB sweep results")
    p.add_argument("--model_type", default="D", choices=["V", "D", "both"])
    p.add_argument(
        "--sweep_dir",
        default="/workspace/AIForCompression/unified_results/lysozyme_caesar_tuned",
        help="Directory containing v/sweep_results.json and d/sweep_results.json.",
    )
    args = p.parse_args()

    base = Path(args.sweep_dir)
    groups_v, groups_d = {}, {}

    if args.model_type in ("V", "both"):
        sweep_path = base / "v" / "sweep_results.json"
        if sweep_path.exists():
            groups_v = load_results(str(sweep_path))
            print(f"Loaded V results: {sum(len(v) for v in groups_v.values())} entries")
            for name, items in sorted(groups_v.items()):
                for r in items:
                    print(f"  {name:30s} eb={r['eb']:.1e}  PSNR={r['psnr']:.2f}  BPP={r['bpp']:.5f}")

            out_dir = base / "v"
            plot_psnr_vs_bpp(groups_v, out_dir, "V")
            plot_time_vs_eb(groups_v, out_dir, "V")

    if args.model_type in ("D", "both"):
        sweep_path = base / "d" / "sweep_results.json"
        if sweep_path.exists():
            groups_d = load_results(str(sweep_path))
            print(f"Loaded D results: {sum(len(v) for v in groups_d.values())} entries")
            for name, items in sorted(groups_d.items()):
                for r in items:
                    print(f"  {name:30s} eb={r['eb']:.1e}  PSNR={r['psnr']:.2f}  BPP={r['bpp']:.5f}")

            out_dir = base / "d"
            plot_psnr_vs_bpp(groups_d, out_dir, "D")
            plot_time_vs_eb(groups_d, out_dir, "D")

    if args.model_type == "both" and groups_v and groups_d:
        make_sweep_comparison_plot(groups_v, groups_d, base)

    print("Done.")


if __name__ == "__main__":
    main()
