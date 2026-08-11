#!/usr/bin/env python3
"""Plot the pre-tune to early fine-tune transition for ERA5 CAESAR models."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def display(label: str) -> str:
    return (label.replace("original", "pre-tune")
            .replace("update", "V ")
            .replace("s1_vae_", "S1 VAE ")
            .replace("s2_diff_", "S2 Diff "))


def plot_model(model_type: str, base: Path) -> None:
    data = json.loads((base / model_type / "sweep_results.json").read_text())
    groups = {}
    for item in data:
        groups.setdefault(item["checkpoint_label"], []).append(item)
    labels = list(groups)
    colors = ["#555555"] + list(plt.cm.Oranges(np.linspace(0.3, 0.95, len(labels) - 1))) \
        if model_type == "D" else ["#555555", "#64B5F6", "#0D47A1"]

    fig, axes = plt.subplots(1, 3, figsize=(21, 5.8))
    for color, label in zip(colors, labels):
        points = sorted(groups[label], key=lambda x: x["eb"])
        style = "-" if label in ("original", "update20k", "s1_vae_best", "s2_diff_40k") else "--"
        axes[0].plot([p["bpp"] for p in points], [p["psnr"] for p in points],
                     marker="o", linewidth=2 if style == "-" else 1.3,
                     linestyle=style, color=color, label=display(label))
        axes[1].plot([p["bpp"] for p in points], [p["per_sample_mean_psnr"] for p in points],
                     marker="o", linewidth=2 if style == "-" else 1.3,
                     linestyle=style, color=color, label=display(label))
    name = f"CAESAR-{model_type}"
    axes[0].set_title(f"{name} ERA5: Global RD", fontweight="bold")
    axes[1].set_title(f"{name} ERA5: Mean Block RD", fontweight="bold")
    axes[0].set_ylabel("PSNR (dB)")
    axes[1].set_ylabel("Mean block PSNR (dB)")
    for ax in axes[:2]:
        ax.set_xlabel("BPP (bits per element)")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(fontsize=7.5, ncol=2 if model_type == "D" else 1, framealpha=0.92)

    fixed = [next(p for p in groups[label] if p["eb"] == 1e-3) for label in labels]
    original = fixed[0]
    x = np.arange(len(labels))
    psnr_delta = [p["psnr"] - original["psnr"] for p in fixed]
    bpp_delta = [p["bpp"] - original["bpp"] for p in fixed]
    axes[2].plot(x, psnr_delta, marker="o", color="#BF360C" if model_type == "D" else "#0D47A1",
                 linewidth=2.2, label="PSNR change")
    rate = axes[2].twinx()
    rate.plot(x, bpp_delta, marker="s", linestyle="--",
              color="#FFB74D" if model_type == "D" else "#64B5F6",
              linewidth=2, label="BPP change")
    axes[2].axhline(0, color="#888888", linewidth=0.8)
    axes[2].set_xticks(x, [display(label).replace(" ", "\n") for label in labels],
                       fontsize=8)
    axes[2].set_title(f"{name}: Transition at EB=1e-3", fontweight="bold")
    axes[2].set_ylabel("PSNR change vs pre-tune (dB)")
    rate.set_ylabel("BPP change vs pre-tune")
    axes[2].grid(True, axis="y", alpha=0.25, linestyle="--")
    h1, l1 = axes[2].get_legend_handles_labels()
    h2, l2 = rate.get_legend_handles_labels()
    axes[2].legend(h1 + h2, l1 + l2, fontsize=9, loc="best")
    fig.tight_layout()
    out = base / model_type / f"caesar_{model_type.lower()}_era5_early_transition.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path,
                        default=Path("results/caesar_era5_early_transition"))
    args = parser.parse_args()
    for model_type in "VD":
        plot_model(model_type, args.input_dir)


if __name__ == "__main__":
    main()
