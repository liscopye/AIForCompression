#!/usr/bin/env python3
"""Plot ERA5 checkpoint sweeps separately for CAESAR-V and CAESAR-D."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "V": ["#555555", "#BBDEFB", "#90CAF9", "#64B5F6", "#42A5F5", "#2196F3", "#0D47A1"],
    "D": ["#555555", "#FFE0B2", "#FFCC80", "#FFB74D", "#FFA726", "#FF9800", "#BF360C"],
}


def plot_model(model_type: str, base: Path) -> Path:
    data = json.loads((base / model_type / "sweep_results.json").read_text())
    groups = {}
    for result in data:
        groups.setdefault(result["checkpoint_label"], []).append(result)

    fig, axes = plt.subplots(1, 3, figsize=(20.5, 5.7))
    for color, (label, items) in zip(COLORS[model_type], groups.items()):
        items.sort(key=lambda item: item["eb"])
        display = ("original (pre-tune)" if label == "original" else
                   label.replace("_resume", " (resume)").replace("update", "step "))
        highlighted = label in ("original", "best")
        style = "-" if highlighted else "--"
        width = 2.7 if highlighted else 1.8
        axes[0].plot([r["bpp"] for r in items], [r["psnr"] for r in items],
                     marker="o", markersize=6, color=color, linestyle=style,
                     linewidth=width, label=display)
        axes[1].plot([r["bpp"] for r in items], [r["per_sample_mean_psnr"] for r in items],
                     marker="o", markersize=6, color=color, linestyle=style,
                     linewidth=width, label=display)

    name = f"CAESAR-{model_type}"
    axes[0].set_title(f"{name} ERA5: Global RD Curve", fontweight="bold")
    axes[1].set_title(f"{name} ERA5: Mean Block RD Curve", fontweight="bold")
    axes[0].set_ylabel("PSNR (dB)")
    axes[1].set_ylabel("Mean block PSNR (dB)")
    for ax in axes[:2]:
        ax.set_xlabel("BPP (bits per element)")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(fontsize=9, framealpha=0.92)

    reference_eb = 1e-3
    fixed = [next(item for item in items if item["eb"] == reference_eb)
             for items in groups.values()]
    labels = [
        ("pre-tune" if item["checkpoint_label"] == "original" else
         item["checkpoint_label"].replace("_resume", "\nresume").replace("update", ""))
        for item in fixed
    ]
    psnr_base = fixed[0]["psnr"]
    bpp_base = fixed[0]["bpp"]
    psnr_delta = [item["psnr"] - psnr_base for item in fixed]
    bpp_delta = [item["bpp"] - bpp_base for item in fixed]
    x = list(range(len(fixed)))
    main_color = COLORS[model_type][-1]
    rate_color = COLORS[model_type][2]
    axes[2].plot(x, psnr_delta, color=main_color, marker="o", linewidth=2.3,
                 label="PSNR change")
    rate_axis = axes[2].twinx()
    rate_axis.plot(x, bpp_delta, color=rate_color, marker="s", linestyle="--",
                   linewidth=2.0, label="BPP change")
    axes[2].axhline(0, color="#888888", linewidth=0.8)
    axes[2].set_title(f"{name}: Trend at EB=1e-3", fontweight="bold")
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel("PSNR change from first checkpoint (dB)", color=main_color)
    rate_axis.set_ylabel("BPP change from first checkpoint", color=rate_color)
    axes[2].grid(True, axis="y", alpha=0.25, linestyle="--")
    handles, legend_labels = axes[2].get_legend_handles_labels()
    handles2, legend_labels2 = rate_axis.get_legend_handles_labels()
    axes[2].legend(handles + handles2, legend_labels + legend_labels2,
                   fontsize=9, framealpha=0.92, loc="best")
    fig.tight_layout()
    output = base / model_type / f"caesar_{model_type.lower()}_era5_checkpoint_comparison.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path,
                        default=Path("results/caesar_era5_checkpoint_sweep"))
    parser.add_argument("--model_type", choices=["V", "D", "both"], default="both")
    args = parser.parse_args()
    models = ["V", "D"] if args.model_type == "both" else [args.model_type]
    for model_type in models:
        plot_model(model_type, args.input_dir)


if __name__ == "__main__":
    main()
