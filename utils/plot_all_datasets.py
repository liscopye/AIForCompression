"""
Plot PSNR vs BPP and throughput bar charts for all test datasets.
Modes: image_video | caesar | all
Usage: python utils/plot_all_datasets.py [--mode image_video|caesar|all]
"""
import json, os, sys
from collections import defaultdict
from pathlib import Path

import math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT = Path("/data/run01/scxj523/zsh/project/AIForCompression")

CAT_COLORS = {"image": "#2563eb", "video": "#dc2626", "caesar": "#16a34a"}
BAR_COLORS = {"encode": "#4C72B0", "decode": "#DD8452"}

MODEL_MARKERS = {
    "DCAE": "o", "LIC-HPCM-base": "s", "LIC-HPCM-large": "D",
    "DCMVC-Intra": "s", "DCMVC-Pframe": "v", "DCVC-RT-Intra": "o", "DCVC-RT-Pframe": "D",
    "CAESAR-V": "^", "CAESAR-D": "v",
}

KEEP_IMAGE_MODELS = {"DCAE", "LIC-HPCM-base", "LIC-HPCM-large"}
KEEP_VIDEO_MODELS = {"DCMVC-Intra", "DCMVC-Pframe", "DCVC-RT-Intra", "DCVC-RT-Pframe"}

# Fixed data range for consistent PSNR (max observed value per dataset)
DATASETS = {
    "Kodak": {
        "title": "Kodak",
        "sources": [
            ("unified_results/kodak_all_models/summary.json", "image"),
            ("unified_results/kodak_dcvc/summary.json", "video"),
            ("unified_results/kodak_dcvc_dcmvc/summary.json", "video"),
            ("unified_results/kodak_caesar/summary.json", "caesar"),
        ],
        "output": "unified_results/kodak_overview",
        "psnr_range": 1.0,
    },
    "UVG": {
        "title": "UVG (Twilight, 1080p)",
        "sources": [
            ("unified_results/uvg/summary.json", "image"),
            ("unified_results/uvg_video/summary.json", "video"),
            ("unified_results/uvg_dcvc_intra/summary.json", "video"),
            ("unified_results/uvg_dcvc_pframe/summary.json", "video"),
            ("unified_results/uvg_dcmvc_pframe/summary_dcmvc_pframe.json", "video"),
            ("unified_results/uvg_caesar/summary.json", "caesar"),
            
        ],
        "output": "unified_results/uvg_overview",
        "psnr_range": 1.0,  # uint8/255
    },
    "Hurricane": {
        "title": "Hurricane (P channel)",
        "sources": [
            ("unified_results/hurricane/summary.json", "image"),
            ("unified_results/hurricane_video/summary.json", "video"),
            ("unified_results/hurricane_dcvc/summary.json", "video"),
            ("unified_results/hurricane_caesar/summary.json", "caesar"),
            
        ],
        "output": "unified_results/hurricane_overview",
        "psnr_range": 1.0,
    },
    "S2C": {
        "title": "Sentinel-2 B02 (1024x1024 tiles)",
        "sources": [
            ("unified_results/s2c/summary.json", "image"),
            ("unified_results/s2c_dcvc/summary.json", "video"),
            ("unified_results/s2c_caesar/summary.json", "caesar"),
        ],
        "output": "unified_results/s2c_overview",
        "psnr_range": 1.0,
    },
    "NYX": {
        "title": "NYX (baryon density)",
        "sources": [
            ("unified_results/nyx_all/summary.json", "image"),
            ("unified_results/nyx_dcvc_intra/summary.json", "video"),
            ("unified_results/nyx_caesar/summary.json", "caesar"),
            
        ],
        "output": "unified_results/nyx_overview",
        "psnr_range": 1.0,
    },
    "isot1024": {
        "title": "Isotropic1024 (pressure)",
        "sources": [
            ("unified_results/isot1024/summary.json", "image"),
            ("unified_results/isot1024_dcvc/summary.json", "video"),
            ("unified_results/isot1024_caesar/summary.json", "caesar"),
            
        ],
        "output": "unified_results/isot1024_overview",
        "psnr_range": 1.0,
    },
    "Tomo": {
        "title": "Tomography (reconstructed, 3ch)",
        "sources": [
            ("unified_results/tomo_recon3ch/summary.json", "image"),
            ("unified_results/tomo_recon/summary.json", "video"),
            ("unified_results/tomo_recon_caesar/summary.json", "caesar"),
        ],
        "output": "unified_results/tomo_recon_overview",
        "psnr_range": 1.0,
    },
    "Lysozyme": {
        "title": "Lysozyme (CHESS)",
        "sources": [
            ("unified_results/lysozyme/summary.json", "image"),
            ("unified_results/lysozyme_dcvc/summary.json", "video"),
            ("unified_results/lysozyme_caesar/summary.json", "caesar"),
        ],
        "output": "unified_results/lysozyme_overview",
        "psnr_range": 1.0,
    },
    "ERA5": {
        "title": "ERA5 (268ch reanalysis)",
        "sources": [
            ("unified_results/era5_full/summary.json", "image"),
            ("unified_results/era5_full_dcvc/summary.json", "video"),
            ("unified_results/era5_caesar/summary.json", "caesar"),
        ],
        "output": "unified_results/era5_overview",
        "psnr_range": 1.0,
    },
}


def load_all(sources):
    """Load all rows, tagging with family. Filter anomalies."""
    all_rows = []
    for rel_path, family in sources:
        path = PROJECT / rel_path
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        for r in data:
            if "error" in r:
                continue
            mid = r.get("model_id", "")
            bpp = r.get("bpp", 0) or 0
            r["_source_file"] = rel_path
            # Skip old DCVC-RT Intra from bad source files (pre-YCbCr fix or broken test)
            bad_sources = ["uvg_video/summary", "kodak_dcvc_dcmvc/summary",
                          "nyx_video/summary", "s2c_video/summary", "hurricane_video/summary"]
            if "DCVC_RT_Intra" in mid and any(b in rel_path for b in bad_sources):
                continue
            # Skip DCVC-RT Intra garbage (bpp > 3, but NOT for multi-channel ERA5)
            if "DCVC_RT_Intra" in mid and bpp > 3 and "era5_full" not in rel_path:
                continue
            # Auto-detect video family from model name for combined files
            if family == "image" and r.get("model_name", "") in ("DCMVC", "DCVC-RT"):
                r["_family"] = "video"
            else:
                r["_family"] = family
            all_rows.append(r)
    return all_rows


def model_label(r):
    name = r.get("model_name", "") or r.get("model", "")
    mid = r.get("model_id", "") or r.get("model", "")
    if name == "LIC-HPCM":
        return "LIC-HPCM-large" if "large" in mid else "LIC-HPCM-base"
    if name == "DCMVC":
        if "Pframe" in mid:
            return "DCMVC-Pframe"
        return "DCMVC-Intra"
    if name == "DCVC-RT":
        if "Pframe" in mid:
            return "DCVC-RT-Pframe"
        return "DCVC-RT-Intra"
    if name == "CAESAR":
        return "CAESAR-V" if "caesar_v" in r.get("model_id", "") else "CAESAR-D"
    return name


def group_into_curves(rows, psnr_range=1.0):
    """Group rows into PSNR/BPP curves, filtering outlier tiles."""
    raw = defaultdict(lambda: defaultdict(list))
    for r in rows:
        label = model_label(r)
        if label is None:
            continue
        # Filter: only keep DCAE + HPCM for image, all video models
        family = r.get("_family", "?")
        if family == "image" and label not in KEEP_IMAGE_MODELS:
            continue
        if family == "video" and label not in KEEP_VIDEO_MODELS:
            continue
        # CAESAR uses eb as curve key, others use model_id
        key = r.get("eb") if "CAESAR" in label else r.get("model_id", "?")
        bpp = r.get("bpp")
        mse = r.get("mse")
        psnr = r.get("psnr")

        # Skip only truly perfect reconstruction (edge/no-data tiles)
        if mse is not None and mse < 1e-30:
            continue

        if psnr is None or bpp is None or not np.isfinite(float(psnr)):
            continue

        enc = r.get("encode_throughput_MBps") or 0
        dec = r.get("decode_throughput_MBps") or 0
        if enc == 0:
            orig_bytes = r.get("original_bytes", 0)
            enc_time = r.get("encode_time_avg", 0)
            enc = (orig_bytes / enc_time / 1e6) if enc_time > 0 else 0
        if dec == 0:
            orig_bytes = r.get("original_bytes", 0)
            dec_time = r.get("decode_time_avg", 0)
            dec = (orig_bytes / dec_time / 1e6) if dec_time > 0 else 0

        family = r.get("_family", "?")
        raw[(label, family)][key].append({
            "psnr": float(psnr), "bpp": float(bpp),
            "cr": float(r.get("compression_ratio", 0) or 1),
            "enc": float(enc), "dec": float(dec),
        })

    result = {}
    for (label, family), key_map in raw.items():
        pts = []
        for key, items in key_map.items():
            pts.append({
                "psnr": np.mean([x["psnr"] for x in items]),
                "bpp":  np.mean([x["bpp"] for x in items]),
                "enc":  np.mean([x["enc"] for x in items if x["enc"] > 0]) or 0,
                "dec":  np.mean([x["dec"] for x in items if x["dec"] > 0]) or 0,
            })
        pts.sort(key=lambda p: p["bpp"])
        if pts:
            result[(label, family)] = pts
    return result


def plot_psnr_vs_bpp(groups, output_dir, title):
    fig, ax = plt.subplots(figsize=(14, 8))
    psnr_min, psnr_max = float("inf"), float("-inf")

    for (label, family) in sorted(groups.keys()):
        pts = groups[(label, family)]
        color = CAT_COLORS.get(family, "#999")
        marker = MODEL_MARKERS.get(label, "o")
        x, y = [p["bpp"] for p in pts], [p["psnr"] for p in pts]
        psnr_min = min(psnr_min, *y)
        psnr_max = max(psnr_max, *y)
        ax.plot(x, y, color=color, marker=marker, linestyle="-",
                linewidth=2.0, markersize=8, label=label)

    ax.set_xlabel("BPP (bits per pixel)", fontsize=14)
    ax.set_ylabel("PSNR (dB)", fontsize=14)
    ax.set_title(f"{title}  —  PSNR vs BPP", fontsize=17, fontweight="bold")

    # Auto-set reasonable y-limits
    margin = max(1, (psnr_max - psnr_min) * 0.1)
    ax.set_ylim(max(0, psnr_min - margin), psnr_max + margin)

    ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.06, linewidth=0.3)

    # Category legend
    cat_handles = [
        Line2D([0],[0], marker='s', color='w', markerfacecolor=CAT_COLORS["image"], markersize=10, label='Image Models'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor=CAT_COLORS["video"], markersize=10, label='Video Models'),
    ]
    leg1 = ax.legend(handles=cat_handles, loc='upper left', fontsize=9, title='Category', title_fontsize=10, framealpha=0.9)
    ax.add_artist(leg1)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles=handles, labels=labels, loc='lower right', fontsize=8, framealpha=0.85)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = "psnr_vs_bpp.png"
    fig.savefig(output_dir / fname, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output_dir / fname}")


def plot_throughput_bar(groups, output_dir, title):
    combined = {}
    for (label, family), pts in groups.items():
        enc = [p["enc"] for p in pts if p["enc"] > 0]
        dec = [p["dec"] for p in pts if p["dec"] > 0]
        if not enc or not dec:
            continue
        combined[label] = {"encode": np.mean(enc), "decode": np.mean(dec), "family": family}

    if not combined:
        print(f"  [skip] no throughput data")
        return

    names = sorted(combined.keys(), key=lambda n: combined[n]["encode"] + combined[n]["decode"])
    enc_vals = [combined[n]["encode"] for n in names]
    dec_vals = [combined[n]["decode"] for n in names]

    x = np.arange(len(names))
    width = 0.32

    fig, ax = plt.subplots(figsize=(max(9, len(names)*1.1), 7))
    tick_colors = [CAT_COLORS.get(combined[n]["family"], "#666") for n in names]

    b1 = ax.bar(x - width/2, enc_vals, width, label="Encode", color=BAR_COLORS["encode"], edgecolor="white")
    b2 = ax.bar(x + width/2, dec_vals, width, label="Decode", color=BAR_COLORS["decode"], edgecolor="white")

    for bar, val in zip(b1, enc_vals):
        fmt = f"{val:.1f}" if val < 100 else f"{val:.0f}"
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), fmt, ha="center", va="bottom", fontsize=7, fontweight="bold")
    for bar, val in zip(b2, dec_vals):
        fmt = f"{val:.1f}" if val < 100 else f"{val:.0f}"
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), fmt, ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_ylabel("Throughput (MB/s)", fontsize=14)
    ax.set_title(f"{title}  —  Encode / Decode Throughput", fontsize=17, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11, rotation=20, ha="right")
    for tick, color in zip(ax.get_xticklabels(), tick_colors):
        tick.set_color(color)
    ax.legend(fontsize=12, framealpha=0.85)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "encode_decode_throughput.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output_dir / 'encode_decode_throughput.png'}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["image_video", "caesar", "all"], default="all")
    args = p.parse_args()

    for ds_name, cfg in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} ({cfg['title']})")

        # Select sources based on mode
        sources = []
        for rel_path, family in cfg["sources"]:
            if args.mode == "image_video" and family == "caesar":
                continue
            if args.mode == "caesar" and family != "caesar":
                continue
            sources.append((rel_path, family))

        if not sources:
            print("  SKIP (no sources for this mode)")
            continue

        rows = load_all(sources)
        print(f"  Loaded {len(rows)} rows")
        if not rows:
            print("  SKIP (no data)")
            continue

        groups = group_into_curves(rows, cfg.get("psnr_range", 1.0))
        for (label, family), pts in sorted(groups.items()):
            print(f"  [{family}] {label}: {len(pts)}pts, BPP={pts[0]['bpp']:.4f}~{pts[-1]['bpp']:.4f}, PSNR={pts[0]['psnr']:.1f}~{pts[-1]['psnr']:.1f}dB")

        if args.mode == "caesar":
            out = PROJECT / (cfg["output"] + "_caesar")
        elif args.mode == "all":
            out = PROJECT / (cfg["output"] + "_all")
        else:
            out = PROJECT / cfg["output"]

        plot_psnr_vs_bpp(groups, out, cfg["title"] + (" [CAESAR]" if args.mode=="caesar" else ""))
        plot_throughput_bar(groups, out, cfg["title"] + (" [CAESAR]" if args.mode=="caesar" else ""))

    print("\nDone all datasets.")


if __name__ == "__main__":
    main()
