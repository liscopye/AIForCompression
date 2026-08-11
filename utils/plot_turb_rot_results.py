#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


CAESAR_LABELS = {
    "caesar_v": "CAESAR-V",
    "caesar_d": "CAESAR-D",
}


def load_labeled_records(path: str | Path, source: str) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    records = json.loads(path.read_text(encoding="utf-8"))
    labeled = []
    for record in records:
        if "error" in record:
            continue
        item = dict(record)
        item["source"] = source
        labeled.append(item)
    return labeled


def plot_label(record: dict) -> str:
    model_id = str(record.get("model_id", ""))
    source = record.get("source", "")
    if record.get("model_name") == "CAESAR" or model_id in CAESAR_LABELS:
        suffix = ""
        if source == "caesar_original":
            suffix = " original"
        elif source == "caesar_tuned":
            suffix = " tuned"
        return f"{CAESAR_LABELS.get(model_id, model_id)}{suffix}"
    if record.get("model_name") == "DCAE" and model_id.startswith("DCAE_"):
        return model_id.removeprefix("DCAE_").removesuffix(".pth")
    if model_id.startswith("LIC-HPCM-"):
        return model_id.replace("LIC-HPCM-", "HPCM-").removesuffix(".pth")
    if model_id.startswith("DCMVC_Intra_"):
        return model_id.replace("DCMVC_Intra_", "DCMVC-")
    if model_id.startswith("DCVC_RT_Intra_"):
        return model_id.replace("DCVC_RT_Intra_", "DCVC-RT-")
    return str(record.get("model_name") or model_id)


def curve_label(row: dict) -> str:
    model_name = row.get("model_name")
    model_id = str(row.get("model_id", ""))
    if model_name == "DCAE":
        return "DCAE"
    if model_name == "LIC-HPCM":
        if "LIC-HPCM-base" in model_id:
            return "HPCM-base"
        if "LIC-HPCM-large" in model_id:
            return "HPCM-large"
        return "HPCM"
    if model_name == "DCMVC":
        return "DCMVC"
    if model_name == "DCVC-RT":
        return "DCVC-RT"
    return str(model_name or row.get("label") or model_id)


def model_family(row: dict) -> str:
    model_name = row.get("model_name")
    model_id = str(row.get("model_id", ""))
    if model_name == "CAESAR" or model_id in CAESAR_LABELS:
        return "CAESAR"
    if model_name in {"DCMVC", "DCVC-RT"}:
        return "Video models"
    return "Image models"


def family_color(row: dict) -> str:
    family = model_family(row)
    model_name = row.get("model_name")
    model_id = str(row.get("model_id", ""))
    if family == "CAESAR":
        return "#f97316" if model_id == "caesar_d" else "#fb923c"
    if family == "Video models":
        return "#7c3aed" if model_name == "DCMVC" else "#2563eb"
    if model_name == "LIC-HPCM":
        return "#0891b2" if "base" in model_id else "#0e7490"
    if model_name == "DCAE":
        return "#16a34a"
    return "#22c55e"


def family_sort_key(row: dict) -> tuple[int, str, str]:
    order = {"CAESAR": 0, "Image models": 1, "Video models": 2}
    return (order.get(model_family(row), 99), curve_label(row), str(row.get("label", "")))


def aggregate_records(records: Iterable[dict], sequence_range: float | None = None) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        key = (
            str(record.get("source", "")),
            str(record.get("model_name", "")),
            str(record.get("model_id", "")),
        )
        groups[key].append(record)

    rows = []
    for (_source, _model_name, _model_id), items in groups.items():
        first = items[0]
        row = {
            "source": first.get("source", ""),
            "model_name": first.get("model_name", ""),
            "model_id": first.get("model_id", ""),
            "label": plot_label(first),
            "sample_count": len(items),
        }
        for field in (
            "mse", "psnr", "average_variable_psnr", "average_frame_psnr", "lpips", "compression_ratio",
            "bpp", "image_bpp", "scientific_bpp", "scientific_bpp_with_side_info",
            "memory_usage_MB", "memory_reserved_MB",
            "encode_throughput_MBps", "decode_throughput_MBps",
            "encode_throughput", "decode_throughput",
        ):
            values = [float(item[field]) for item in items if item.get(field) is not None]
            row[field] = mean(values) if values else None
        if sequence_range is not None and row["source"] == "image_models" and row.get("mse") is not None:
            row["sequence_psnr"] = psnr_from_mse(row["mse"], sequence_range)
            row["psnr"] = row["sequence_psnr"]
            row["average_variable_psnr"] = row["sequence_psnr"]
        row["average_lpips"] = row["lpips"]
        row["average_memory_usage_MB"] = row["memory_usage_MB"]
        row["average_memory_reserved_MB"] = row["memory_reserved_MB"]
        row["average_encode_throughput_MBps"] = row["encode_throughput_MBps"]
        row["average_decode_throughput_MBps"] = row["decode_throughput_MBps"]
        if row["average_encode_throughput_MBps"] is None and row.get("encode_throughput") is not None:
            row["average_encode_throughput_MBps"] = row["encode_throughput"] / 1e6
        if row["average_decode_throughput_MBps"] is None and row.get("decode_throughput") is not None:
            row["average_decode_throughput_MBps"] = row["decode_throughput"] / 1e6
        if row.get("average_variable_psnr") is None:
            row["average_variable_psnr"] = row.get("psnr")
        if row.get("average_frame_psnr") is None:
            row["average_frame_psnr"] = row.get("psnr")
        rows.append(row)
    return sorted(rows, key=lambda r: (r["source"], r["label"], str(r["model_id"])))


def psnr_from_mse(mse: float, data_range: float) -> float:
    if mse < 1e-30:
        return float("inf")
    if data_range < 1e-8:
        data_range = 1.0
    return float(10 * np.log10(data_range ** 2 / mse))


def load_npz_sequence_range(path: str | Path, section_index: int, max_samples: int) -> float:
    handle = np.load(path, allow_pickle=False)
    try:
        if "data" not in handle:
            raise KeyError(f"{path} does not contain a 'data' array")
        data = handle["data"]
        if data.ndim != 5:
            raise ValueError(f"Expected NPZ data in [V,S,T,H,W], got {data.shape}")
        if section_index < 0 or section_index >= data.shape[1]:
            raise ValueError(f"section_index {section_index} out of range for S={data.shape[1]}")
        sequence = data[:, section_index]
        if max_samples > 0:
            sequence = sequence[:,:max_samples]
        return float(np.max(sequence) - np.min(sequence))
    finally:
        handle.close()


def rd_bpp(record: dict) -> float | None:
    value = record.get("scientific_bpp_with_side_info")
    if value is None:
        value = record.get("scientific_bpp")
    if value is None:
        value = record.get("bpp")
    return value


def rd_psnr(record: dict) -> float | None:
    value = record.get("average_frame_psnr")
    if value is None:
        value = record.get("average_variable_psnr")
    if value is None:
        value = record.get("psnr")
    return value


def plot_caesar_rd(records: list[dict], output: Path, dataset_title: str) -> None:
    caesar = [r for r in records if r.get("model_name") == "CAESAR" or str(r.get("model_id", "")).startswith("caesar_")]
    if not caesar:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        ("caesar_v", "caesar_original"): ("#2563eb", "o", "-"),
        ("caesar_v", "caesar_tuned"): ("#1d4ed8", "o", "--"),
        ("caesar_d", "caesar_original"): ("#f97316", "s", "-"),
        ("caesar_d", "caesar_tuned"): ("#c2410c", "s", "--"),
    }
    for model_id in ("caesar_v", "caesar_d"):
        for source in ("caesar_original", "caesar_tuned"):
            items = sorted(
                [r for r in caesar if r.get("model_id") == model_id and r.get("source") == source],
                key=lambda r: float(r.get("eb", 0.0)),
            )
            if not items:
                continue
            color, marker, linestyle = styles[(model_id, source)]
            ax.plot(
                [rd_bpp(r) for r in items],
                [rd_psnr(r) for r in items],
                marker=marker,
                linestyle=linestyle,
                color=color,
                label=plot_label(items[0]),
            )
    ax.set_xlabel("Scientific BPP incl. side info")
    ax.set_ylabel("Average Frame PSNR (dB)")
    sources = {r.get("source") for r in caesar}
    if "caesar_tuned" in sources:
        ax.set_title(f"{dataset_title} CAESAR Original vs Tuned")
    else:
        ax.set_title(f"{dataset_title} CAESAR Original")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_all_rd(records: list[dict], rows: list[dict], output: Path, dataset_title: str) -> None:
    if not records and not rows:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    styles = {
        "CAESAR-V original": ("#2563eb", "o", "-"),
        "CAESAR-V tuned": ("#1d4ed8", "o", "--"),
        "CAESAR-D original": ("#f97316", "s", "-"),
        "CAESAR-D tuned": ("#c2410c", "s", "--"),
        "DCAE": ("#16a34a", "^", "-"),
        "HPCM-base": ("#0891b2", "D", "-"),
        "HPCM-large": ("#0e7490", "D", "--"),
        "DCMVC": ("#7c3aed", "v", "-"),
        "DCVC-RT": ("#dc2626", "P", "-"),
    }
    for model_id in ("caesar_v", "caesar_d"):
        for source in ("caesar_original", "caesar_tuned"):
            items = sorted(
                [
                    r for r in records
                    if r.get("model_id") == model_id
                    and r.get("source") == source
                    and rd_bpp(r) is not None
                    and rd_psnr(r) is not None
                ],
                key=lambda r: float(r.get("eb", 0.0)),
            )
            if not items:
                continue
            label = plot_label(items[0])
            color, marker, linestyle = styles[label]
            ax.plot([rd_bpp(r) for r in items], [rd_psnr(r) for r in items], marker=marker, linestyle=linestyle, color=color, label=label)

    image_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("source") != "image_models" or rd_bpp(row) is None or rd_psnr(row) is None:
            continue
        image_groups[curve_label(row)].append(row)
    for label, items in sorted(image_groups.items()):
        items = sorted(items, key=lambda r: rd_bpp(r))
        color, marker, linestyle = styles.get(label, ("#4b5563", "x", "-"))
        ax.plot([rd_bpp(r) for r in items], [rd_psnr(r) for r in items], marker=marker, linestyle=linestyle, color=color, label=label)

    if not ax.lines:
        plt.close(fig)
        return

    ax.set_xlabel("Scientific BPP incl. side info")
    ax.set_ylabel("Average Frame PSNR (dB)")
    ax.set_title(f"{dataset_title} Rate-Distortion Comparison")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_combined(rows: list[dict], output: Path) -> None:
    if not rows:
        return
    rows = sorted(rows, key=family_sort_key)
    labels = [r["label"] for r in rows]
    x_spacing = 1.55
    x = [i * x_spacing for i in range(len(rows))]
    colors = [family_color(r) for r in rows]
    fig_width = max(18, len(rows) * 0.58)
    fig, axes = plt.subplots(3, 2, figsize=(fig_width, 13))
    fields = [
        ("average_frame_psnr", "Average Frame PSNR (dB)", axes[0][0]),
        ("average_lpips", "Average LPIPS", axes[0][1]),
        ("compression_ratio", "Average Compression Ratio", axes[1][0]),
        ("average_memory_usage_MB", "Average Memory Usage (MB)", axes[1][1]),
        ("average_encode_throughput_MBps", "Average Encode Throughput (MB/s)", axes[2][0]),
        ("average_decode_throughput_MBps", "Average Decode Throughput (MB/s)", axes[2][1]),
    ]
    legend_handles = [
        Patch(facecolor="#f97316", label="CAESAR"),
        Patch(facecolor="#16a34a", label="Image models"),
        Patch(facecolor="#7c3aed", label="Video models"),
    ]
    for field, title, ax in fields:
        values = [r.get(field) for r in rows]
        numeric = [v if v is not None else 0.0 for v in values]
        bars = ax.bar(x, numeric, width=0.72, color=colors)
        for bar, value in zip(bars, values):
            text = "n/a" if value is None else f"{value:.3g}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), text, ha="center", va="bottom", fontsize=7)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.2)
        ax.legend(handles=legend_handles, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_csv(rows: list[dict], output: Path) -> None:
    fields = [
        "label", "source", "model_name", "model_id", "sample_count", "psnr",
        "sequence_psnr", "mse",
        "average_lpips", "compression_ratio", "bpp", "image_bpp",
        "scientific_bpp", "scientific_bpp_with_side_info", "average_variable_psnr", "average_frame_psnr",
        "average_memory_usage_MB", "average_memory_reserved_MB",
        "average_encode_throughput_MBps", "average_decode_throughput_MBps",
        "lpips", "memory_usage_MB", "memory_reserved_MB",
        "encode_throughput_MBps", "decode_throughput_MBps",
        "encode_throughput", "decode_throughput",
    ]
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join("" if row.get(field) is None else str(row.get(field)) for field in fields))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Turb_Rot benchmark summaries.")
    parser.add_argument("--image", required=True, help="Image models summary.json")
    parser.add_argument("--caesar_orig", required=True, help="Original CAESAR summary.json")
    parser.add_argument("--caesar_tuned", help="Tuned CAESAR summary.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_title", default="Turb_Rot")
    parser.add_argument("--output_prefix", default="turb_rot")
    parser.add_argument("--sequence_npz", help="Optional [V,S,T,H,W] NPZ used to recompute image/video sequence PSNR from mean MSE.")
    parser.add_argument("--sequence_section_index", type=int, default=0)
    parser.add_argument("--sequence_max_samples", type=int, default=16)
    args = parser.parse_args()

    records = []
    records.extend(load_labeled_records(args.image, "image_models"))
    records.extend(load_labeled_records(args.caesar_orig, "caesar_original"))
    if args.caesar_tuned:
        records.extend(load_labeled_records(args.caesar_tuned, "caesar_tuned"))
    sequence_range = None
    if args.sequence_npz:
        sequence_range = load_npz_sequence_range(args.sequence_npz, args.sequence_section_index, args.sequence_max_samples)
    rows = aggregate_records(records, sequence_range=sequence_range)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    caesar_plot_name = f"{prefix}_caesar_original_vs_tuned.png" if args.caesar_tuned else f"{prefix}_caesar_original.png"
    plot_caesar_rd(records, output_dir / caesar_plot_name, args.dataset_title)
    plot_all_rd(records, rows, output_dir / f"{prefix}_rd_all_models.png", args.dataset_title)
    plot_combined(rows, output_dir / f"{prefix}_all_models_metrics.png")
    write_csv(rows, output_dir / f"{prefix}_summary.csv")
    (output_dir / f"{prefix}_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved plots and CSV to {output_dir}")


if __name__ == "__main__":
    main()
