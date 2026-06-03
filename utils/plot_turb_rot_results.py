#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    return str(record.get("model_name") or model_id)


def aggregate_records(records: Iterable[dict]) -> list[dict]:
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
        for field in ("psnr", "lpips", "compression_ratio", "bpp", "memory_usage_MB", "memory_reserved_MB"):
            values = [float(item[field]) for item in items if item.get(field) is not None]
            row[field] = mean(values) if values else None
        rows.append(row)
    return sorted(rows, key=lambda r: (r["source"], r["label"], str(r["model_id"])))


def plot_caesar_rd(records: list[dict], output: Path) -> None:
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
                [r.get("bpp") for r in items],
                [r.get("psnr") for r in items],
                marker=marker,
                linestyle=linestyle,
                color=color,
                label=plot_label(items[0]),
            )
            for item in items:
                if item.get("eb") is not None:
                    ax.annotate(f"{float(item['eb']):.0e}", (item.get("bpp"), item.get("psnr")), fontsize=7)
    ax.set_xlabel("BPP")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Turb_Rot CAESAR Original vs Tuned")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_combined(rows: list[dict], output: Path) -> None:
    if not rows:
        return
    labels = [r["label"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fields = [
        ("psnr", "Average PSNR (dB)", axes[0][0]),
        ("lpips", "Average LPIPS", axes[0][1]),
        ("compression_ratio", "Average Compression Ratio", axes[1][0]),
        ("memory_usage_MB", "Average Memory Usage (MB)", axes[1][1]),
    ]
    for field, title, ax in fields:
        values = [r.get(field) for r in rows]
        numeric = [v if v is not None else 0.0 for v in values]
        bars = ax.bar(labels, numeric, color="#4b5563")
        for bar, value in zip(bars, values):
            text = "n/a" if value is None else f"{value:.3g}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), text, ha="center", va="bottom", fontsize=7)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_csv(rows: list[dict], output: Path) -> None:
    fields = ["label", "source", "model_name", "model_id", "sample_count", "psnr", "lpips", "compression_ratio", "bpp", "memory_usage_MB", "memory_reserved_MB"]
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join("" if row.get(field) is None else str(row.get(field)) for field in fields))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Turb_Rot benchmark summaries.")
    parser.add_argument("--image", required=True, help="Image models summary.json")
    parser.add_argument("--caesar_orig", required=True, help="Original CAESAR summary.json")
    parser.add_argument("--caesar_tuned", required=True, help="Tuned CAESAR summary.json")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    records = []
    records.extend(load_labeled_records(args.image, "image_models"))
    records.extend(load_labeled_records(args.caesar_orig, "caesar_original"))
    records.extend(load_labeled_records(args.caesar_tuned, "caesar_tuned"))
    rows = aggregate_records(records)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_caesar_rd(records, output_dir / "turb_rot_caesar_original_vs_tuned.png")
    plot_combined(rows, output_dir / "turb_rot_all_models_metrics.png")
    write_csv(rows, output_dir / "turb_rot_summary.csv")
    print(f"Saved plots and CSV to {output_dir}")


if __name__ == "__main__":
    main()
