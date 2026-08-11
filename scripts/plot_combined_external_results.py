#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STYLES = {
    "CAESAR-D original": ("#E69F00", "s", "-"),
    "CAESAR-V original": ("#F0B44C", "o", "-"),
    "CAESAR-D tuned": ("#D55E00", "s", "--"),
    "CAESAR-V tuned": ("#CC79A7", "o", "--"),
    "CAESAR-D no PCA": ("#8B4513", "s", ":"),
    "CAESAR-V no PCA": ("#B07AA1", "o", ":"),
    "DCAE": ("#009E73", "^", "-"),
    "DCAE+CAESAR-PCA": ("#009E73", "^", "--"),
    "HPCM-base": ("#56B4E9", "D", "-"),
    "HPCM-base+CAESAR-PCA": ("#56B4E9", "D", "--"),
    "HPCM-large": ("#0072B2", "D", "--"),
    "HPCM-large+CAESAR-PCA": ("#0072B2", "D", ":"),
    "DCMVC I-frame": ("#7A5195", "v", "-"),
    "DCMVC P-frame": ("#7A5195", "^", "--"),
    "DCVC-RT I-frame": ("#EF5675", "P", "-"),
    "DCVC-RT P-frame": ("#EF5675", "X", "--"),
    "cuSZ-Hi": ("#111111", "X", "-"),
    "cuSZ-Hi-3D": ("#555555", "X", "--"),
    "nvJPEG": ("#8A2BE2", "p", "-"),
    "nvJPEG2000": ("#00A6D6", "P", "-"),
    "GraphComp": ("#666666", "*", "-"),
    "visemz": ("#2F4B7C", "h", ""),
}

CURVE_ORDER = {
    "CAESAR-D no PCA": 0,
    "CAESAR-V no PCA": 1,
    "CAESAR-D original": 2,
    "CAESAR-V original": 3,
    "CAESAR-D tuned": 4,
    "CAESAR-V tuned": 5,
    "cuSZ-Hi": 6,
    "cuSZ-Hi-3D": 7,
    "nvJPEG": 8,
    "nvJPEG2000": 9,
    "DCAE": 10,
    "DCAE+CAESAR-PCA": 11,
    "HPCM-base": 12,
    "HPCM-base+CAESAR-PCA": 13,
    "HPCM-large": 14,
    "HPCM-large+CAESAR-PCA": 15,
    "DCMVC I-frame": 16,
    "DCMVC P-frame": 17,
    "DCVC-RT I-frame": 18,
    "DCVC-RT P-frame": 19,
    "visemz": 20,
    "GraphComp": 21,
}

FALLBACK_PARAMS = {
    "CAESAR-D original": 36_763_025,
    "CAESAR-V original": 1_501_356,
    "CAESAR-D tuned": 36_763_025,
    "CAESAR-V tuned": 1_501_356,
    "CAESAR-D no PCA": 36_763_025,
    "CAESAR-V no PCA": 1_501_356,
    "DCAE": 119_400_351,
    "DCAE+CAESAR-PCA": 119_400_351,
    "HPCM-base": 68_505_299,
    "HPCM-base+CAESAR-PCA": 68_505_299,
    "HPCM-large": 89_718_803,
    "HPCM-large+CAESAR-PCA": 89_718_803,
    "DCMVC I-frame": 30_998_187,
    "DCMVC P-frame": 30_998_187,
    "DCVC-RT I-frame": 45_651_314,
    "DCVC-RT P-frame": 45_651_314,
    "cuSZ-Hi": 0,
    "cuSZ-Hi-3D": 0,
    "nvJPEG": 0,
    "nvJPEG2000": 0,
    "GraphComp": 0,
    "visemz": 2_986_459,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--zoom_bpp", type=float, default=2.0)
    parser.add_argument("--single_rd", action="store_true", help="Only write the all-model RD plot.")
    parser.add_argument("--extra_metrics", action="store_true", help="Write LPIPS, throughput, and params-throughput plots.")
    parser.add_argument("--exclude_model", action="append", default=[], help="Model name to exclude from plots.")
    args = parser.parse_args()
    setup_plot_style()

    rows = load_summary(Path(args.summary))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded = set(args.exclude_model)
    plot_rows = [normalize_row(r) for r in rows if point_ok(r) and r.get("model_name") not in excluded]

    plot_rd(plot_rows, output_dir / f"{args.prefix}_rd_all_models.png", args.title, zoom_bpp=None)
    if args.extra_metrics:
        plot_lpips(plot_rows, output_dir / f"{args.prefix}_lpips_all_models.png", args.title)
        plot_throughput(plot_rows, output_dir / f"{args.prefix}_throughput_all_models.png", args.title)
        plot_params_throughput(plot_rows, output_dir / f"{args.prefix}_params_throughput.png", args.title)
        plot_memory(plot_rows, output_dir / f"{args.prefix}_memory_all_models.png", args.title)
    if args.single_rd:
        print(f"wrote RD plot{' and extra metrics' if args.extra_metrics else ''} to {output_dir}")
        return
    plot_rd(plot_rows, output_dir / f"{args.prefix}_rd_all_models_zoom.png", args.title, zoom_bpp=args.zoom_bpp)
    plot_metrics(plot_rows, output_dir / f"{args.prefix}_all_models_metrics.png", args.title)
    print(f"wrote plots to {output_dir}")


def load_summary(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("results", "records", "summary"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported summary format: {path}")


def point_ok(row: dict) -> bool:
    value = psnr(row)
    return bpp(row) is not None and value is not None and bpp(row) > 0 and value >= 0


def bpp(row: dict) -> float | None:
    if row.get("model_name") == "GraphComp":
        value = row.get("scientific_bpp_with_side_info")
        if isinstance(value, (int, float)):
            return float(value)
    for key in ("bpp", "scientific_bpp", "scientific_bpp_with_side_info"):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def psnr(row: dict) -> float | None:
    for key in ("average_frame_psnr", "average_variable_psnr", "psnr"):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def normalize_row(row: dict) -> dict:
    item = dict(row)
    item["_curve"] = curve_name(row)
    item["_bpp"] = bpp(row)
    item["_psnr"] = psnr(row)
    item["_encode_throughput_MBps"] = throughput_mb_s(row, "encode")
    item["_decode_throughput_MBps"] = throughput_mb_s(row, "decode")
    item["_params"] = params(row, item["_curve"])
    item["_memory_usage_MB"] = metric_value(row, ("memory_usage_MB",))
    item["_memory_reserved_MB"] = metric_value(row, ("memory_reserved_MB",))
    item["_input_megapixels"] = input_megapixels(row)
    item["_input_label"] = input_label(row)
    return item


def curve_name(row: dict) -> str:
    model_name = row.get("model_name")
    model_id = str(row.get("model_id", ""))
    label = str(row.get("label", ""))
    source = str(row.get("source", ""))
    if model_name == "CAESAR":
        family = "CAESAR-D" if model_id.startswith("caesar_d") else "CAESAR-V"
        if str(row.get("caesar_postprocess", "")) == "none" or model_id.endswith("_no_pca"):
            return f"{family} no PCA"
        if source == "caesar_tuned":
            return f"{family} tuned"
        return f"{family} original"
    if model_name == "LIC-HPCM":
        return "HPCM-large" if "large" in model_id else "HPCM-base"
    if str(model_name).endswith("+CAESAR-PCA"):
        if label:
            return label
        if str(model_name).startswith("DCAE"):
            return "DCAE+CAESAR-PCA"
        return "HPCM-large+CAESAR-PCA" if "large" in model_id else "HPCM-base+CAESAR-PCA"
    if model_name == "cuSZ-Hi" and (
        model_id.startswith("cuSZ-Hi-3D") or row.get("cuszhi_whole3d") is True
    ):
        return "cuSZ-Hi-3D"
    if model_name in {"DCMVC", "DCVC-RT"}:
        mode = str(row.get("video_coding_mode") or "").lower()
        suffix = "P-frame" if mode == "pframe" or "pframe" in model_id.lower() else "I-frame"
        return f"{model_name} {suffix}"
    if model_name in {"DCAE", "cuSZ-Hi", "nvJPEG", "nvJPEG2000", "GraphComp", "visemz"}:
        return str(model_name)
    if label.startswith("HPCM"):
        return "HPCM-large" if "large" in label else "HPCM-base"
    return label or model_id


def grouped(rows: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["_curve"]].append(row)
    return dict(groups)


def ordered_group_names(groups: dict[str, list[dict]]) -> list[str]:
    return sorted(groups, key=lambda name: (CURVE_ORDER.get(name, 999), name))


def sort_points(items: list[dict]) -> list[dict]:
    return sorted(items, key=lambda r: (r["_bpp"], r.get("eb") if r.get("eb") is not None else 0))


def plot_rd(rows: list[dict], output: Path, title: str, zoom_bpp: float | None) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    groups = grouped(rows)
    for name in ordered_group_names(groups):
        items = sort_points(groups[name])
        if zoom_bpp is not None:
            items = [r for r in items if r["_bpp"] <= zoom_bpp]
        if not items:
            continue
        color, marker, linestyle = STYLES.get(name, ("#4b5563", "x", "-"))
        is_no_pca = "no PCA" in name
        ax.plot(
            [r["_bpp"] for r in items],
            [r["_psnr"] for r in items],
            label=name,
            color=color,
            marker=marker,
            linestyle=linestyle or "None",
            linewidth=2.0 if not is_no_pca else 2.4,
            markersize=6.5 if not is_no_pca else 8.5,
            markeredgewidth=0.8 if not is_no_pca else 1.4,
            markeredgecolor="white" if not is_no_pca else "black",
            zorder=10 if is_no_pca else 3,
        )
    ax.set_xlabel("Scientific BPP")
    ax.set_ylabel("PSNR (dB)")
    suffix = " (low-BPP zoom)" if zoom_bpp is not None else ""
    ax.set_title(f"{title} RD Comparison{suffix}", pad=8)
    ax.grid(True, which="major", alpha=0.22, linewidth=0.7)
    ax.grid(True, which="minor", alpha=0.10, linewidth=0.45)
    if zoom_bpp is None:
        ax.set_xscale("log")
    else:
        ax.set_xlim(left=0, right=zoom_bpp)
    ax.margins(x=0.04, y=0.08)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        fontsize=8.5,
        ncol=3,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.1,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, output)
    plt.close(fig)


def representative_rows(rows: list[dict]) -> list[dict]:
    reps = []
    for name, items in grouped(rows).items():
        candidates = [r for r in items if r["_bpp"] <= 1.0]
        if not candidates:
            candidates = items
        reps.append(max(candidates, key=lambda r: r["_psnr"]))
    return sorted(reps, key=lambda r: r["_curve"])


def representative_metric_rows(rows: list[dict], value_key: str, *, lower_is_better: bool = False) -> list[dict]:
    reps = []
    for name, items in grouped(rows).items():
        candidates = [r for r in items if isinstance(r.get(value_key), (int, float)) and np.isfinite(r[value_key])]
        if not candidates:
            continue
        selector = min if lower_is_better else max
        reps.append(selector(candidates, key=lambda r: r[value_key]))
    return sorted(reps, key=lambda r: r["_curve"])


def mean_metric_rows(rows: list[dict], value_key: str) -> list[dict]:
    reps = []
    for name, items in grouped(rows).items():
        candidates = [r for r in items if isinstance(r.get(value_key), (int, float)) and np.isfinite(r[value_key])]
        if not candidates:
            continue
        values = [float(r[value_key]) for r in candidates]
        item = dict(candidates[0])
        item[value_key] = float(np.mean(values))
        reps.append(item)
    return sorted(reps, key=lambda r: r["_curve"])


def metric_value(row: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            return float(value)
    return None


def throughput_mb_s(row: dict, kind: str) -> float | None:
    mbps_keys = (f"average_{kind}_throughput_MBps", f"{kind}_throughput_MBps")
    value = metric_value(row, mbps_keys)
    if value is not None:
        return value
    bytes_per_sec = metric_value(row, (f"average_{kind}_throughput", f"{kind}_throughput"))
    if bytes_per_sec is not None:
        return bytes_per_sec / 1e6
    return None


def params(row: dict, curve: str) -> float | None:
    value = metric_value(row, ("params",))
    if value is not None:
        return value
    return FALLBACK_PARAMS.get(curve)


def input_megapixels(row: dict) -> float | None:
    value = metric_value(row, ("input_megapixels",))
    if value is not None:
        return value
    shape = row.get("input_shape") or row.get("shape")
    if isinstance(shape, list) and len(shape) >= 2 and all(isinstance(x, (int, float)) for x in shape):
        return float(np.prod(shape)) / 1e6
    bytes_value = metric_value(row, ("input_bytes", "original_bytes"))
    if bytes_value is not None:
        return bytes_value / 4e6
    return None


def input_label(row: dict) -> str | None:
    shape = row.get("input_shape") or row.get("shape")
    parts = []
    if isinstance(shape, list) and shape:
        compact_shape = "x".join(str(int(x)) for x in shape if isinstance(x, (int, float)))
        if compact_shape:
            parts.append(compact_shape)
    bytes_value = metric_value(row, ("input_bytes", "original_bytes"))
    if bytes_value is not None:
        parts.append(f"{bytes_value / 1e6:.2g} MB input")
    return " / ".join(parts) if parts else None


def plot_bar_metric(
    rows: list[dict],
    output: Path,
    title: str,
    value_key: str,
    ylabel: str,
    *,
    lower_is_better: bool = False,
    log_y: bool = False,
) -> None:
    reps = mean_metric_rows(rows, value_key)
    groups = grouped(rows)
    fig, ax = plt.subplots(figsize=(max(7.2, len(reps) * 0.62), 4.8))
    if not reps:
        ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
    else:
        labels = [r["_curve"] for r in reps]
        values = [r[value_key] for r in reps]
        x = np.arange(len(reps))
        colors = [STYLES.get(r["_curve"], ("#4b5563",))[0] for r in reps]
        bars = ax.bar(x, values, color=colors, width=0.72)
        lower_errors = []
        upper_errors = []
        for row, value in zip(reps, values):
            curve_values = [
                float(item[value_key])
                for item in groups.get(row["_curve"], [])
                if isinstance(item.get(value_key), (int, float)) and np.isfinite(item[value_key])
            ]
            if isinstance(value, (int, float)) and np.isfinite(value) and curve_values:
                vmin = min(curve_values)
                vmax = max(curve_values)
                lower_errors.append(max(0.0, value - vmin))
                upper_errors.append(max(0.0, vmax - value))
            else:
                lower_errors.append(0.0)
                upper_errors.append(0.0)
        if any(err > 0 for err in lower_errors + upper_errors):
            ax.errorbar(
                x,
                values,
                yerr=np.array([lower_errors, upper_errors]),
                fmt="none",
                ecolor="#111111",
                elinewidth=1.35,
                capsize=4,
                capthick=1.2,
                zorder=3,
            )
        for bar, value, lo, hi in zip(bars, values, lower_errors, upper_errors):
            x_mid = bar.get_x() + bar.get_width() / 2
            ymin = value - lo if isinstance(value, (int, float)) and np.isfinite(value) else None
            ymax = value + hi if isinstance(value, (int, float)) and np.isfinite(value) else None
            if ymin is not None and ymax is not None and (lo > 0 or hi > 0):
                ax.text(x_mid, ymin, f"{ymin:.3g}", ha="center", va="top", fontsize=6.5, color="#111111")
                ax.text(x_mid, ymax, f"{ymax:.3g}", ha="center", va="bottom", fontsize=6.5, color="#111111")
            ax.text(
                x_mid,
                bar.get_height() + hi,
                f"{value:.3g}",
                ha="center",
                va="bottom" if hi == 0 else "center",
                fontsize=8,
                color="#374151",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        if log_y:
            ax.set_yscale("log")
    direction = "lower is better" if lower_is_better else "higher is better"
    ax.set_title(f"{title} Mean {ylabel} with Min-Max Range ({direction})", pad=8)
    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)


def plot_lpips(rows: list[dict], output: Path, title: str) -> None:
    plot_bar_metric(rows, output, title, "lpips", "LPIPS", lower_is_better=True, log_y=True)


def plot_throughput(rows: list[dict], output: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    for key, ylabel, ax in [
        ("_encode_throughput_MBps", "Encode Throughput (MB/s)", axes[0]),
        ("_decode_throughput_MBps", "Decode Throughput (MB/s)", axes[1]),
    ]:
        reps = representative_metric_rows(rows, key)
        if not reps:
            ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue
        labels = [r["_curve"] for r in reps]
        values = [r[key] for r in reps]
        x = np.arange(len(reps))
        colors = [STYLES.get(r["_curve"], ("#4b5563",))[0] for r in reps]
        bars = ax.bar(x, values, color=colors, width=0.72)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3g}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    fig.suptitle(f"{title} Representative Throughput (higher is better)", y=1.02)
    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)


def plot_memory(rows: list[dict], output: Path, title: str) -> None:
    reps = representative_metric_rows(rows, "_memory_usage_MB")
    fig, ax = plt.subplots(figsize=(max(11, len(reps) * 0.9), 6.5))
    if not reps:
        ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
    else:
        labels = [r["_curve"] for r in reps]
        values = [r["_memory_usage_MB"] for r in reps]
        input_labels = [r.get("_input_label") for r in reps]
        x = np.arange(len(reps))
        colors = [STYLES.get(r["_curve"], ("#4b5563",))[0] for r in reps]
        bars = ax.bar(x, values, color=colors, width=0.72)
        for bar, value, input_label_value in zip(bars, values, input_labels):
            suffix = "" if not input_label_value else f"\n{input_label_value}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3g} MB{suffix}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.set_ylabel("Process Memory Usage (MB)")
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.set_title(f"{title} Representative Memory Usage (input shape and bytes annotated)")
    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)


def plot_params_throughput(rows: list[dict], output: Path, title: str) -> None:
    reps = representative_rows(rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for row in reps:
        y = row.get("_encode_throughput_MBps")
        x = row.get("_params")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)) or not np.isfinite(y) or y <= 0:
            continue
        color, marker, _ = STYLES.get(row["_curve"], ("#4b5563", "x", "-"))
        ax.scatter(max(x, 1), y, label=row["_curve"], color=color, marker=marker, s=70)
        ax.annotate(row["_curve"], (max(x, 1), y), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel("Model Parameters (log scale, external codecs shown at 1)")
    ax.set_ylabel("Representative Encode Throughput (MB/s)")
    ax.set_title(f"{title} Parameters vs Encode Throughput")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25, linestyle="--")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=8, ncol=2)
    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)


def plot_metrics(rows: list[dict], output: Path, title: str) -> None:
    groups = grouped(rows)
    reps_by_key = {
        "_psnr": mean_metric_rows(rows, "_psnr"),
        "_bpp": mean_metric_rows(rows, "_bpp"),
        "_encode_throughput_MBps": mean_metric_rows(rows, "_encode_throughput_MBps"),
        "_decode_throughput_MBps": mean_metric_rows(rows, "_decode_throughput_MBps"),
    }
    all_labels = ordered_group_names(groups)
    fig, axes = plt.subplots(2, 2, figsize=(max(12, len(all_labels) * 0.8), 8.5))
    specs = [
        ("_psnr", "PSNR (dB)", axes[0][0], False),
        ("_bpp", "BPP", axes[0][1], True),
        ("_encode_throughput_MBps", "Encode Throughput (MB/s)", axes[1][0], True),
        ("_decode_throughput_MBps", "Decode Throughput (MB/s)", axes[1][1], True),
    ]
    for key_spec, ylabel, ax, log_y in specs:
        rep_map = {r["_curve"]: r for r in reps_by_key[key_spec]}
        labels = [label for label in all_labels if label in rep_map]
        reps = [rep_map[label] for label in labels]
        values = [r.get(key_spec) for r in reps]
        x = np.arange(len(labels))
        colors = [STYLES.get(label, ("#4b5563",))[0] for label in labels]
        plotted = [v if isinstance(v, (int, float)) and np.isfinite(v) else 0 for v in values]
        bars = ax.bar(x, plotted, color=colors, width=0.7)
        lower_errors = []
        upper_errors = []
        for label, value in zip(labels, values):
            curve_values = [
                float(item[key_spec])
                for item in groups.get(label, [])
                if isinstance(item.get(key_spec), (int, float)) and np.isfinite(item[key_spec])
            ]
            if isinstance(value, (int, float)) and np.isfinite(value) and curve_values:
                vmin = min(curve_values)
                vmax = max(curve_values)
                lower_errors.append(max(0.0, value - vmin))
                upper_errors.append(max(0.0, vmax - value))
            else:
                lower_errors.append(0.0)
                upper_errors.append(0.0)
        if any(err > 0 for err in lower_errors + upper_errors):
            ax.errorbar(
                x,
                plotted,
                yerr=np.array([lower_errors, upper_errors]),
                fmt="none",
                ecolor="#111111",
                elinewidth=1.35,
                capsize=4,
                capthick=1.2,
                zorder=3,
            )
        for bar, value, lo, hi in zip(bars, values, lower_errors, upper_errors):
            text = "n/a" if value is None or not np.isfinite(value) else f"{value:.3g}"
            x_mid = bar.get_x() + bar.get_width() / 2
            top = bar.get_height() + hi
            if isinstance(value, (int, float)) and np.isfinite(value) and (lo > 0 or hi > 0):
                ymin = value - lo
                ymax = value + hi
                ax.text(x_mid, ymin, f"{ymin:.3g}", ha="center", va="top", fontsize=6.2, color="#111111")
                ax.text(x_mid, ymax, f"{ymax:.3g}", ha="center", va="bottom", fontsize=6.2, color="#111111")
            ax.text(x_mid, top, text, ha="center", va="bottom" if hi == 0 else "center", fontsize=7, color="#374151")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(True, axis="y", alpha=0.2)
        if log_y and any(v > 0 for v in plotted):
            ax.set_yscale("log")
    fig.suptitle(f"{title} Mean Metrics with Min-Max Range", y=1.01)
    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11.5,
            "axes.linewidth": 0.85,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig, output: Path) -> None:
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.03)
    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)


if __name__ == "__main__":
    main()
