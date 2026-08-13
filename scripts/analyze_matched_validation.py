#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROOT = Path("/workspace/tmp/aifc_matched_validation")
COLORS = {
    "DCAE": "#087E8B",
    "HPCM": "#4C78A8",
    "CAESAR-V": "#E69500",
    "CAESAR-D": "#A9561E",
    "cuSZ-Hi": "#111111",
    "nvJPEG2000": "#6F4E9C",
}
MARKERS = {"DCAE": "D", "HPCM": "o", "CAESAR-V": "s", "CAESAR-D": "P", "cuSZ-Hi": "x", "nvJPEG2000": "^"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze checksum-matched GPU validation results.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def curve_name(row: dict[str, Any]) -> str:
    if row.get("model_name") == "DCAE":
        return "DCAE"
    if row.get("model_name") == "LIC-HPCM":
        return "HPCM"
    if row.get("model_name") == "CAESAR":
        return "CAESAR-V" if str(row.get("model_id", "")).startswith("caesar_v") else "CAESAR-D"
    return str(row.get("model_name"))


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def valid_row(row: dict[str, Any]) -> bool:
    if "error" in row or not finite(row.get("scientific_bpp_with_side_info")) or not finite(row.get("psnr")):
        return False
    if row.get("model_name") == "cuSZ-Hi" and row.get("error_bound_satisfied") is not True:
        return False
    return float(row["scientific_bpp_with_side_info"]) > 0


def frontier(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    rows = sorted((row for row in rows if finite(row.get(metric))), key=lambda row: float(row["scientific_bpp_with_side_info"]))
    kept = []
    best = -float("inf")
    for row in rows:
        quality = float(row[metric])
        if quality > best + 1e-8:
            kept.append(row)
            best = quality
    return kept


def interpolate(rows: list[dict[str, Any]], anchors: np.ndarray, metric: str) -> np.ndarray:
    xs = np.log10([float(row["scientific_bpp_with_side_info"]) for row in rows])
    ys = [float(row[metric]) for row in rows]
    return np.interp(np.log10(anchors), xs, ys)


def pairwise_comparisons(curves: dict[str, list[dict[str, Any]]], metric: str) -> list[dict[str, Any]]:
    output = []
    names = sorted(curves)
    for index, left_name in enumerate(names):
        left = frontier(curves[left_name], metric)
        if len(left) < 2:
            continue
        for right_name in names[index + 1 :]:
            right = frontier(curves[right_name], metric)
            if len(right) < 2:
                continue
            low = max(float(left[0]["scientific_bpp_with_side_info"]), float(right[0]["scientific_bpp_with_side_info"]))
            high = min(float(left[-1]["scientific_bpp_with_side_info"]), float(right[-1]["scientific_bpp_with_side_info"]))
            if high <= low * 1.05:
                continue
            anchors = np.geomspace(low, high, 7)
            delta = interpolate(left, anchors, metric) - interpolate(right, anchors, metric)
            output.append({
                "left": left_name, "right": right_name, "bpp_low": low, "bpp_high": high,
                "mean_delta_db": float(np.mean(delta)), "min_delta_db": float(np.min(delta)), "max_delta_db": float(np.max(delta)),
            })
    return output


def plot_rd(dataset_id: str, curves: dict[str, list[dict[str, Any]]], output: Path, metric: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    for name in COLORS:
        rows = frontier(curves.get(name, []), metric)
        if not rows:
            continue
        ax.plot(
            [row["scientific_bpp_with_side_info"] for row in rows], [row[metric] for row in rows],
            color=COLORS[name], marker=MARKERS[name], linewidth=2.0, markersize=6, label=name,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Scientific BPP (including side information)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{dataset_id}: checksum-matched RD")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_throughput(dataset_id: str, curves: dict[str, list[dict[str, Any]]], output: Path) -> None:
    names, medians, lows, highs = [], [], [], []
    for name in COLORS:
        values = [float(row["sample_wall_throughput_MBps"]) for row in curves.get(name, []) if finite(row.get("sample_wall_throughput_MBps"))]
        if not values:
            continue
        names.append(name)
        medians.append(statistics.median(values))
        lows.append(min(values))
        highs.append(max(values))
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.bar(x, medians, color=[COLORS[name] for name in names], width=0.68)
    ax.errorbar(x, medians, yerr=[np.array(medians) - np.array(lows), np.array(highs) - np.array(medians)], fmt="none", color="#333333", capsize=4)
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylabel("Wall roundtrip throughput (MB/s)")
    ax.set_title(f"{dataset_id}: matched integration throughput")
    ax.grid(True, axis="y", alpha=0.22)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def fmt(value: Any, digits: int = 4) -> str:
    return "n/a" if not finite(value) else f"{float(value):.{digits}g}"


def main() -> None:
    args = parse_args()
    analysis_dir = args.root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    datasets = []
    all_rows = []
    for summary_path in sorted(args.root.glob("*/summary.json")):
        dataset_id = summary_path.parent.name
        manifest_path = summary_path.parent / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        rows = json.loads(summary_path.read_text())
        hashes = {row.get("canonical_sha256") for row in rows}
        checksum_ok = hashes == {manifest["canonical_sha256"]}
        curves: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            if valid_row(row):
                curves.setdefault(curve_name(row), []).append(row)
        invalid = [row for row in rows if not valid_row(row)]
        curve_summary = []
        for name, curve_rows in sorted(curves.items()):
            bpps = [float(row["scientific_bpp_with_side_info"]) for row in curve_rows]
            wall = [float(row["sample_wall_throughput_MBps"]) for row in curve_rows if finite(row.get("sample_wall_throughput_MBps"))]
            curve_summary.append({
                "curve": name, "valid_points": len(curve_rows), "bpp_min": min(bpps), "bpp_max": max(bpps),
                "psnr_min": min(float(row["psnr"]) for row in curve_rows), "psnr_max": max(float(row["psnr"]) for row in curve_rows),
                "average_variable_psnr_min": min(float(row["average_variable_psnr"]) for row in curve_rows if finite(row.get("average_variable_psnr"))),
                "average_variable_psnr_max": max(float(row["average_variable_psnr"]) for row in curve_rows if finite(row.get("average_variable_psnr"))),
                "wall_throughput_MBps_median": statistics.median(wall) if wall else None,
                "wall_throughput_MBps_min": min(wall) if wall else None,
                "wall_throughput_MBps_max": max(wall) if wall else None,
            })
        plot_rd(dataset_id, curves, analysis_dir / f"{dataset_id}_rd_global.png", "psnr", "Global PSNR (dB)")
        plot_rd(dataset_id, curves, analysis_dir / f"{dataset_id}_rd_variable.png", "average_variable_psnr", "Average-variable PSNR (dB)")
        plot_throughput(dataset_id, curves, analysis_dir / f"{dataset_id}_throughput.png")
        entry = {
            "dataset_id": dataset_id, "canonical_sha256": manifest["canonical_sha256"], "checksum_ok": checksum_ok,
            "canonical_shape": manifest["canonical_shape"], "selection": manifest["selection"],
            "total_rows": len(rows), "valid_rows": sum(len(value) for value in curves.values()), "invalid_rows": len(invalid),
            "invalid_points": [{"model_id": row.get("model_id"), "control": row.get("control"), "reason": row.get("error", "error-bound violation")} for row in invalid],
            "curves": curve_summary,
            "pairwise_global_psnr": pairwise_comparisons(curves, "psnr"),
            "pairwise_average_variable_psnr": pairwise_comparisons(curves, "average_variable_psnr"),
        }
        datasets.append(entry)
        all_rows.extend(rows)

    payload = {"root": str(args.root), "datasets": datasets, "total_rows": len(all_rows)}
    (analysis_dir / "matched_analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# GPU 同输入压缩验证摘要", "",
        "所有可比曲线均要求 canonical SHA256 一致；cuSZ-Hi 还必须实际满足请求的绝对误差界。", "",
    ]
    for entry in datasets:
        lines.extend([
            f"## {entry['dataset_id']}", "",
            f"- canonical shape：`{entry['canonical_shape']}`",
            f"- 输入选择：{entry['selection']}",
            f"- checksum：{'通过' if entry['checksum_ok'] else '不通过'}",
            f"- 记录：{entry['valid_rows']} 个有效点，{entry['invalid_rows']} 个无效或失败点", "",
            "| 曲线 | 有效点 | BPP 范围 | 全局 PSNR 范围 | 逐变量 PSNR 范围 | wall 吞吐量中位数 MB/s |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in entry["curves"]:
            lines.append(
                f"| {row['curve']} | {row['valid_points']} | {fmt(row['bpp_min'])}-{fmt(row['bpp_max'])} | "
                f"{fmt(row['psnr_min'])}-{fmt(row['psnr_max'])} | "
                f"{fmt(row['average_variable_psnr_min'])}-{fmt(row['average_variable_psnr_max'])} | "
                f"{fmt(row['wall_throughput_MBps_median'])} |"
            )
        if entry["invalid_points"]:
            lines.extend(["", "无效点："])
            for row in entry["invalid_points"]:
                reason = str(row["reason"]).splitlines()[-1]
                lines.append(f"- `{row['model_id']}`（control={row['control']}）：{reason[:180]}")
        lines.append("")
    (analysis_dir / "matched_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(analysis_dir / "matched_analysis.md")


if __name__ == "__main__":
    main()
