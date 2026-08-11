#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_objective_benchmark import model_family, row_gates


DEFAULT_PROTOCOL = PROJECT_ROOT / "benchmark_protocols/objective_v1.json"
COLORS = {
    "DCAE": "#087E8B", "LIC-HPCM-base": "#4C78A8", "LIC-HPCM-large": "#2F4B7C",
    "CAESAR-V": "#E69500", "CAESAR-D": "#A9561E", "cuSZ-Hi": "#111111",
    "CAESAR-V-Turb-tuned": "#F2C14E", "CAESAR-D-Turb-tuned": "#7A3E00",
    "nvJPEG2000": "#6F4E9C", "nvJPEG": "#777777", "DCMVC-I": "#009E73", "DCVC-RT-I": "#CC476B",
    "DCMVC-IP": "#0072B2", "DCVC-RT-IP": "#D55E00",
}
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h", "*", "<", ">")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate objective-v1 samples into corpus-level results.")
    parser.add_argument("--root", type=Path, default=Path("unified_results/objective_v1"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def curve_name(row: dict[str, Any]) -> str:
    family = model_family(row)
    model_id = str(row.get("model_id", ""))
    if family == "LIC-HPCM":
        return "LIC-HPCM-large" if "large" in model_id.lower() else "LIC-HPCM-base"
    return family


def point_key(row: dict[str, Any]) -> tuple[str, str]:
    return curve_name(row), str(row.get("model_id"))


def valid_row(row: dict[str, Any], protocol: dict[str, Any]) -> bool:
    gates = row_gates(row, protocol)
    return all(gates.values())


def aggregate_point(rows: list[dict[str, Any]], expected_samples: set[str]) -> dict[str, Any] | None:
    corpus_rows = [
        row for row in rows
        if set(str(item) for item in row.get("covered_canonical_sample_ids", [])) == expected_samples
    ]
    if corpus_rows:
        if len(rows) != 1:
            return None
        ordered = corpus_rows
    else:
        by_sample = {str(row["canonical_sample_id"]): row for row in rows}
        if set(by_sample) != expected_samples:
            return None
        ordered = [by_sample[sample_id] for sample_id in sorted(expected_samples)]
    symbols = sum(int(row["canonical_symbol_count"]) for row in ordered)
    valid_symbols = sum(int(row["canonical_valid_symbol_count"]) for row in ordered)
    total_bytes = sum(float(row["total_bytes_with_side_info"]) for row in ordered)
    sse = sum(float(row["normalized_mse"]) * int(row["canonical_valid_symbol_count"]) for row in ordered)
    mse = sse / valid_symbols
    timing_count = min(len(row["timing_repetitions"]) for row in ordered)
    wall_repeats = [
        sum(float(row["timing_repetitions"][index]["roundtrip_seconds"]) for row in ordered)
        for index in range(timing_count)
    ]
    encode_repeats = [
        sum(float(row["timing_repetitions"][index].get("encode_seconds", 0.0)) for row in ordered)
        for index in range(timing_count)
    ]
    decode_repeats = [
        sum(float(row["timing_repetitions"][index].get("decode_seconds", 0.0)) for row in ordered)
        for index in range(timing_count)
    ]
    lpips_rows = [row for row in ordered if isinstance(row.get("lpips"), (int, float))]
    lpips = (
        sum(float(row["lpips"]) * int(row["canonical_symbol_count"]) for row in lpips_rows)
        / sum(int(row["canonical_symbol_count"]) for row in lpips_rows)
        if lpips_rows else None
    )
    psnr_samples = []
    if len(ordered) > 1:
        rng = np.random.default_rng(20260722)
        for _ in range(1000):
            chosen = rng.choice(ordered, size=len(ordered), replace=True)
            chosen_valid = sum(int(row["canonical_valid_symbol_count"]) for row in chosen)
            chosen_sse = sum(float(row["normalized_mse"]) * int(row["canonical_valid_symbol_count"]) for row in chosen)
            psnr_samples.append(-10.0 * math.log10(max(chosen_sse / chosen_valid, 1e-30)))
    return {
        "curve": curve_name(ordered[0]),
        "model_id": ordered[0]["model_id"],
        "control": ordered[0].get("control"),
        "sample_count": len(expected_samples),
        "canonical_symbol_count": symbols,
        "scientific_bpp_with_side_info": total_bytes * 8.0 / symbols,
        "normalized_mse": mse,
        "normalized_psnr": -10.0 * math.log10(max(mse, 1e-30)),
        "normalized_psnr_ci95": [float(np.percentile(psnr_samples, 2.5)), float(np.percentile(psnr_samples, 97.5))] if psnr_samples else None,
        "lpips": lpips,
        "wall_seconds_median": statistics.median(wall_repeats),
        "wall_seconds_p10": float(np.percentile(wall_repeats, 10)),
        "wall_seconds_p90": float(np.percentile(wall_repeats, 90)),
        "wall_throughput_MBps": symbols * 4 / statistics.median(wall_repeats) / 1e6,
        "encode_throughput_MBps": (
            symbols * 4 / statistics.median(encode_repeats) / 1e6
            if statistics.median(encode_repeats) > 0 else None
        ),
        "decode_throughput_MBps": (
            symbols * 4 / statistics.median(decode_repeats) / 1e6
            if statistics.median(decode_repeats) > 0 else None
        ),
        "peak_memory_MB": max(
            (float(row["memory_usage_MB"]) for row in ordered if isinstance(row.get("memory_usage_MB"), (int, float))),
            default=None,
        ),
    }


def pareto_partition(points: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept = []
    rejected = []
    for curve in sorted({point["curve"] for point in points}):
        curve_points = [point for point in points if point["curve"] == curve]
        seen_coordinates = set()
        for point in sorted(curve_points, key=lambda row: (row["scientific_bpp_with_side_info"], -row["normalized_psnr"], row["model_id"])):
            bpp = float(point["scientific_bpp_with_side_info"])
            psnr = float(point["normalized_psnr"])
            coordinate = (round(bpp, 12), round(psnr, 9))
            duplicate = coordinate in seen_coordinates
            dominated = any(
                float(other["scientific_bpp_with_side_info"]) <= bpp + 1e-12
                and float(other["normalized_psnr"]) >= psnr - 1e-9
                and (
                    float(other["scientific_bpp_with_side_info"]) < bpp - 1e-12
                    or float(other["normalized_psnr"]) > psnr + 1e-9
                )
                for other in curve_points
            )
            if duplicate or dominated:
                rejected.append({**point, "pareto_reason": "duplicate" if duplicate else "dominated"})
            else:
                kept.append(point)
                seen_coordinates.add(coordinate)
    return kept, rejected


def plot_dataset(dataset_id: str, points: list[dict[str, Any]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)
    for index, curve in enumerate(sorted({row["curve"] for row in points})):
        rows = sorted((row for row in points if row["curve"] == curve), key=lambda row: row["scientific_bpp_with_side_info"])
        if not rows:
            continue
        ax.plot(
            [row["scientific_bpp_with_side_info"] for row in rows],
            [row["normalized_psnr"] for row in rows],
            marker=MARKERS[index % len(MARKERS)], linewidth=1.8, markersize=5.5,
            color=COLORS.get(curve, plt.get_cmap("tab20")(index)), label=curve,
        )
    ax.set_xscale("log")
    ax.set_xlabel("BPP including required side information")
    ax.set_ylabel("Dataset-normalized PSNR (dB)")
    ax.set_title(f"{dataset_id} · Objective-v1 corpus RD", loc="left", fontweight="semibold")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_metric_ranges(
    dataset_id: str,
    points: list[dict[str, Any]],
    field: str,
    ylabel: str,
    output: Path,
    log_scale: bool = True,
) -> None:
    grouped = []
    for curve in sorted({row["curve"] for row in points}):
        values = [float(row[field]) for row in points if row["curve"] == curve and isinstance(row.get(field), (int, float)) and float(row[field]) > 0]
        if values:
            grouped.append((curve, min(values), statistics.mean(values), max(values)))
    if not grouped:
        return
    fig_height = max(3.8, 0.42 * len(grouped) + 1.8)
    fig, ax = plt.subplots(figsize=(8.8, fig_height), constrained_layout=True)
    for index, (curve, minimum, mean, maximum) in enumerate(grouped):
        color = COLORS.get(curve, plt.get_cmap("tab20")(index))
        ax.errorbar(
            mean, index, xerr=[[mean - minimum], [maximum - mean]], fmt="o", color=color,
            ecolor=color, elinewidth=2.2, capsize=4, markersize=6,
        )
        ax.annotate(f"{minimum:.3g}", (minimum, index), xytext=(-5, 7), textcoords="offset points", ha="right", fontsize=7.5)
        ax.annotate(f"{maximum:.3g}", (maximum, index), xytext=(5, 7), textcoords="offset points", ha="left", fontsize=7.5)
    ax.set_yticks(range(len(grouped)), [item[0] for item in grouped])
    if log_scale:
        ax.set_xscale("log")
    ax.set_xlabel(ylabel)
    ax.set_title(f"{dataset_id} · Mean and measured range", loc="left", fontweight="semibold")
    ax.grid(True, axis="x", which="both", alpha=0.22)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_lpips(dataset_id: str, points: list[dict[str, Any]], output: Path) -> None:
    curves = sorted({row["curve"] for row in points if isinstance(row.get("lpips"), (int, float))})
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)
    for index, curve in enumerate(curves):
        rows = sorted(
            (row for row in points if row["curve"] == curve and isinstance(row.get("lpips"), (int, float))),
            key=lambda row: row["scientific_bpp_with_side_info"],
        )
        ax.plot(
            [row["scientific_bpp_with_side_info"] for row in rows], [row["lpips"] for row in rows],
            marker=MARKERS[index % len(MARKERS)], linewidth=1.8, markersize=5.5,
            color=COLORS.get(curve, plt.get_cmap("tab20")(index)), label=curve,
        )
    ax.set_xscale("log")
    ax.set_xlabel("BPP including required side information")
    ax.set_ylabel("LPIPS (lower is better)")
    ax.set_title(f"{dataset_id} · Perceptual RD", loc="left", fontweight="semibold")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def fmt_range(values: list[float], digits: int = 3) -> str:
    if not values:
        return "—"
    return f"{min(values):.{digits}g}–{max(values):.{digits}g}"


def write_html(payload: dict[str, Any], root: Path) -> Path:
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    categories = [
        ("general", "通用图像与视频", ["kodak", "uvg_twilight_1080p"]),
        ("scientific-images", "科学图像", ["tomo", "s2c", "lysozyme"]),
        ("scientific-fields", "科学场数据", ["e3sm_npz", "era5_npy", "hurricane", "nyx", "turb_rot_npz"]),
    ]
    by_dataset = {item["dataset_id"]: item for item in payload["datasets"]}
    dataset_count = len(payload["datasets"])
    complete_dataset_count = sum(
        not item["invalid_rows"] and not item["incomplete_points"]
        for item in payload["datasets"]
    )
    raw_row_count = sum(int(item["raw_rows"]) for item in payload["datasets"])
    valid_row_count = sum(int(item["valid_rows"]) for item in payload["datasets"])
    sections = []
    for category_id, category_label, dataset_ids in categories:
        blocks = []
        for dataset_id in dataset_ids:
            dataset = by_dataset[dataset_id]
            points = dataset["pareto_points"]
            rows = []
            for curve in sorted({point["curve"] for point in points}):
                curve_points = [point for point in points if point["curve"] == curve]
                bpps = [float(point["scientific_bpp_with_side_info"]) for point in curve_points]
                psnrs = [float(point["normalized_psnr"]) for point in curve_points]
                lpips_values = [float(point["lpips"]) for point in curve_points if isinstance(point.get("lpips"), (int, float))]
                wall = [float(point["wall_throughput_MBps"]) for point in curve_points if isinstance(point.get("wall_throughput_MBps"), (int, float))]
                encode = [float(point["encode_throughput_MBps"]) for point in curve_points if isinstance(point.get("encode_throughput_MBps"), (int, float))]
                decode = [float(point["decode_throughput_MBps"]) for point in curve_points if isinstance(point.get("decode_throughput_MBps"), (int, float))]
                memory = [float(point["peak_memory_MB"]) for point in curve_points if isinstance(point.get("peak_memory_MB"), (int, float))]
                rows.append(
                    "<tr>"
                    f"<th>{html.escape(curve)}</th><td>{len(curve_points)}</td><td>{fmt_range(bpps)}</td>"
                    f"<td>{fmt_range(psnrs)}</td><td>{fmt_range(lpips_values)}</td>"
                    f"<td>{statistics.mean(wall):.3g} ({fmt_range(wall)})</td>"
                    f"<td>{statistics.mean(encode):.3g}</td><td>{statistics.mean(decode):.3g}</td>"
                    f"<td>{fmt_range(memory, 4)}</td></tr>"
                )
            image_items = [
                (f"../analysis/{dataset_id}_objective_rd.png", "率失真"),
                (f"../analysis/{dataset_id}_throughput_range.png", "端到端吞吐量"),
                (f"../analysis/{dataset_id}_memory_range.png", "峰值显存"),
            ]
            if any(isinstance(point.get("lpips"), (int, float)) for point in points):
                image_items.append((f"../analysis/{dataset_id}_lpips_rd.png", "LPIPS"))
            figures = "".join(
                f'<figure><img src="{src}" alt="{html.escape(dataset_id)} {label}"><figcaption>{label}</figcaption></figure>'
                for src, label in image_items
            )
            blocks.append(f"""
              <article id="{html.escape(dataset_id)}">
                <header><h3>{html.escape(dataset_id)}</h3><p>{dataset['raw_rows']} 条合规记录 · {len(dataset['points'])} 个完整 corpus 点 · {len(dataset['pareto_points'])} 个主图点</p></header>
                <div class="figures">{figures}</div>
                <div class="table-wrap"><table><thead><tr><th>方法</th><th>点</th><th>BPP 范围</th><th>固定尺度 PSNR</th><th>LPIPS</th><th>端到端 MB/s 平均（范围）</th><th>编码 MB/s</th><th>解码 MB/s</th><th>峰值显存 MB</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
              </article>
            """)
        sections.append(f'<section id="{category_id}"><h2>{category_label}</h2>{"".join(blocks)}</section>')
    nav_links = "".join(f'<a href="#{category_id}">{label}</a>' for category_id, label, _ in categories)
    content = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AIForCompression Objective-v1</title><style>
:root{{--ink:#181b20;--muted:#646b75;--line:#d9dde3;--paper:#fff;--wash:#f4f6f8;--accent:#087e8b;--warm:#a9561e}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;color:var(--ink);background:var(--paper);font-family:Inter,"Noto Sans SC","Microsoft YaHei",sans-serif;letter-spacing:0;line-height:1.5}}
nav{{position:sticky;top:0;z-index:4;display:flex;align-items:center;gap:22px;padding:11px max(20px,calc((100vw - 1440px)/2));background:rgba(255,255,255,.96);border-bottom:1px solid var(--line)}}nav strong{{margin-right:auto}}nav a{{color:var(--muted);text-decoration:none;font-size:14px}}nav a:hover{{color:var(--accent)}}
main{{max-width:1440px;margin:0 auto;padding:34px 28px 80px}}h1{{font-size:34px;margin:0 0 8px}}.lede{{max-width:980px;color:var(--muted);margin:0 0 24px}}.summary{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0;border-block:1px solid var(--line);margin-bottom:30px}}.metric{{padding:18px 20px;border-right:1px solid var(--line)}}.metric:last-child{{border:0}}.metric b{{display:block;font-size:25px}}.metric span{{font-size:13px;color:var(--muted)}}
.method{{padding:18px 20px;background:var(--wash);border-left:4px solid var(--accent);margin:0 0 44px}}.method p{{margin:4px 0}}section{{margin-top:52px}}h2{{font-size:25px;border-bottom:2px solid var(--ink);padding-bottom:8px;margin-bottom:32px}}article{{padding:0 0 54px;margin-bottom:44px;border-bottom:1px solid var(--line)}}article header{{display:flex;align-items:baseline;gap:20px}}h3{{font-size:21px;margin:0}}article header p{{color:var(--muted);font-size:13px;margin:0}}
.figures{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px;margin:18px 0 24px}}figure{{margin:0;min-width:0}}figure img{{display:block;width:100%;height:auto;border:1px solid var(--line)}}figcaption{{font-size:12px;color:var(--muted);margin-top:5px}}.table-wrap{{overflow:auto;border-block:1px solid var(--line)}}table{{border-collapse:collapse;width:100%;min-width:1080px;font-size:12px}}th,td{{padding:9px 10px;text-align:right;border-bottom:1px solid #eceef1;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}thead th{{color:var(--muted);font-weight:600;background:var(--wash)}}tbody th{{font-weight:600}}
footer{{color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:16px}}footer a{{color:var(--accent)}}
@media(max-width:800px){{nav{{overflow:auto}}nav strong{{display:none}}main{{padding:24px 15px 60px}}h1{{font-size:28px}}.summary{{grid-template-columns:repeat(2,1fr)}}.metric:nth-child(2){{border-right:0}}.figures{{grid-template-columns:1fr}}article header{{display:block}}}}
</style></head><body><nav><strong>Objective-v1</strong>{nav_links}</nav><main>
<h1>统一压缩基准结果</h1><p class="lede">同一数据集的所有 codec 从相同 canonical float32 数值、相同 crop、相同 mask 和相同数据集级外部归一化开始。codec 内部归一化、PCA、predictor、padding 与熵模型保持原实现。</p>
<div class="summary"><div class="metric"><b>{complete_dataset_count} / {dataset_count}</b><span>完整数据集</span></div><div class="metric"><b>{valid_row_count} / {raw_row_count}</b><span>严格合规记录</span></div><div class="metric"><b>2 + 5</b><span>预热 + 正式重复</span></div><div class="metric"><b>64</b><span>CAESAR 官方推理 batch</span></div></div>
<div class="method"><p><strong>主质量指标：</strong>数据集固定单位范围上的 normalized PSNR；BPP 包含必要辅助信息。</p><p><strong>吞吐量：</strong>canonical host tensor 到内存 bitstream 再回到 canonical host tensor，不含磁盘 I/O、模型加载和指标计算。</p><p><strong>LPIPS：</strong>仅 RGB 通用媒体主轨报告；科学数据暂不把未定义的伪彩色渲染作为排名指标。</p></div>
{''.join(sections)}
<footer>原始汇总：<a href="../combined_summary.json">combined_summary.json</a> · 审计：<a href="../objective_protocol_audit.json">objective_protocol_audit.json</a> · 分析：<a href="../analysis/objective_analysis.json">objective_analysis.json</a></footer>
</main></body></html>"""
    output = report_dir / "index.html"
    output.write_text(content, encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    analysis_dir = args.root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    datasets = []
    for dataset_id, contract in protocol["datasets"].items():
        summary_path = args.root / dataset_id / "summary.json"
        rows = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
        expected_samples = set(contract["objective_samples"])
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        invalid = []
        for row in rows:
            if valid_row(row, protocol):
                groups[point_key(row)].append(row)
            else:
                invalid.append({"model_id": row.get("model_id"), "sample_id": row.get("canonical_sample_id"), "error": row.get("error")})
        points = []
        incomplete = []
        for key, point_rows in sorted(groups.items()):
            point = aggregate_point(point_rows, expected_samples)
            if point is None:
                incomplete.append({"curve": key[0], "model_id": key[1], "present_samples": sorted({row["canonical_sample_id"] for row in point_rows})})
            else:
                points.append(point)
        pareto_points, dominated_points = pareto_partition(points)
        if pareto_points:
            plot_dataset(dataset_id, pareto_points, analysis_dir / f"{dataset_id}_objective_rd.png")
            plot_metric_ranges(
                dataset_id, pareto_points, "wall_throughput_MBps", "End-to-end throughput (MB/s)",
                analysis_dir / f"{dataset_id}_throughput_range.png",
            )
            plot_metric_ranges(
                dataset_id, pareto_points, "peak_memory_MB", "Peak allocated memory (MB)",
                analysis_dir / f"{dataset_id}_memory_range.png",
            )
            plot_lpips(dataset_id, pareto_points, analysis_dir / f"{dataset_id}_lpips_rd.png")
        datasets.append({
            "dataset_id": dataset_id, "expected_samples": sorted(expected_samples), "raw_rows": len(rows),
            "valid_rows": sum(len(value) for value in groups.values()), "invalid_rows": invalid,
            "incomplete_points": incomplete, "points": points, "pareto_points": pareto_points,
            "dominated_points": dominated_points,
        })
    payload = {"protocol_id": protocol["protocol_id"], "datasets": datasets}
    output_json = analysis_dir / "objective_analysis.json"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Objective-v1 汇总", "", "只有覆盖完整 objective sample 清单并通过全部合规 gate 的点才进入 corpus 曲线。", ""]
    for dataset in datasets:
        lines.extend([
            f"## {dataset['dataset_id']}", "",
            f"- 原始记录：{dataset['raw_rows']}",
            f"- 完整 corpus 点：{len(dataset['points'])}",
            f"- Pareto 主图点：{len(dataset['pareto_points'])}",
            f"- 被支配/重复点：{len(dataset['dominated_points'])}",
            f"- 无效记录：{len(dataset['invalid_rows'])}",
            f"- sample 不完整点：{len(dataset['incomplete_points'])}", "",
        ])
    output_md = analysis_dir / "objective_analysis.md"
    output_md.write_text("\n".join(lines), encoding="utf-8")
    output_html = write_html(payload, args.root)
    print(output_json)
    print(output_md)
    print(output_html)


if __name__ == "__main__":
    main()
