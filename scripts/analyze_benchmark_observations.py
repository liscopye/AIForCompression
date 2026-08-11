#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SUMMARY = Path(
    "unified_results/final/"
    "all_models_fullstack_cuszhi_nvjpeg_pca_pcaanchored_nozero_zoom6_uvg_turb/"
    "combined_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit cross-codec benchmark comparability and ranges.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=Path("unified_results/analysis/benchmark_observations"))
    return parser.parse_args()


def model_family(row: dict[str, Any]) -> str:
    name = str(row.get("model_name", ""))
    label = str(row.get("label", ""))
    if "CAESAR-PCA" in name or "PCA" in label:
        return "AI+CAESAR-PCA"
    if name == "CAESAR":
        return "CAESAR"
    if name in {"DCAE", "LIC-HPCM", "DCMVC", "DCVC-RT"}:
        return "AI image/video"
    if name in {"cuSZ-Hi", "nvJPEG", "nvJPEG2000"}:
        return "traditional/scientific codec"
    return "other"


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def median(values: Iterable[Any]) -> float | None:
    valid = [float(value) for value in values if finite_number(value)]
    return statistics.median(valid) if valid else None


def shape_signature(row: dict[str, Any]) -> str:
    shape = row.get("shape") or row.get("input_shape")
    if isinstance(shape, list) and shape and isinstance(shape[0], list):
        unique = sorted({tuple(item) for item in shape if isinstance(item, list)})
        shape_text = "+".join(f"{list(item)}" for item in unique)
    else:
        shape_text = str(shape)
    count = row.get("sample_count") or row.get("success_count") or 1
    mode = row.get("cuszhi_sample_mode") or row.get("model_view") or ""
    return f"shape={shape_text}; samples={count}; mode={mode}"


def bpp_value(row: dict[str, Any]) -> float | None:
    for key in ("scientific_bpp_with_side_info", "scientific_bpp", "bpp"):
        value = row.get(key)
        if finite_number(value) and float(value) > 0:
            return float(value)
    return None


def throughput_mbps(row: dict[str, Any], direction: str) -> float | None:
    direct = row.get(f"{direction}_throughput_MBps")
    if finite_number(direct):
        return float(direct)
    legacy = row.get(f"{direction}_throughput")
    if finite_number(legacy):
        return float(legacy) / 1e6
    return None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row.get("dataset_id")), model_family(row))].append(row)

    family_rows = []
    for (dataset_id, family), rows in sorted(grouped.items()):
        bpps = [value for row in rows if (value := bpp_value(row)) is not None]
        psnrs = [float(row["psnr"]) for row in rows if finite_number(row.get("psnr"))]
        axis_deltas = [
            float(row["psnr"]) - float(row["average_variable_psnr"])
            for row in rows
            if finite_number(row.get("psnr")) and finite_number(row.get("average_variable_psnr"))
        ]
        signatures = sorted({shape_signature(row) for row in rows})
        family_rows.append(
            {
                "dataset_id": dataset_id,
                "family": family,
                "record_count": len(rows),
                "bpp_min": min(bpps) if bpps else None,
                "bpp_max": max(bpps) if bpps else None,
                "bpp_decades": math.log10(max(bpps) / min(bpps)) if bpps and min(bpps) > 0 else None,
                "psnr_min": min(psnrs) if psnrs else None,
                "psnr_max": max(psnrs) if psnrs else None,
                "global_minus_axis0_psnr_median": median(axis_deltas),
                "global_minus_axis0_psnr_max": max(axis_deltas) if axis_deltas else None,
                "encode_throughput_MBps_median": median(throughput_mbps(row, "encode") for row in rows),
                "decode_throughput_MBps_median": median(throughput_mbps(row, "decode") for row in rows),
                "wall_time_coverage": sum(finite_number(row.get("sample_wall_time_total")) for row in rows) / len(rows),
                "input_signatures": signatures,
            }
        )

    comparability = []
    for dataset_id in sorted({str(row.get("dataset_id")) for row in records}):
        rows = [row for row in family_rows if row["dataset_id"] == dataset_id]
        signatures = {row["family"]: row["input_signatures"] for row in rows}
        canonical = {family: tuple(values) for family, values in signatures.items() if family != "AI+CAESAR-PCA"}
        comparability.append(
            {
                "dataset_id": dataset_id,
                "same_input_signature": len(set(canonical.values())) <= 1,
                "families": canonical,
                "warning": (
                    None
                    if len(set(canonical.values())) <= 1
                    else "不同模型族使用了不同的形状、样本数量或封装模式。"
                ),
            }
        )

    timing_coverage = defaultdict(lambda: {"rows": 0, "wall_rows": 0, "codec_rows": 0})
    for row in records:
        family = model_family(row)
        timing_coverage[family]["rows"] += 1
        timing_coverage[family]["wall_rows"] += int(finite_number(row.get("sample_wall_time_total")))
        timing_coverage[family]["codec_rows"] += int(
            finite_number(row.get("encode_time_avg")) or finite_number(row.get("decode_time_avg"))
        )

    model_dataset_rows = []
    by_model_dataset: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_model_dataset[(str(row.get("dataset_id")), str(row.get("model_name")))].append(row)
    for (dataset_id, model_name), rows in sorted(by_model_dataset.items()):
        model_dataset_rows.append(
            {
                "dataset_id": dataset_id,
                "model_name": model_name,
                "family": model_family(rows[0]),
                "record_count": len(rows),
                "encode_throughput_MBps_median": median(throughput_mbps(row, "encode") for row in rows),
                "decode_throughput_MBps_median": median(throughput_mbps(row, "decode") for row in rows),
                "wall_throughput_MBps_median": median(row.get("sample_wall_throughput_MBps") for row in rows),
                "original_bytes_median": median(row.get("original_bytes") for row in rows),
                "group_count_median": median(row.get("group_count", row.get("groups")) for row in rows),
            }
        )

    return {
        "source_records": len(records),
        "family_dataset_summary": family_rows,
        "input_comparability": comparability,
        "model_dataset_throughput": model_dataset_rows,
        "timing_coverage": dict(timing_coverage),
        "methodology_notes": [
            "BPP 优先采用包含辅助信息的 scientific_bpp_with_side_info。",
            "输入签名根据记录的形状、样本数和模式进行保守审计。",
            "签名不同可以证明测试输入不等价；签名相同仍不能证明数值完全一致。",
            "全局 PSNR 与 axis-0 平均 PSNR 的差值用于量化异构平面共享全局 range 造成的指标抬高。",
            "吞吐量中位数保留各 codec 封装现有的计时边界，并非严格统一的跨 codec 端到端计时。",
        ],
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    scalar_keys = [key for key in rows[0] if key != "input_signatures"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*scalar_keys, "input_signatures"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "input_signatures": " | ".join(row["input_signatures"])})


def fmt(value: Any, digits: int = 3) -> str:
    if not finite_number(value):
        return "n/a"
    return f"{float(value):.{digits}g}"


def write_markdown(analysis: dict[str, Any], path: Path) -> None:
    family_labels = {
        "AI image/video": "AI 图像/视频模型",
        "AI+CAESAR-PCA": "AI 模型+CAESAR-PCA",
        "CAESAR": "CAESAR",
        "traditional/scientific codec": "传统/科学数据压缩器",
        "other": "其他",
    }
    lines = [
        "# 压缩测试观察审计",
        "",
        "下表优先使用包含辅助信息的科学数据 BPP。",
        "不同模型族之间不预设输入相同。",
        "",
        "| 数据集 | 模型族 | BPP 范围 | PSNR 范围 | 全局-axis0 PSNR 差值中位数 | 编码 MB/s | 解码 MB/s |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["family_dataset_summary"]:
        lines.append(
            "| {dataset_id} | {family} | {b0}-{b1} | {p0}-{p1} | {delta} | {enc} | {dec} |".format(
                dataset_id=row["dataset_id"],
                family=family_labels.get(row["family"], row["family"]),
                b0=fmt(row["bpp_min"]),
                b1=fmt(row["bpp_max"]),
                p0=fmt(row["psnr_min"]),
                p1=fmt(row["psnr_max"]),
                delta=fmt(row["global_minus_axis0_psnr_median"]),
                enc=fmt(row["encode_throughput_MBps_median"]),
                dec=fmt(row["decode_throughput_MBps_median"]),
            )
        )
    lines.extend(["", "## 输入可比性", ""])
    for item in analysis["input_comparability"]:
        status = "通过" if item["same_input_signature"] else "不通过"
        lines.append(f"- `{item['dataset_id']}`: {status}")
        if not item["same_input_signature"]:
            for family, signatures in item["families"].items():
                lines.append(f"  - {family_labels.get(family, family)}: {' | '.join(signatures)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    records = payload["records"] if isinstance(payload, dict) else payload
    analysis = summarize(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    write_csv(analysis["family_dataset_summary"], args.output_dir / "family_dataset_summary.csv")
    throughput_rows = analysis["model_dataset_throughput"]
    with (args.output_dir / "model_dataset_throughput.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(throughput_rows[0]))
        writer.writeheader()
        writer.writerows(throughput_rows)
    write_markdown(analysis, args.output_dir / "audit.md")
    print(args.output_dir / "audit.md")


if __name__ == "__main__":
    main()
