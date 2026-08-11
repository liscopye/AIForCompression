#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "benchmark_protocols/objective_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a result set against the objective benchmark protocol.")
    parser.add_argument("result", type=Path, help="combined_summary.json or a directory containing per-dataset summary.json files")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, help="Output JSON path; defaults beside the input")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless every eligible row and dataset passes all gates")
    return parser.parse_args()


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            return payload["records"]
        raise ValueError(f"Unsupported result JSON schema: {path}")

    rows: list[dict[str, Any]] = []
    for summary_path in sorted(path.glob("*/summary.json")):
        dataset_rows = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(dataset_rows, list):
            continue
        manifest_path = summary_path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        for source in dataset_rows:
            row = dict(manifest)
            row.update(source)
            rows.append(row)
    return rows


def model_family(row: dict[str, Any]) -> str:
    model_id = str(row.get("model_id", "")).lower()
    name = str(row.get("model_name") or row.get("model_view") or "").lower()
    text = f"{name} {model_id}"
    if "caesar_pca" in text or "+caesar-pca" in text:
        return "PCA-hybrid"
    if "no_pca" in text:
        return "CAESAR-no-PCA"
    if "caesar_v_turb_tuned" in text:
        return "CAESAR-V-Turb-tuned"
    if "caesar_d_turb_tuned" in text:
        return "CAESAR-D-Turb-tuned"
    if "caesar_v" in text:
        return "CAESAR-V"
    if "caesar_d" in text:
        return "CAESAR-D"
    if "dcmvc_pframe" in text:
        return "DCMVC-IP"
    if "dcvc_rt_pframe" in text:
        return "DCVC-RT-IP"
    if "dcmvc" in text:
        return "DCMVC-I"
    if "dcvc" in text:
        return "DCVC-RT-I"
    if "lic-hpcm" in text or "hpcm" in text:
        return "LIC-HPCM"
    if "dcae" in text:
        return "DCAE"
    if "cusz" in text:
        return "cuSZ-Hi"
    if "nvjpeg2000" in text or "nvjpeg2k" in text:
        return "nvJPEG2000"
    if "nvjpeg" in text:
        return "nvJPEG"
    return str(row.get("model_name") or row.get("model_view") or "unknown")


def expected_families(protocol: dict[str, Any], dataset_id: str) -> set[str]:
    expected: set[str] = set()
    for track_id, track in protocol["tracks"].items():
        if track_id == "ablation" or dataset_id not in track.get("datasets", []):
            continue
        expected.update(track.get("main_methods", []))
    expected.difference_update(protocol.get("datasets", {}).get(dataset_id, {}).get("excluded_main_methods", []))
    return expected


def optional_families(protocol: dict[str, Any], dataset_id: str) -> set[str]:
    return set(protocol.get("datasets", {}).get(dataset_id, {}).get("optional_methods", []))


def curve_name(row: dict[str, Any]) -> str:
    family = model_family(row)
    if family == "LIC-HPCM":
        return "LIC-HPCM-large" if "large" in str(row.get("model_id", "")).lower() else "LIC-HPCM-base"
    return family


def expected_curves(protocol: dict[str, Any], dataset_id: str) -> dict[str, int]:
    requirements = protocol.get("curve", {}).get("required_points_by_curve", {})
    curves: dict[str, int] = {}
    for family in expected_families(protocol, dataset_id) | optional_families(protocol, dataset_id):
        names = ["LIC-HPCM-base", "LIC-HPCM-large"] if family == "LIC-HPCM" else [family]
        for name in names:
            curves[name] = int(requirements.get(name, protocol["curve"]["minimum_valid_points"]))
    return curves


def row_gates(row: dict[str, Any], protocol: dict[str, Any]) -> dict[str, bool]:
    dataset_id = str(row.get("dataset_id", ""))
    family = model_family(row)
    symbol_count = row.get("canonical_symbol_count", row.get("voxel_count", row.get("input_numel")))
    payload = row.get("payload_bytes", row.get("bitstream_bytes"))
    side = row.get("side_info_bytes")
    total = row.get("total_bytes_with_side_info")
    bpp = row.get("scientific_bpp_with_side_info")

    rate_complete = all(finite(value) for value in (symbol_count, payload, side, total, bpp)) and float(symbol_count) > 0
    if rate_complete:
        expected_total = float(payload) + float(side)
        expected_bpp = 8.0 * float(total) / float(symbol_count)
        rate_complete = math.isclose(float(total), expected_total, rel_tol=1e-6, abs_tol=1e-6) and math.isclose(
            float(bpp), expected_bpp, rel_tol=1e-5, abs_tol=1e-8
        )

    repetitions = row.get("timing_repetitions")
    timing_complete = (
        row.get("timing_protocol") == protocol["protocol_id"]
        and isinstance(repetitions, list)
        and len(repetitions) >= int(protocol["timing"]["measured_repetitions"])
        and all(isinstance(item, dict) and finite(item.get("roundtrip_seconds")) for item in repetitions)
    )
    codec_valid = "error" not in row and finite(row.get("psnr"))
    if family == "cuSZ-Hi":
        codec_valid = codec_valid and row.get("error_bound_satisfied") is True
    expected_caesar_batch = protocol["timing"].get("codec_execution", {}).get("caesar_inference_batch_size")
    codec_execution_declared = (
        family not in {"CAESAR-V", "CAESAR-D", "CAESAR-V-Turb-tuned", "CAESAR-D-Turb-tuned"}
        or expected_caesar_batch is None
        or row.get("caesar_inference_batch_size") == expected_caesar_batch
    )
    normalized_mse = row.get("normalized_mse")
    normalized_psnr = row.get("normalized_psnr")
    metric_fixed_scale = (
        row.get("metric_protocol") == protocol["protocol_id"]
        and finite(normalized_mse)
        and float(normalized_mse) >= 0.0
        and finite(normalized_psnr)
        and math.isclose(
            float(normalized_psnr),
            -10.0 * math.log10(max(float(normalized_mse), 1e-30)),
            rel_tol=1e-9,
            abs_tol=1e-8,
        )
        and row.get("fixed_scale_data_range") == 1.0
    )

    return {
        "known_dataset": dataset_id in protocol["datasets"],
        "track_eligible": family in expected_families(protocol, dataset_id)
        or family in optional_families(protocol, dataset_id)
        or family in {"CAESAR-no-PCA", "PCA-hybrid"},
        "canonical_identity": all(
            row.get(field) not in (None, "")
            for field in ("canonical_sample_id", "canonical_sha256", "canonical_shape", "canonical_valid_symbol_count")
        ),
        "rate_complete": rate_complete,
        "metric_fixed_scale": metric_fixed_scale,
        "external_input_declared": (
            isinstance(row.get("external_input_manifest"), dict)
            and row.get("external_input_manifest", {}).get("scope") == "dataset"
            and row.get("external_input_manifest", {}).get("normalization_id") not in (None, "")
            and row.get("normalized_canonical_sha256") not in (None, "")
        ),
        "mask_policy_declared": (
            dataset_id != "lysozyme"
            or row.get("external_input_manifest", {}).get("validity_mask_policy")
            == "shared_benchmark_metadata"
        ),
        "timing_repeated": timing_complete,
        "hardware_declared": isinstance(row.get("hardware_manifest"), dict),
        "codec_execution_declared": codec_execution_declared,
        "codec_valid": codec_valid,
        "protocol_tagged": row.get("protocol_id") == protocol["protocol_id"],
    }


def audit(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    gate_names = [
        "known_dataset", "track_eligible", "canonical_identity", "rate_complete", "metric_fixed_scale",
        "external_input_declared", "mask_policy_declared", "timing_repeated", "hardware_declared", "codec_valid", "protocol_tagged",
        "codec_execution_declared",
    ]
    row_reports = []
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        gates = row_gates(row, protocol)
        report = {
            "dataset_id": row.get("dataset_id"),
            "model_id": row.get("model_id"),
            "family": model_family(row),
            "sample_id": row.get("canonical_sample_id", row.get("sample_id")),
            "gates": gates,
            "compliant": all(gates.values()),
        }
        row_reports.append(report)
        by_dataset[str(report["dataset_id"])].append(report)
        by_family[report["family"]].append(report)

    dataset_reports = []
    for dataset_id, contract in protocol["datasets"].items():
        reports = by_dataset.get(dataset_id, [])
        source_rows = [row for row in rows if row.get("dataset_id") == dataset_id]
        sample_ids: set[str] = set()
        for row in source_rows:
            covered = row.get("covered_canonical_sample_ids")
            if isinstance(covered, list):
                sample_ids.update(str(sample_id) for sample_id in covered)
            elif row.get("canonical_sample_id", row.get("sample_id")):
                sample_ids.add(str(row.get("canonical_sample_id", row.get("sample_id"))))
        expected_samples = set(contract.get("objective_samples", []))
        exact_samples = expected_samples.issubset(sample_ids)
        hashes_by_sample: dict[str, set[str]] = defaultdict(set)
        normalized_hashes_by_sample: dict[str, set[str]] = defaultdict(set)
        for row in source_rows:
            covered_hashes = row.get("covered_canonical_sha256")
            covered_normalized_hashes = row.get("covered_normalized_canonical_sha256")
            if isinstance(covered_hashes, dict):
                for sample_id, value in covered_hashes.items():
                    hashes_by_sample[str(sample_id)].add(str(value))
                if isinstance(covered_normalized_hashes, dict):
                    for sample_id, value in covered_normalized_hashes.items():
                        normalized_hashes_by_sample[str(sample_id)].add(str(value))
                continue
            sample_id = row.get("canonical_sample_id", row.get("sample_id", row.get("selection", "__single__")))
            checksum = row.get("canonical_sha256")
            if sample_id and checksum:
                hashes_by_sample[str(sample_id)].add(str(checksum))
            normalized_checksum = row.get("normalized_canonical_sha256")
            if sample_id and normalized_checksum:
                normalized_hashes_by_sample[str(sample_id)].add(str(normalized_checksum))
        checksum_consistent = bool(hashes_by_sample) and all(len(values) == 1 for values in hashes_by_sample.values())
        normalized_checksum_consistent = bool(normalized_hashes_by_sample) and all(
            len(values) == 1 for values in normalized_hashes_by_sample.values()
        )
        present = {item["family"] for item in reports}
        expected = expected_families(protocol, dataset_id)
        compliant_ids = {
            (str(report["model_id"]), str(report["sample_id"]))
            for report in reports if report["compliant"]
        }
        complete_points: dict[str, set[str]] = defaultdict(set)
        grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in source_rows:
            grouped_rows[(curve_name(row), str(row.get("model_id")))].append(row)
        for (curve, model_id), point_rows in grouped_rows.items():
            covered: set[str] = set()
            all_compliant = True
            for row in point_rows:
                row_covered = row.get("covered_canonical_sample_ids")
                if isinstance(row_covered, list):
                    covered.update(str(value) for value in row_covered)
                else:
                    sample_id = str(row.get("canonical_sample_id", row.get("sample_id", "")))
                    if sample_id:
                        covered.add(sample_id)
                report_key = (model_id, str(row.get("canonical_sample_id", row.get("sample_id"))))
                all_compliant = all_compliant and report_key in compliant_ids
            if all_compliant and expected_samples.issubset(covered):
                complete_points[curve].add(model_id)
        curve_requirements = expected_curves(protocol, dataset_id)
        curve_counts = {curve: len(complete_points.get(curve, set())) for curve in curve_requirements}
        insufficient_curves = {
            curve: {"required": required, "present": curve_counts[curve]}
            for curve, required in curve_requirements.items() if curve_counts[curve] < required
        }
        dataset_reports.append({
            "dataset_id": dataset_id,
            "rows": len(reports),
            "compliant_rows": sum(item["compliant"] for item in reports),
            "expected_samples": sorted(expected_samples),
            "present_samples": sorted(sample_ids),
            "objective_sample_coverage": exact_samples,
            "checksum_consistent_by_sample": checksum_consistent,
            "normalized_checksum_consistent_by_sample": normalized_checksum_consistent,
            "expected_main_families": sorted(expected),
            "present_families": sorted(present),
            "missing_main_families": sorted(expected - present),
            "complete_curve_points": curve_counts,
            "insufficient_curves": insufficient_curves,
            "complete": bool(reports) and exact_samples and checksum_consistent and normalized_checksum_consistent
            and expected.issubset(present) and not insufficient_curves
            and all(item["compliant"] for item in reports if item["family"] in expected),
        })

    gate_counts = {gate: sum(report["gates"][gate] for report in row_reports) for gate in gate_names}
    family_reports = []
    for family, reports in sorted(by_family.items()):
        family_reports.append({
            "family": family,
            "rows": len(reports),
            "compliant_rows": sum(item["compliant"] for item in reports),
            "gate_passes": {gate: sum(item["gates"][gate] for item in reports) for gate in gate_names},
        })
    return {
        "protocol_id": protocol["protocol_id"],
        "rows": len(rows),
        "compliant_rows": sum(report["compliant"] for report in row_reports),
        "gate_passes": gate_counts,
        "datasets_complete": sum(item["complete"] for item in dataset_reports),
        "datasets": dataset_reports,
        "families": family_reports,
        "row_failure_patterns": Counter(
            ",".join(name for name, passed in report["gates"].items() if not passed) for report in row_reports
        ).most_common(),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Objective benchmark 合规审计：{report['protocol_id']}", "",
        f"- 记录：{report['rows']}",
        f"- 完全合规记录：{report['compliant_rows']}",
        f"- 完整数据集：{report['datasets_complete']}/{len(report['datasets'])}", "",
        "## 全局检查", "",
        "| 检查项 | 通过记录 | 总记录 |", "|---|---:|---:|",
    ]
    for gate, count in report["gate_passes"].items():
        lines.append(f"| {gate} | {count} | {report['rows']} |")
    lines.extend(["", "## 数据集覆盖", "", "| 数据集 | 记录 | objective samples | raw checksum | normalized checksum | 缺少主方法 | 点数不足 | 完整 |", "|---|---:|---:|---:|---:|---|---|---:|"])
    for dataset in report["datasets"]:
        missing = ", ".join(dataset["missing_main_families"]) or "-"
        insufficient = ", ".join(
            f"{curve} {value['present']}/{value['required']}"
            for curve, value in dataset["insufficient_curves"].items()
        ) or "-"
        lines.append(
            f"| {dataset['dataset_id']} | {dataset['rows']} | {'是' if dataset['objective_sample_coverage'] else '否'} | "
            f"{'是' if dataset['checksum_consistent_by_sample'] else '否'} | "
            f"{'是' if dataset['normalized_checksum_consistent_by_sample'] else '否'} | "
            f"{missing} | {insufficient} | {'是' if dataset['complete'] else '否'} |"
        )
    lines.extend(["", "## 说明", "", "旧结果未通过新协议不等于 codec 输出无效，而是不能支撑严格的跨 codec 客观排名。"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    rows = load_rows(args.result)
    report = audit(rows, protocol)
    if args.output:
        output = args.output
    elif args.result.is_dir():
        output = args.result / "objective_protocol_audit.json"
    else:
        output = args.result.with_name(f"{args.result.stem}_objective_protocol_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = output.with_suffix(".md")
    write_markdown(report, markdown)
    print(output)
    print(markdown)
    if args.strict and (report["compliant_rows"] != report["rows"] or report["datasets_complete"] != len(report["datasets"])):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
