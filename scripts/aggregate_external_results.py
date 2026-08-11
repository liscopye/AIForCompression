#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]

NUMERIC_FIELDS = [
    "mse",
    "rmse",
    "psnr",
    "average_variable_psnr",
    "average_frame_psnr",
    "lpips",
    "compression_ratio",
    "bpp",
    "image_bpp",
    "scientific_bpp",
    "scientific_bpp_with_side_info",
    "bitstream_bytes",
    "side_info_bytes",
    "total_bytes_with_side_info",
    "original_bytes",
    "encode_time_total",
    "decode_time_total",
    "encode_time_avg",
    "decode_time_avg",
    "encode_throughput_MBps",
    "decode_throughput_MBps",
    "encode_throughput",
    "decode_throughput",
    "memory_usage_MB",
    "memory_reserved_MB",
    "eb",
    "graphcomp_scale",
    "graphcomp_sigma",
    "graphcomp_min_size",
    "graphcomp_region_count",
    "graphcomp_max_error",
    "graphcomp_error_bound_abs",
    "graphcomp_residual_bitstream_bytes",
    "graphcomp_side_bitstream_bytes",
]

SUMMARY_FIELDS = [
    "label",
    "source",
    "model_name",
    "model_id",
    "sample_count",
    "success_count",
    "error_count",
    "eb",
    *NUMERIC_FIELDS,
]


DATASETS = {
    "e3sm_npz_all_models_with_external_n64": {
        "old_summary": ROOT
        / "unified_results/e3sm_npz_all_models_vars3_frame_psnr_n64/plots/e3sm_vars3_frame_n64_summary.json",
        "old_records": [
            ROOT / "unified_results/e3sm_npz_all_models_vars3_frame_psnr_n64/image_models/summary.json",
            ROOT / "unified_results/e3sm_npz_all_models_vars3_frame_psnr_n64/caesar_original/summary.json",
        ],
        "external": ROOT / "unified_results/e3sm_npz_external_models_n64",
    },
    "turb_rot_npz_all_models_with_external_n64": {
        "old_summary": ROOT
        / "unified_results/turb_rot_npz_all_models_sections3_frame_psnr_n64/plots/turb_rot_sections3_frame_n64_summary.json",
        "old_records": [
            ROOT / "unified_results/turb_rot_npz_all_models_sections3_frame_psnr_n64/image_models/summary.json",
            ROOT / "unified_results/turb_rot_npz_all_models_sections3_frame_psnr_n64/caesar_original/summary.json",
            ROOT / "unified_results/turb_rot_npz_all_models_sections3_frame_psnr_n64/caesar_tuned/summary.json",
        ],
        "external": ROOT / "unified_results/turb_rot_npz_external_models_n64",
    },
    "era5_npy_external_models_c3_t16_240": {
        "old_summary": None,
        "old_records": [],
        "external": ROOT / "unified_results/era5_npy_external_models_c3_t16_240",
    },
}


def main() -> None:
    for output_name, config in DATASETS.items():
        output_dir = ROOT / "unified_results" / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        old_summary_raw = load_json(config["old_summary"]) if config["old_summary"] else []
        old_summary = [r for r in old_summary_raw if r.get("model_name") != "CAESAR"]
        old_records = []
        caesar_summary = []
        for path in config["old_records"]:
            records_for_path = load_records(path)
            old_records.extend(records_for_path)
            if "caesar" in path.parent.name:
                for record in records_for_path:
                    item = dict(record)
                    item.setdefault("source", path.parent.name)
                    item.setdefault("model_name", "CAESAR")
                    item.setdefault("label", label_for(item.get("model_name"), item.get("model_id", ""), item.get("eb")))
                    caesar_summary.append(item)
        external_records = []
        for path in sorted(Path(config["external"]).glob("*/summary.json")):
            source = path.parent.name
            for record in load_records(path):
                record = dict(record)
                record.setdefault("source", source)
                external_records.append(record)

        summary = list(old_summary)
        summary.extend(caesar_summary)
        summary.extend(summarize_external(external_records))
        records = old_records + external_records

        write_json(output_dir / "summary.json", summary)
        write_json(output_dir / "records.json", records)
        write_csv(output_dir / "summary.csv", summary)
        print(f"{output_dir}: summary={len(summary)} records={len(records)}")


def load_json(path: Path | None):
    if path is None or not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_records(path: Path) -> list[dict]:
    data = load_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        return data["records"]
    return []


def summarize_external(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, float | None], list[dict]] = {}
    for record in records:
        key = (record.get("source", "external"), record.get("model_id", record.get("model_name", "unknown")), record.get("eb"))
        groups.setdefault(key, []).append(record)

    rows = []
    for (source, model_id, eb), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], str(item[0][2]))):
        successes = [r for r in group if "error" not in r]
        errors = [r for r in group if "error" in r]
        model_name = first_value(group, "model_name") or model_id
        row = {
            "source": f"external_{source}",
            "model_name": model_name,
            "model_id": model_id,
            "label": label_for(model_name, model_id, eb),
            "sample_count": len(group),
            "success_count": len(successes),
            "error_count": len(errors),
            "eb": eb,
        }
        for field in NUMERIC_FIELDS:
            values = [r[field] for r in successes if isinstance(r.get(field), (int, float))]
            if values:
                row[field] = mean(values)
        rows.append(row)
    return rows


def first_value(records: list[dict], key: str):
    for record in records:
        value = record.get(key)
        if value is not None:
            return value
    return None


def label_for(model_name: str, model_id: str, eb: float | None) -> str:
    if model_name == "visemz":
        return "visemz"
    if eb is None:
        return model_id
    return f"{model_name} eb={eb:g}"


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    extra = sorted({key for row in rows for key in row.keys()} - set(SUMMARY_FIELDS))
    fields = SUMMARY_FIELDS + extra
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


if __name__ == "__main__":
    main()
