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
    "params",
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
    "era5_npy": {
        "output": ROOT / "unified_results/era5_npy_all_models_c3_t16_240",
        "inputs": [
            ("image_core", ROOT / "unified_results/era5_npy_all_models_c3_t16_240/image_core/summary.json"),
            ("image_dcvc_rt", ROOT / "unified_results/era5_npy_all_models_c3_t16_240/image_dcvc_rt/summary.json"),
            ("caesar_original", ROOT / "unified_results/era5_npy_all_models_c3_t16_240/caesar_final7/summary.json"),
            ("external_cuszhi_final7", ROOT / "unified_results/era5_npy_external_models_c3_t16_240/cuszhi/summary.json"),
            ("external_graphcomp", ROOT / "unified_results/era5_npy_external_models_c3_t16_240/graphcomp/summary.json"),
            ("external_visemz", ROOT / "unified_results/era5_npy_external_models_c3_t16_240/visemz/summary.json"),
        ],
    },
    "kodak": {
        "output": ROOT / "unified_results/kodak_all_models",
        "inputs": [
            ("image_core", ROOT / "unified_results/kodak_all_models/image_core/summary.json"),
            ("image_dcvc_rt", ROOT / "unified_results/kodak_all_models/image_dcvc_rt/summary.json"),
            ("caesar_original", ROOT / "unified_results/kodak_all_models/caesar/summary.json"),
            ("external_cuszhi_final7", ROOT / "unified_results/kodak_external_models/external_final7/summary.json"),
            ("external_graphcomp_coarse", ROOT / "unified_results/kodak_external_models/graphcomp_coarse/summary.json"),
            ("external_visemz", ROOT / "unified_results/kodak_external_models/visemz/summary.json"),
        ],
    },
    "s2c": {
        "output": ROOT / "unified_results/s2c_all_models_n16",
        "inputs": [
            ("image_core", ROOT / "unified_results/s2c_all_models_n16/image_core/summary.json"),
            ("image_dcvc_rt", ROOT / "unified_results/s2c_all_models_n16/image_dcvc_rt/summary.json"),
            ("caesar_original", ROOT / "unified_results/s2c_all_models_n16/caesar/summary.json"),
            ("external_cuszhi_final7", ROOT / "unified_results/s2c_external_models_n16/external_final7/summary.json"),
            ("external_graphcomp_coarse", ROOT / "unified_results/s2c_external_models_n16/graphcomp_coarse/summary.json"),
            ("external_visemz", ROOT / "unified_results/s2c_external_models_n16/visemz/summary.json"),
        ],
    },
    "tomo": {
        "output": ROOT / "unified_results/tomo_all_models_n16",
        "inputs": [
            ("image_core", ROOT / "unified_results/tomo_all_models_n16/image_core/summary.json"),
            ("image_dcvc_rt", ROOT / "unified_results/tomo_all_models_n16/image_dcvc_rt/summary.json"),
            ("caesar_original", ROOT / "unified_results/tomo_all_models_n16/caesar/summary.json"),
            ("external_cuszhi_final7", ROOT / "unified_results/tomo_external_models_n16/cuszhi_final7/summary.json"),
            ("external_graphcomp_coarse", ROOT / "unified_results/tomo_external_models_n16/graphcomp_coarse/summary.json"),
            ("external_visemz", ROOT / "unified_results/tomo_external_models_n16/visemz/summary.json"),
        ],
    },
    "hurricane": {
        "output": ROOT / "unified_results/hurricane_all_models_n16",
        "inputs": [
            ("image_core", ROOT / "unified_results/hurricane_all_models_n16/image_core/summary.json"),
            ("image_dcvc_rt", ROOT / "unified_results/hurricane_all_models_n16/image_dcvc_rt/summary.json"),
            ("caesar_original", ROOT / "unified_results/hurricane_all_models_n16/caesar/summary.json"),
            ("external_cuszhi_final7", ROOT / "unified_results/hurricane_external_models_n16/cuszhi_final7/summary.json"),
            ("external_graphcomp", ROOT / "unified_results/hurricane_external_models_n16/graphcomp/summary.json"),
            ("external_visemz", ROOT / "unified_results/hurricane_external_models_n16/visemz/summary.json"),
        ],
    },
    "nyx": {
        "output": ROOT / "unified_results/nyx_all_models_n16",
        "inputs": [
            ("image_core", ROOT / "unified_results/nyx_all_models_n16/image_core/summary.json"),
            ("image_dcvc_rt", ROOT / "unified_results/nyx_all_models_n16/image_dcvc_rt/summary.json"),
            ("caesar_original", ROOT / "unified_results/nyx_all_models_n16/caesar_final7/summary.json"),
            ("external_cuszhi_final7", ROOT / "unified_results/nyx_external_models_n16/cuszhi_final7/summary.json"),
            ("external_graphcomp", ROOT / "unified_results/nyx_external_models_n16/graphcomp/summary.json"),
            ("external_visemz", ROOT / "unified_results/nyx_external_models_n16/visemz/summary.json"),
        ],
    },
    "lysozyme": {
        "output": ROOT / "unified_results/lysozyme_all_models_n16",
        "inputs": [
            ("image_core", ROOT / "unified_results/lysozyme_all_models_n16/image_core/summary.json"),
            ("image_dcvc_rt", ROOT / "unified_results/lysozyme_all_models_n16/image_dcvc_rt/summary.json"),
            ("caesar_original", ROOT / "unified_results/lysozyme_all_models_n16/caesar/summary.json"),
            ("external_cuszhi_final7", ROOT / "unified_results/lysozyme_external_models_n16/cuszhi_final7/summary.json"),
            ("external_graphcomp", ROOT / "unified_results/lysozyme_external_models_n16/graphcomp/summary.json"),
            ("external_visemz", ROOT / "unified_results/lysozyme_external_models_n16/visemz/summary.json"),
        ],
    },
}


def main() -> None:
    schedule_rows = []
    for dataset_id, config in DATASETS.items():
        records = []
        missing = []
        for source, path in config["inputs"]:
            if not path.exists():
                missing.append(str(path.relative_to(ROOT)))
                continue
            for record in load_records(path):
                if source == "external_cuszhi_final7" and record.get("model_name") != "cuSZ-Hi":
                    continue
                if source.startswith("external_graphcomp") and record.get("model_name") != "GraphComp":
                    continue
                if source == "external_visemz" and record.get("model_name") != "visemz":
                    continue
                item = dict(record)
                item.setdefault("dataset_id", dataset_id)
                item["source"] = source
                records.append(item)

        summary = summarize(records)
        output_dir = config["output"]
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "summary.json", summary)
        write_json(output_dir / "records.json", records)
        write_csv(output_dir / "summary.csv", summary)
        for row in summary:
            eb = row.get("eb")
            if eb is not None:
                schedule_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "source": row.get("source"),
                        "model_name": row.get("model_name"),
                        "model_id": row.get("model_id"),
                        "eb": eb,
                        "bpp": row.get("bpp") or row.get("scientific_bpp_with_side_info") or row.get("scientific_bpp"),
                        "psnr": row.get("average_frame_psnr") or row.get("psnr"),
                    }
                )
        print(
            f"{dataset_id}: summary={len(summary)} records={len(records)} "
            f"missing={len(missing)} output={output_dir.relative_to(ROOT)}"
        )
        for item in missing:
            print(f"  missing {item}")
    write_json(ROOT / "unified_results/new_dataset_all_models_eb_schedules.json", schedule_rows)


def load_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        return data["records"]
    return []


def summarize(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str, float | None], list[dict]] = {}
    for record in records:
        key = (
            str(record.get("source", "")),
            str(record.get("model_name", "")),
            str(record.get("model_id", record.get("checkpoint", ""))),
            record.get("eb"),
        )
        groups.setdefault(key, []).append(record)

    rows = []
    for (source, model_name, model_id, eb), group in sorted(groups.items(), key=lambda item: item[0]):
        successes = [r for r in group if "error" not in r]
        errors = [r for r in group if "error" in r]
        if not successes:
            continue
        row = {
            "source": source,
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


def label_for(model_name: str, model_id: str, eb: float | None) -> str:
    if model_name == "visemz":
        return "visemz"
    if model_name == "cuSZ-Hi" and model_id.startswith("cuSZ-Hi-3D"):
        if eb is None:
            return "cuSZ-Hi-3D"
        return f"cuSZ-Hi-3D eb={eb:g}"
    if eb is None:
        return model_id or model_name
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
