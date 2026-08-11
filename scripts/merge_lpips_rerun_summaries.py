#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
RERUN_ROOT = ROOT / "unified_results_archive_20260611_035153/lpips_rerun"

DATASETS = {
    "e3sm_npz": {
        "summary": ROOT / "unified_results/e3sm_npz_all_models_with_external_n64/summary.json",
        "sources": {
            "external_cuszhi_eb0001": "external_cuszhi",
            "external_cuszhi_eb0005": "external_cuszhi",
            "external_cuszhi_eb001": "external_cuszhi",
            "external_cuszhi_eb005": "external_cuszhi",
            "external_cuszhi_eb01": "external_cuszhi",
            "external_cuszhi_eb05": "external_cuszhi",
            "external_cuszhi_eb1": "external_cuszhi",
            "external_graphcomp": "external_graphcomp",
            "external_visemz": "external_visemz",
        },
    },
    "era5_npy": {
        "summary": ROOT / "unified_results/era5_npy_all_models_c3_t16_240/summary.json",
        "sources": {
            "image_core": "image_core",
            "image_dcvc_rt": "image_dcvc_rt",
            "caesar_final7": "caesar_original",
            "external_cuszhi": "external_cuszhi_final7",
            "external_graphcomp": "external_graphcomp",
            "external_visemz": "external_visemz",
        },
    },
    "hurricane": {
        "summary": ROOT / "unified_results/hurricane_all_models_n16/summary.json",
        "sources": {
            "image_core": "image_core",
            "image_dcvc_rt": "image_dcvc_rt",
            "caesar": "caesar_original",
            "external_cuszhi": "external_cuszhi_final7",
            "external_graphcomp": "external_graphcomp",
            "external_visemz": "external_visemz",
        },
    },
    "kodak": {
        "summary": ROOT / "unified_results/kodak_all_models/summary.json",
        "sources": {
            "image_core": "image_core",
            "image_dcvc_rt": "image_dcvc_rt",
            "caesar": "caesar_original",
            "external_cuszhi": "external_cuszhi_final7",
            "external_graphcomp": "external_graphcomp_coarse",
            "external_visemz": "external_visemz",
        },
    },
    "lysozyme": {
        "summary": ROOT / "unified_results/lysozyme_all_models_n16/summary.json",
        "sources": {
            "image_core": "image_core",
            "image_dcvc_rt": "image_dcvc_rt",
            "caesar": "caesar_original",
            "external_cuszhi": "external_cuszhi_final7",
            "external_graphcomp": "external_graphcomp",
            "external_visemz": "external_visemz",
        },
    },
    "nyx": {
        "summary": ROOT / "unified_results/nyx_all_models_n16/summary.json",
        "sources": {
            "image_core": "image_core",
            "image_dcvc_rt": "image_dcvc_rt",
            "caesar_final7": "caesar_original",
            "external_cuszhi": "external_cuszhi_final7",
            "external_graphcomp": "external_graphcomp",
            "external_visemz": "external_visemz",
        },
    },
    "s2c": {
        "summary": ROOT / "unified_results/s2c_all_models_n16/summary.json",
        "sources": {
            "image_core": "image_core",
            "image_dcvc_rt": "image_dcvc_rt",
            "caesar": "caesar_original",
            "external_cuszhi": "external_cuszhi_final7",
            "external_graphcomp": "external_graphcomp_coarse",
            "external_visemz": "external_visemz",
        },
    },
    "tomo": {
        "summary": ROOT / "unified_results/tomo_all_models_n16/summary.json",
        "sources": {
            "image_core": "image_core",
            "image_dcvc_rt": "image_dcvc_rt",
            "caesar": "caesar_original",
            "external_cuszhi": "external_cuszhi_final7",
            "external_graphcomp": "external_graphcomp_coarse",
            "external_visemz": "external_visemz",
        },
    },
    "turb_rot_npz": {
        "summary": ROOT / "unified_results/turb_rot_npz_all_models_with_external_n64/summary.json",
        "sources": {
            "external_cuszhi_eb001": "external_cuszhi",
            "external_cuszhi_eb002_n16": "external_cuszhi",
            "external_cuszhi_eb005_n16": "external_cuszhi",
            "external_cuszhi_eb01_n16": "external_cuszhi",
            "external_cuszhi_eb02_n16": "external_cuszhi",
            "external_cuszhi_eb05_n16": "external_cuszhi",
            "external_cuszhi_eb1_n16": "external_cuszhi",
            "external_graphcomp": "external_graphcomp",
            "external_visemz": "external_visemz",
        },
    },
}

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

PRESERVE_IF_MISSING = ["input_shape", "input_numel", "input_bytes", "input_megapixels"]
LPIPS_ONLY_DATASETS = {"e3sm_npz", "turb_rot_npz"}
LPIPS_ONLY_FIELDS = {"lpips", *PRESERVE_IF_MISSING}
DATASET_DEFAULT_INPUT_SHAPES = {
    "e3sm_npz": [3, 240, 240],
    "turb_rot_npz": [3, 256, 256],
}


def load_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("records", "results", "summary"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported JSON summary format: {path}")


def group_key(row: dict) -> tuple[str, str, str, float | None]:
    return (
        str(row.get("source", "")),
        str(row.get("model_name", "")),
        str(row.get("model_id", row.get("checkpoint", ""))),
        row.get("eb"),
    )


def relaxed_group_key(row: dict) -> tuple[str, str, float | None]:
    return (
        str(row.get("source", "")),
        str(row.get("model_name", "")),
        row.get("eb"),
    )


def label_for(model_name: str, model_id: str, eb: float | None) -> str:
    if model_name == "visemz":
        return "visemz"
    if eb is None:
        return model_id or model_name
    return f"{model_name} eb={eb:g}"


def summarize(records: list[dict]) -> dict[tuple[str, str, str, float | None], dict]:
    groups: dict[tuple[str, str, str, float | None], list[dict]] = {}
    for record in records:
        if record.get("error"):
            continue
        groups.setdefault(group_key(record), []).append(record)

    summary = {}
    for key, group in groups.items():
        source, model_name, model_id, eb = key
        row = {
            "source": source,
            "model_name": model_name,
            "model_id": model_id,
            "label": label_for(model_name, model_id, eb),
            "sample_count": len(group),
            "success_count": len(group),
            "error_count": 0,
            "eb": eb,
        }
        for field in NUMERIC_FIELDS:
            values = [r[field] for r in group if isinstance(r.get(field), (int, float))]
            if values:
                row[field] = mean(values)
        for field in PRESERVE_IF_MISSING:
            values = [r.get(field) for r in group if r.get(field) is not None]
            if values:
                row[field] = values[0]
        summary[key] = row
    return summary


def rerun_records(dataset_id: str, source_map: dict[str, str]) -> list[dict]:
    records = []
    for rerun_dir, final_source in source_map.items():
        path = RERUN_ROOT / dataset_id / rerun_dir / "summary.json"
        if not path.exists():
            print(f"missing rerun summary: {path.relative_to(ROOT)}")
            continue
        for record in load_json(path):
            if final_source == "external_cuszhi_final7" and record.get("model_name") != "cuSZ-Hi":
                continue
            if final_source.startswith("external_graphcomp") and record.get("model_name") != "GraphComp":
                continue
            if final_source == "external_visemz" and record.get("model_name") != "visemz":
                continue
            item = dict(record)
            item["source"] = final_source
            records.append(item)
    return records


def fill_input_size(row: dict, dataset_id: str | None = None) -> None:
    if row.get("input_shape") is None and dataset_id in DATASET_DEFAULT_INPUT_SHAPES:
        row["input_shape"] = DATASET_DEFAULT_INPUT_SHAPES[dataset_id]
    if row.get("input_bytes") is None and isinstance(row.get("original_bytes"), (int, float)):
        row["input_bytes"] = row["original_bytes"]
    if row.get("input_numel") is None and isinstance(row.get("input_shape"), list):
        numel = 1
        for dim in row["input_shape"]:
            if not isinstance(dim, (int, float)):
                numel = None
                break
            numel *= int(dim)
        if numel is not None:
            row["input_numel"] = numel
    if row.get("input_bytes") is None and isinstance(row.get("input_numel"), (int, float)):
        row["input_bytes"] = int(row["input_numel"]) * 4
    if row.get("input_numel") is None and isinstance(row.get("input_bytes"), (int, float)):
        row["input_numel"] = int(round(row["input_bytes"] / 4))
    if row.get("input_megapixels") is None and isinstance(row.get("input_numel"), (int, float)):
        row["input_megapixels"] = float(row["input_numel"]) / 1e6


def main() -> None:
    for dataset_id, config in DATASETS.items():
        summary_path = config["summary"]
        current = load_json(summary_path)
        updates = summarize(rerun_records(dataset_id, config["sources"]))
        relaxed_updates = {relaxed_group_key(row): row for row in updates.values()}
        merged = []
        matched = 0
        for row in current:
            key = group_key(row)
            update = updates.get(key) or relaxed_updates.get(relaxed_group_key(row))
            if update is None:
                merged.append(row)
                continue
            merged_row = dict(row)
            for k, v in update.items():
                if k in {"source", "model_name", "model_id", "label", "eb", "sample_count", "success_count", "error_count"}:
                    continue
                if dataset_id in LPIPS_ONLY_DATASETS and k not in LPIPS_ONLY_FIELDS:
                    continue
                if k in PRESERVE_IF_MISSING and v is None:
                    continue
                merged_row[k] = v
            for field in PRESERVE_IF_MISSING:
                if merged_row.get(field) is None:
                    merged_row[field] = row.get(field)
            fill_input_size(merged_row, dataset_id)
            merged.append(merged_row)
            matched += 1

        for row in merged:
            fill_input_size(row, dataset_id)

        summary_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        with_lpips = sum(1 for row in merged if isinstance(row.get("lpips"), (int, float)))
        with_memory = sum(1 for row in merged if isinstance(row.get("memory_usage_MB"), (int, float)))
        with_input = sum(1 for row in merged if row.get("input_bytes") is not None or row.get("input_megapixels") is not None)
        print(
            f"{dataset_id}: rows={len(merged)} matched={matched} "
            f"lpips={with_lpips} memory={with_memory} input_size={with_input}"
        )


if __name__ == "__main__":
    main()
