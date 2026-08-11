#!/usr/bin/env python3
from __future__ import annotations

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
]

DATASETS = {
    "e3sm_npz": {
        "base": ROOT / "unified_results/merged/legacy_with_external/e3sm_npz_all_models_with_external_n64/summary.json",
        "no_pca": ROOT / "unified_results/raw/caesar_no_pca/e3sm_npz_caesar_no_pca/summary.json",
        "cuszhi_3d": ROOT / "unified_results/e3sm_npz_external_models_n64/cuszhi_3d_packz_extreme7_n1/summary.json",
        "output": ROOT / "unified_results/final/no_pca_cuszhi3d_packz_full_n1/e3sm_npz_all_models_with_no_pca_and_cuszhi3d_packz_full_n1",
        "title": "E3SM",
        "prefix": "e3sm_no_pca",
        "zoom_bpp": 2.0,
    },
    "turb_rot_npz": {
        "base": ROOT / "unified_results/merged/legacy_with_external/turb_rot_npz_all_models_with_external_n64/summary.json",
        "no_pca": ROOT / "unified_results/raw/caesar_no_pca/turb_rot_npz_caesar_no_pca/summary.json",
        "cuszhi_3d": ROOT / "unified_results/turb_rot_npz_external_models_n64/cuszhi_3d_packz_extreme7_n1/summary.json",
        "output": ROOT / "unified_results/final/no_pca_cuszhi3d_packz_full_n1/turb_rot_npz_all_models_with_no_pca_and_cuszhi3d_packz_full_n1",
        "title": "Turb-Rot",
        "prefix": "turb_rot_no_pca",
        "zoom_bpp": 2.0,
    },
    "era5_npy": {
        "base": ROOT / "unified_results/raw/all_models_base/era5_npy_all_models_c3_t16_240/summary.json",
        "no_pca": ROOT / "unified_results/raw/caesar_no_pca/era5_npy_caesar_no_pca/summary.json",
        "cuszhi_3d": ROOT / "unified_results/era5_npy_external_models_c3_t16_240/cuszhi_3d_packz_extreme7_n1/summary.json",
        "output": ROOT / "unified_results/final/no_pca_cuszhi3d_packz_full_n1/era5_npy_all_models_c3_t16_240_with_no_pca_and_cuszhi3d_packz_full_n1",
        "title": "ERA5",
        "prefix": "era5_npy_no_pca",
        "zoom_bpp": 2.0,
    },
    "kodak": {
        "base": ROOT / "unified_results/raw/all_models_base/kodak_all_models/summary.json",
        "no_pca": ROOT / "unified_results/raw/caesar_no_pca/kodak_caesar_no_pca/summary.json",
        "cuszhi_3d": ROOT / "unified_results/kodak_external_models/cuszhi_3d_packz_extreme7_n1/summary.json",
        "output": ROOT / "unified_results/final/no_pca_cuszhi3d_packz_full_n1/kodak_all_models_with_no_pca_and_cuszhi3d_packz_full_n1",
        "title": "Kodak",
        "prefix": "kodak_no_pca",
        "zoom_bpp": 2.0,
    },
    "s2c": {
        "base": ROOT / "unified_results/raw/all_models_base/s2c_all_models_n16/summary.json",
        "no_pca": ROOT / "unified_results/raw/caesar_no_pca/s2c_caesar_no_pca/summary.json",
        "cuszhi_3d": ROOT / "unified_results/s2c_external_models_n16/cuszhi_3d_packz_extreme7_n1/summary.json",
        "output": ROOT / "unified_results/final/no_pca_cuszhi3d_packz_full_n1/s2c_all_models_n16_with_no_pca_and_cuszhi3d_packz_full_n1",
        "title": "S2C",
        "prefix": "s2c_no_pca",
        "zoom_bpp": 2.5,
    },
    "tomo": {
        "base": ROOT / "unified_results/raw/all_models_base/tomo_all_models_n16/summary.json",
        "no_pca": ROOT / "unified_results/raw/caesar_no_pca/tomo_caesar_no_pca/summary.json",
        "cuszhi_3d": ROOT / "unified_results/tomo_external_models_n16/cuszhi_3d_packz_extreme7_n1/summary.json",
        "output": ROOT / "unified_results/final/no_pca_cuszhi3d_packz_full_n1/tomo_all_models_n16_with_no_pca_and_cuszhi3d_packz_full_n1",
        "title": "Tomo",
        "prefix": "tomo_no_pca",
        "zoom_bpp": 1.2,
    },
    "hurricane": {
        "base": ROOT / "unified_results/raw/all_models_base/hurricane_all_models_n16/summary.json",
        "no_pca": ROOT / "unified_results/raw/caesar_no_pca/hurricane_caesar_no_pca/summary.json",
        "cuszhi_3d": ROOT / "unified_results/hurricane_external_models_n16/cuszhi_3d_packz_extreme7_n1/summary.json",
        "output": ROOT / "unified_results/final/no_pca_cuszhi3d_packz_full_n1/hurricane_all_models_n16_with_no_pca_and_cuszhi3d_packz_full_n1",
        "title": "Hurricane",
        "prefix": "hurricane_no_pca",
        "zoom_bpp": 2.0,
    },
    "nyx": {
        "base": ROOT / "unified_results/raw/all_models_base/nyx_all_models_n16/summary.json",
        "no_pca": ROOT / "unified_results/raw/caesar_no_pca/nyx_caesar_no_pca/summary.json",
        "cuszhi_3d": ROOT / "unified_results/nyx_external_models_n16/cuszhi_3d_packz_extreme7_n1/summary.json",
        "output": ROOT / "unified_results/final/no_pca_cuszhi3d_packz_full_n1/nyx_all_models_n16_with_no_pca_and_cuszhi3d_packz_full_n1",
        "title": "NYX",
        "prefix": "nyx_no_pca",
        "zoom_bpp": 1.0,
    },
    "lysozyme": {
        "base": ROOT / "unified_results/raw/all_models_base/lysozyme_all_models_n16/summary.json",
        "no_pca": ROOT / "unified_results/raw/caesar_no_pca/lysozyme_caesar_no_pca/summary.json",
        "cuszhi_3d": ROOT / "unified_results/lysozyme_external_models_n16/cuszhi_3d_packz_extreme7_n1/summary.json",
        "output": ROOT / "unified_results/final/no_pca_cuszhi3d_packz_full_n1/lysozyme_all_models_n16_with_no_pca_and_cuszhi3d_packz_full_n1",
        "title": "Lysozyme",
        "prefix": "lysozyme_no_pca",
        "zoom_bpp": 1.0,
    },
}


def main() -> None:
    index = {}
    for dataset_id, cfg in DATASETS.items():
        base_rows = [row for row in load_rows(cfg["base"]) if row.get("model_name") != "cuSZ-Hi"]
        no_pca_rows = load_rows(cfg["no_pca"])
        cuszhi_3d_rows = summarize_cuszhi_3d(load_rows(cfg["cuszhi_3d"]))
        merged = list(base_rows)
        for row in no_pca_rows:
            item = dict(row)
            item["source"] = "caesar_no_pca"
            item["label"] = label_for_no_pca(item)
            item["metric"] = item.get("metric", "mse")
            item.setdefault("sample_count", 1)
            item.setdefault("success_count", 1)
            item.setdefault("error_count", 0)
            merged.append(item)
        merged.extend(cuszhi_3d_rows)

        output = cfg["output"]
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "summary.json", merged)
        index[dataset_id] = {
            "summary": str((output / "summary.json").relative_to(ROOT)),
            "base_count": len(base_rows),
            "no_pca_count": len(no_pca_rows),
            "cuszhi_3d_count": len(cuszhi_3d_rows),
            "total_count": len(merged),
            "title": cfg["title"],
            "prefix": cfg["prefix"],
            "zoom_bpp": cfg["zoom_bpp"],
        }
        print(
            f"{dataset_id}: base={len(base_rows)} no_pca={len(no_pca_rows)} "
            f"cuszhi_3d={len(cuszhi_3d_rows)} total={len(merged)}"
        )

    write_json(ROOT / "unified_results/metadata/no_pca_cuszhi3d_merge_index.json", index)


def load_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("summary", "records", "results"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported summary format: {path}")


def label_for_no_pca(row: dict) -> str:
    model_id = str(row.get("model_id", ""))
    family = "CAESAR-D" if model_id.startswith("caesar_d") else "CAESAR-V"
    return f"{family} no PCA"


def summarize_cuszhi_3d(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, float | None], list[dict]] = {}
    for record in records:
        if record.get("model_name") != "cuSZ-Hi":
            continue
        model_id = str(record.get("model_id", ""))
        if not model_id.startswith("cuSZ-Hi-3D"):
            continue
        key = ("cuSZ-Hi", model_id, record.get("eb"))
        groups.setdefault(key, []).append(record)

    rows = []
    for (model_name, model_id, eb), group in sorted(groups.items(), key=lambda item: (str(item[0][2]), item[0][1])):
        successes = [r for r in group if "error" not in r]
        errors = [r for r in group if "error" in r]
        if not successes:
            continue
        row = {
            "source": "external_cuszhi_3d_packz_extreme7_n1",
            "model_name": model_name,
            "model_id": model_id,
            "label": f"cuSZ-Hi-3D eb={eb:g}" if isinstance(eb, (int, float)) else "cuSZ-Hi-3D",
            "sample_count": len(group),
            "success_count": len(successes),
            "error_count": len(errors),
            "eb": eb,
            "cuszhi_whole3d": True,
        }
        for field in NUMERIC_FIELDS:
            values = [r[field] for r in successes if isinstance(r.get(field), (int, float))]
            if values:
                row[field] = mean(values)
        rows.append(row)
    return rows


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
