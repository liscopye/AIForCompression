#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "e3sm_npz": ROOT / "unified_results/e3sm_npz_external_models_n64/cuszhi_3d_quick_probe/summary.json",
    "turb_rot_npz": ROOT / "unified_results/turb_rot_npz_external_models_n64/cuszhi_3d_quick_probe/summary.json",
    "era5_npy": ROOT / "unified_results/era5_npy_external_models_c3_t16_240/cuszhi_3d_quick_probe/summary.json",
    "kodak": ROOT / "unified_results/kodak_external_models/cuszhi_3d_quick_probe/summary.json",
    "s2c": ROOT / "unified_results/s2c_external_models_n16/cuszhi_3d_quick_probe/summary.json",
    "tomo": ROOT / "unified_results/tomo_external_models_n16/cuszhi_3d_quick_probe/summary.json",
    "hurricane": ROOT / "unified_results/hurricane_external_models_n16/cuszhi_3d_quick_probe/summary.json",
    "nyx": ROOT / "unified_results/nyx_external_models_n16/cuszhi_3d_quick_probe/summary.json",
    "lysozyme": ROOT / "unified_results/lysozyme_external_models_n16/cuszhi_3d_quick_probe/summary.json",
}


def main() -> None:
    rows = []
    for dataset_id, path in DATASETS.items():
        if not path.exists():
            continue
        records = json.loads(path.read_text(encoding="utf-8"))
        grouped: dict[float | None, list[dict]] = defaultdict(list)
        for record in records:
            grouped[record.get("eb")].append(record)
        for eb, group in sorted(grouped.items(), key=lambda item: float(item[0]) if item[0] is not None else -1):
            successes = [r for r in group if "error" not in r]
            errors = [r for r in group if "error" in r]
            row = {
                "dataset_id": dataset_id,
                "model_name": "cuSZ-Hi",
                "model_id": f"cuSZ-Hi-3D_eb{format_eb(eb)}" if eb is not None else "cuSZ-Hi-3D",
                "label": f"cuSZ-Hi-3D eb={eb:g}" if isinstance(eb, (int, float)) else "cuSZ-Hi-3D",
                "source": "external_cuszhi_3d_quick_probe",
                "eb": eb,
                "sample_count": len(group),
                "success_count": len(successes),
                "error_count": len(errors),
                "bpp": avg(successes, "bpp"),
                "psnr": avg_any(successes, ("average_frame_psnr", "average_variable_psnr", "psnr")),
                "encode_throughput_MBps": avg(successes, "encode_throughput_MBps"),
                "decode_throughput_MBps": avg(successes, "decode_throughput_MBps"),
                "cuszhi_whole3d": True,
            }
            rows.append(row)

    output_dir = ROOT / "unified_results/cuszhi_3d_quick_probe"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", rows)
    write_csv(output_dir / "summary.csv", rows)
    for row in rows:
        bpp = row["bpp"]
        psnr = row["psnr"]
        print(
            f"{row['dataset_id']:14s} eb={row['eb']:<8g} "
            f"ok={row['success_count']:2d} err={row['error_count']:2d} "
            f"bpp={bpp:.4g} psnr={psnr:.2f}"
        )
    print(f"wrote {output_dir / 'summary.json'}")


def avg(records: list[dict], key: str) -> float | None:
    values = [r[key] for r in records if isinstance(r.get(key), (int, float))]
    return mean(values) if values else None


def avg_any(records: list[dict], keys: tuple[str, ...]) -> float | None:
    values = []
    for record in records:
        for key in keys:
            value = record.get(key)
            if isinstance(value, (int, float)):
                values.append(value)
                break
    return mean(values) if values else None


def format_eb(value: float | None) -> str:
    if value is None:
        return "none"
    return f"{value:g}".replace("-", "m").replace(".", "p")


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "dataset_id",
        "source",
        "model_name",
        "model_id",
        "label",
        "eb",
        "sample_count",
        "success_count",
        "error_count",
        "bpp",
        "psnr",
        "encode_throughput_MBps",
        "decode_throughput_MBps",
        "cuszhi_whole3d",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


if __name__ == "__main__":
    main()
