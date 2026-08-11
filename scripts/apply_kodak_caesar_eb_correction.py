#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "unified_results/kodak_all_models/summary.json"
SOURCE = {
    "caesar_v": ROOT / "unified_results_archive_20260611_035153/kodak_caesar_eb_probe_v/summary.json",
    "caesar_d": ROOT / "unified_results_archive_20260611_035153/kodak_caesar_eb_probe_d/summary.json",
}
EBS = [0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3]

RD_KEYS = {
    "eb",
    "mse",
    "rmse",
    "psnr",
    "average_variable_psnr",
    "average_frame_psnr",
    "compression_ratio",
    "bpp",
    "scientific_bpp",
    "scientific_bpp_with_side_info",
    "bitstream_bytes",
    "side_info_bytes",
    "total_bytes_with_side_info",
    "original_bytes",
    "params",
    "model_view",
    "shape",
    "timestamps",
    "start_index",
}


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def label_for(model_id: str, eb: float) -> str:
    family = "CAESAR-V" if model_id == "caesar_v" else "CAESAR-D"
    return f"{family} eb={eb:g}"


def main() -> None:
    final_rows = load(FINAL)
    replacements: dict[str, list[dict]] = {}
    schedule = {}

    for model_id, source_path in SOURCE.items():
        points = [
            r
            for r in load(source_path)
            if r.get("model_name") == "CAESAR"
            and r.get("model_id") == model_id
            and "error" not in r
            and float(r.get("eb")) in EBS
        ]
        points = sorted(points, key=lambda r: EBS.index(float(r["eb"])))
        if len(points) != len(EBS):
            raise RuntimeError(f"{model_id}: expected {len(EBS)} source points, got {len(points)}")

        templates = sorted(
            [r for r in final_rows if r.get("model_name") == "CAESAR" and r.get("model_id") == model_id],
            key=lambda r: float(r.get("eb", 0.0)),
        )
        if len(templates) != len(EBS):
            raise RuntimeError(f"{model_id}: expected {len(EBS)} old rows, got {len(templates)}")

        updated = []
        for template, point in zip(templates, points):
            eb = float(point["eb"])
            row = dict(template)
            for key in RD_KEYS:
                if key in point:
                    row[key] = point[key]
            row["source"] = "caesar_original_eb_corrected"
            row["label"] = label_for(model_id, eb)
            row["sample_id"] = point.get("sample_id", row.get("sample_id"))
            row["caesar_eb_correction"] = True
            row["caesar_eb_source_summary"] = str(source_path.relative_to(ROOT))
            row["throughput_lpips_memory_reused_from_previous_caesar"] = True
            updated.append(row)
        replacements[model_id] = updated
        schedule[model_id] = {
            "effective_eb": EBS,
            "bpp": [float(r["scientific_bpp_with_side_info"]) for r in points],
            "psnr": [float(r["average_frame_psnr"]) for r in points],
            "source_summary": str(source_path.relative_to(ROOT)),
            "note": "Kodak old CAESAR eb=1..100 points saturated and collapsed to one plotted point; replaced with lower EB sweep at 512x512 crop.",
        }

    cursors = {model_id: iter(rows) for model_id, rows in replacements.items()}
    out_rows = []
    for row in final_rows:
        if row.get("model_name") == "CAESAR" and row.get("model_id") in cursors:
            out_rows.append(next(cursors[row["model_id"]]))
        else:
            out_rows.append(row)
    write(FINAL, out_rows)

    schedule_path = ROOT / "unified_results/kodak_caesar_effective_eb_schedules.json"
    schedule_path.write_text(json.dumps(schedule, indent=2) + "\n", encoding="utf-8")
    print(f"updated {FINAL}")
    print(f"wrote {schedule_path}")


if __name__ == "__main__":
    main()
