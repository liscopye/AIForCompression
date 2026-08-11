#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "unified_results_archive_20260611_035153"

DATASETS = {
    "s2c": {
        "final": ROOT / "unified_results/s2c_all_models_n16/summary.json",
        "psnr_only": ARCHIVE / "cuszhi_psnr_only_s2c/summary.json",
        "note": "RGB data is 0..255; old eb=1..100 over-relaxed cuSZ-Hi and produced negative PSNR.",
    },
    "kodak": {
        "final": ROOT / "unified_results/kodak_all_models/summary.json",
        "psnr_only": ARCHIVE / "cuszhi_psnr_only_kodak/summary.json",
        "note": "RGB data is 0..255; old eb=1..100 over-relaxed cuSZ-Hi and produced negative PSNR.",
    },
    "lysozyme": {
        "final": ROOT / "unified_results/lysozyme_all_models_n16/summary.json",
        "psnr_only": ARCHIVE / "cuszhi_psnr_only_lysozyme/summary.json",
        "note": "Dropped old eb=1.25/1.5 negative-PSNR points; retained lightweight PSNR-only EB calibration.",
    },
    "tomo": {
        "final": ROOT / "unified_results/tomo_all_models_n16/summary.json",
        "psnr_only": ARCHIVE / "cuszhi_psnr_only_tomo/summary.json",
        "note": "Dropped old eb=0.4 negative-PSNR point; retained lightweight PSNR-only EB calibration.",
    },
}

RD_KEYS = {
    "eb",
    "mse",
    "rmse",
    "psnr",
    "average_variable_psnr",
    "average_frame_psnr",
    "compression_ratio",
    "bpp",
    "image_bpp",
    "scientific_bpp",
    "scientific_bpp_with_side_info",
    "bitstream_bytes",
    "side_info_bytes",
    "total_bytes_with_side_info",
    "original_bytes",
    "group_count",
    "cuszhi_channelwise",
    "cuszhi_sample_mode",
    "cuszhi_min_abs_eb",
}


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def format_eb(eb: float) -> str:
    return f"{eb:g}".replace(".", "p").replace("-", "m")


def rd_value(row: dict, key: str):
    return row.get(key)


def main() -> None:
    schedule: dict[str, dict] = {}
    for dataset, cfg in DATASETS.items():
        final_path = cfg["final"]
        psnr_path = cfg["psnr_only"]
        final_rows = load_rows(final_path)
        old_cuszhi = [r for r in final_rows if r.get("model_name") == "cuSZ-Hi"]
        new_points = sorted(
            [r for r in load_rows(psnr_path) if r.get("model_name") == "cuSZ-Hi" and "error" not in r],
            key=lambda r: float(r["eb"]),
        )
        if len(old_cuszhi) != 7:
            raise RuntimeError(f"{dataset}: expected 7 old cuSZ-Hi rows, got {len(old_cuszhi)}")
        if len(new_points) != 7:
            raise RuntimeError(f"{dataset}: expected 7 PSNR-only cuSZ-Hi rows, got {len(new_points)}")

        old_templates = sorted(old_cuszhi, key=lambda r: float(r.get("eb", 0.0)))
        replacements = []
        for template, point in zip(old_templates, new_points):
            eb = float(point["eb"])
            row = dict(template)
            for key in RD_KEYS:
                if key in point:
                    row[key] = rd_value(point, key)
            row["source"] = "external_cuszhi_psnr_only_corrected"
            row["model_id"] = f"cuSZ-Hi_eb{format_eb(eb)}"
            row["label"] = f"cuSZ-Hi eb={eb:g}"
            row["metric"] = "error_bound"
            row["eb"] = eb
            row["psnr_only_correction"] = True
            row["psnr_only_source_summary"] = str(psnr_path.relative_to(ROOT))
            row["psnr_only_sample_id"] = point.get("sample_id")
            row["psnr_only_sample_count"] = 1
            row["throughput_lpips_memory_reused_from_previous_cuszhi"] = True
            replacements.append(row)

        replacement_iter = iter(replacements)
        corrected = [next(replacement_iter) if r.get("model_name") == "cuSZ-Hi" else r for r in final_rows]
        dump_rows(final_path, corrected)

        schedule[dataset] = {
            "model_name": "cuSZ-Hi",
            "effective_eb": [float(r["eb"]) for r in new_points],
            "psnr": [float(r.get("average_frame_psnr", r.get("psnr"))) for r in new_points],
            "bpp": [float(r.get("scientific_bpp_with_side_info", r.get("bpp"))) for r in new_points],
            "psnr_only_source_summary": str(psnr_path.relative_to(ROOT)),
            "final_summary": str(final_path.relative_to(ROOT)),
            "psnr_only_sample_count": 1,
            "other_metrics_policy": "LPIPS, throughput, memory, and input-size fields reused from the previous cuSZ-Hi final-summary rows.",
            "note": cfg["note"],
        }

    out = ROOT / "unified_results/cuszhi_effective_eb_schedules.json"
    out.write_text(json.dumps(schedule, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
