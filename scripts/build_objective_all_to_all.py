#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_objective_benchmark import expected_curves, model_family, row_gates


CORPUS_STACK_DATASETS = {"s2c", "kodak", "uvg_twilight_1080p"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the strict objective all-to-all result tree.")
    parser.add_argument("--baseline", type=Path, default=Path("unified_results/objective_v1"))
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument(
        "--schedule-overlays",
        type=Path,
        nargs="*",
        default=[],
        help="Dataset/curve schedules merged over --schedule.",
    )
    parser.add_argument("--output", type=Path, default=Path("unified_results/objective_all_to_all_v1"))
    return parser.parse_args()


def curve_name(row: dict[str, Any]) -> str:
    family = model_family(row)
    if family == "LIC-HPCM":
        return "LIC-HPCM-large" if "large" in str(row.get("model_id", "")).lower() else "LIC-HPCM-base"
    return family


def key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("dataset_id")),
        str(row.get("canonical_sample_id")),
        str(row.get("model_id")),
    )


def read_dataset_rows(root: Path, dataset: str) -> list[dict[str, Any]]:
    path = root / dataset / "summary.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def selected_control(row: dict[str, Any], schedule: dict[str, Any]) -> bool:
    curve = curve_name(row)
    if curve not in {"CAESAR-V", "CAESAR-D", "CAESAR-V-Turb-tuned", "CAESAR-D-Turb-tuned", "cuSZ-Hi"}:
        return True
    controls = schedule.get(curve, {}).get("controls", [])
    value = row.get("eb", row.get("control"))
    return isinstance(value, (int, float)) and any(
        abs(float(value) - float(control)) <= max(1e-12, abs(float(control)) * 1e-8)
        for control in controls
    )


def normalize_exact_lossless_metrics(row: dict[str, Any]) -> dict[str, Any]:
    mse = row.get("mse")
    psnr = row.get("psnr")
    if not isinstance(mse, (int, float)) or float(mse) > 1e-30:
        return row
    if isinstance(psnr, (int, float)) and math.isfinite(float(psnr)):
        return row
    output = dict(row)
    output["psnr"] = 300.0
    repetitions = output.get("timing_repetitions")
    if isinstance(repetitions, list):
        output["timing_repetitions"] = [
            {**item, "psnr": 300.0}
            if isinstance(item, dict)
            and isinstance(item.get("psnr"), (int, float))
            and not math.isfinite(float(item["psnr"]))
            else item
            for item in repetitions
        ]
    return output


def main() -> None:
    args = parse_args()
    protocol_path = ROOT / "benchmark_protocols/objective_v1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
    for overlay_path in args.schedule_overlays:
        overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
        for dataset, curves in overlay.items():
            schedule.setdefault(dataset, {}).update(curves)
    args.output.mkdir(parents=True, exist_ok=True)
    combined = []
    for dataset in protocol["datasets"]:
        merged: dict[tuple[str, str, str], dict[str, Any]] = {}
        for source in [args.baseline, *args.sources]:
            for row in read_dataset_rows(source, dataset):
                merged[key(row)] = row
        required = expected_curves(protocol, dataset)
        rows = []
        for row in merged.values():
            row = normalize_exact_lossless_metrics(row)
            if dataset == "lysozyme" and isinstance(row.get("external_input_manifest"), dict):
                row = dict(row)
                external = dict(row["external_input_manifest"])
                external.update({
                    "validity_mask_policy": "shared_benchmark_metadata",
                    "validity_mask_rate_bytes": 0,
                })
                row["external_input_manifest"] = external
            curve = curve_name(row)
            if dataset in CORPUS_STACK_DATASETS and curve in {
                "CAESAR-V", "CAESAR-D", "CAESAR-V-Turb-tuned", "CAESAR-D-Turb-tuned", "cuSZ-Hi"
            }:
                covered = row.get("covered_canonical_sample_ids")
                if not isinstance(covered, list) or set(map(str, covered)) != set(
                    map(str, protocol["datasets"][dataset]["objective_samples"])
                ):
                    continue
            if curve not in required or not selected_control(row, schedule.get(dataset, {})):
                continue
            gates = row_gates(row, protocol)
            if not all(gates.values()):
                continue
            rows.append(row)
        rows.sort(key=lambda row: (curve_name(row), str(row.get("model_id")), str(row.get("canonical_sample_id"))))
        target = args.output / dataset
        target.mkdir(parents=True, exist_ok=True)
        for filename in ("samples.json", "normalization.json"):
            shutil.copy2(args.baseline / dataset / filename, target / filename)
        (target / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        combined.extend(rows)
        print(f"{dataset}: {len(rows)} rows")
    (args.output / "combined_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    shutil.copy2(protocol_path, args.output / "protocol.json")
    (args.output / "eb_schedule.json").write_text(json.dumps(schedule, indent=2), encoding="utf-8")
    print(args.output / "combined_summary.json")


if __name__ == "__main__":
    main()
