#!/usr/bin/env python3
"""Merge deterministic LPIPS-only reruns into an existing objective result tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source", type=Path, nargs="*", default=[])
    parser.add_argument("--memory-source", type=Path)
    parser.add_argument("--memory-only", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row.get("model_id")), str(row.get("canonical_sample_id")), str(row.get("control"))


def load(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a result list: {path}")
    return payload


def compatible(target: dict[str, Any], source: dict[str, Any]) -> bool:
    return (
        target.get("canonical_sha256") == source.get("canonical_sha256")
        and target.get("normalized_canonical_sha256") == source.get("normalized_canonical_sha256")
    )


def main() -> None:
    args = parse_args()
    datasets = sorted(path.parent.name for path in args.target.glob("*/summary.json"))
    total_updated = 0
    total_missing = 0
    combined: list[dict[str, Any]] = []
    for dataset in datasets:
        target_path = args.target / dataset / "summary.json"
        targets = load(target_path)
        sources = []
        for source_root in args.source:
            source_path = source_root / dataset / "summary.json"
            if source_path.exists():
                sources.extend(load(source_path))
        memory_path = args.memory_source / dataset / "summary.json" if args.memory_source else None
        memory_rows = load(memory_path) if memory_path is not None and memory_path.exists() else []
        memory_by_model: dict[str, float] = {}
        for memory_row in memory_rows:
            value = memory_row.get("memory_usage_MB")
            if isinstance(value, (int, float)) and float(value) > 0:
                model_name = str(memory_row.get("model_name"))
                memory_by_model[model_name] = max(memory_by_model.get(model_name, 0.0), float(value))
        source_by_key = {key(row): row for row in sources if isinstance(row.get("lpips"), (int, float))}
        updated = 0
        missing = 0
        for row in targets:
            probed_memory = memory_by_model.get(str(row.get("model_name")))
            if probed_memory is not None:
                row["memory_usage_MB"] = probed_memory
                row["memory_measurement"] = "single_control_external_gpu_peak_probe"
            if args.memory_only:
                continue
            source = source_by_key.get(key(row))
            if source is None or not compatible(row, source):
                if isinstance(row.get("lpips"), (int, float)):
                    continue
                missing += 1
                continue
            row["lpips"] = float(source["lpips"])
            row["lpips_view"] = "native_rgb" if dataset in {"kodak", "uvg_twilight_1080p"} else "frozen_normalized_grayscale"
            row["lpips_measurement"] = "single_decode_rerun"
            for field in ("scientific_bpp_with_side_info", "normalized_psnr"):
                if isinstance(row.get(field), (int, float)) and isinstance(source.get(field), (int, float)):
                    row[f"lpips_rerun_{field}_delta"] = float(source[field]) - float(row[field])
            updated += 1
        target_path.write_text(json.dumps(targets, indent=2), encoding="utf-8")
        combined.extend(targets)
        total_updated += updated
        total_missing += missing
        print(f"{dataset}: updated={updated} missing={missing}")
    if args.require_complete and total_missing:
        raise SystemExit(f"Refusing incomplete merge: {total_missing} rows do not have matching LPIPS")
    (args.target / "combined_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"total: updated={total_updated} missing={total_missing}")


if __name__ == "__main__":
    main()
