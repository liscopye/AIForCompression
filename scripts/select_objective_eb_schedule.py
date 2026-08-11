#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select valid, log-BPP-spaced CAESAR/cuSZ controls.")
    parser.add_argument("summaries", type=Path, nargs="+")
    parser.add_argument("--points", type=int, default=7)
    parser.add_argument("--target-max-bpp", type=float, default=32.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def curve_name(row: dict[str, Any]) -> str | None:
    model_id = str(row.get("model_id", "")).lower()
    if model_id.startswith("caesar_v_turb_tuned"):
        return "CAESAR-V-Turb-tuned"
    if model_id.startswith("caesar_d_turb_tuned"):
        return "CAESAR-D-Turb-tuned"
    if model_id.startswith("caesar_v"):
        return "CAESAR-V"
    if model_id.startswith("caesar_d"):
        return "CAESAR-D"
    if str(row.get("model_name", "")).lower() == "cusz-hi":
        return "cuSZ-Hi"
    return None


def aggregate_candidates(
    rows: list[dict[str, Any]], expected_samples: set[str] | None = None
) -> list[dict[str, float]]:
    by_control: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "error" in row or curve_name(row) is None:
            continue
        if curve_name(row) == "cuSZ-Hi" and row.get("error_bound_satisfied") is not True:
            continue
        control = row.get("eb", row.get("control"))
        bpp = row.get("scientific_bpp_with_side_info")
        psnr = row.get("normalized_psnr", row.get("psnr"))
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in (control, bpp, psnr)):
            continue
        if float(control) <= 0 or float(bpp) <= 0:
            continue
        by_control[float(control)].append(row)
    candidates = []
    for control, group in by_control.items():
        covered: set[str] = set()
        for row in group:
            row_covered = row.get("covered_canonical_sample_ids")
            if isinstance(row_covered, list):
                covered.update(map(str, row_covered))
            elif row.get("canonical_sample_id"):
                covered.add(str(row["canonical_sample_id"]))
        if expected_samples and not expected_samples.issubset(covered):
            continue
        psnr_values = [
            float(row["normalized_psnr"] if row.get("normalized_psnr") is not None else row["psnr"])
            for row in group
        ]
        candidates.append({
            "control": control,
            "bpp": sum(float(row["scientific_bpp_with_side_info"]) for row in group) / len(group),
            "psnr": sum(psnr_values) / len(psnr_values),
        })
    return candidates


def pareto(candidates: list[dict[str, float]]) -> list[dict[str, float]]:
    kept = []
    for point in sorted(candidates, key=lambda item: (item["bpp"], -item["psnr"])):
        if any(
            other["bpp"] <= point["bpp"] and other["psnr"] >= point["psnr"]
            and (other["bpp"] < point["bpp"] or other["psnr"] > point["psnr"])
            for other in candidates
        ):
            continue
        if kept and point["bpp"] <= kept[-1]["bpp"] * (1 + 1e-8):
            continue
        kept.append(point)
    return kept


def select(points: list[dict[str, float]], count: int, target_max_bpp: float = 32.0) -> list[dict[str, float]]:
    if points and target_max_bpp > 0 and points[-1]["bpp"] > target_max_bpp:
        endpoint = min(points, key=lambda item: abs(math.log(item["bpp"]) - math.log(target_max_bpp)))
        points = points[: points.index(endpoint) + 1]
    if len(points) <= count:
        return points
    low = math.log(points[0]["bpp"])
    high = math.log(points[-1]["bpp"])
    targets = [math.exp(low + index * (high - low) / (count - 1)) for index in range(count)]
    selected = []
    available = list(points)
    for target in targets:
        choice = min(available, key=lambda item: abs(math.log(item["bpp"]) - math.log(target)))
        selected.append(choice)
        available.remove(choice)
    return sorted(selected, key=lambda item: item["bpp"])


def main() -> None:
    args = parse_args()
    protocol_path = Path(__file__).resolve().parents[1] / "benchmark_protocols" / "objective_v1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in args.summaries:
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            curve = curve_name(row)
            dataset = row.get("dataset_id")
            if curve and dataset:
                grouped[(str(dataset), curve)].append(row)
    output: dict[str, dict[str, Any]] = defaultdict(dict)
    for (dataset, curve), rows in sorted(grouped.items()):
        expected_samples = set(protocol.get("datasets", {}).get(dataset, {}).get("objective_samples", []))
        frontier = pareto(aggregate_candidates(rows, expected_samples))
        selected = select(frontier, args.points, args.target_max_bpp)
        output[dataset][curve] = {
            "controls": [item["control"] for item in selected],
            "bpp_range": [selected[0]["bpp"], selected[-1]["bpp"]] if selected else None,
            "valid_frontier": frontier,
            "selected": selected,
            "enough_points": len(selected) >= args.points,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
