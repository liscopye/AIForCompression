#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh objective-v1 fixed-unit-range metric fields.")
    parser.add_argument("root", type=Path, nargs="?", default=Path("unified_results/objective_v1"))
    return parser.parse_args()


def update_summary(path: Path) -> int:
    rows = json.loads(path.read_text(encoding="utf-8"))
    changed = 0
    for row in rows:
        mse = row.get("normalized_mse", row.get("mse"))
        if not isinstance(mse, (int, float)) or not math.isfinite(float(mse)) or float(mse) < 0:
            continue
        expected = -10.0 * math.log10(max(float(mse), 1e-30))
        if row.get("normalized_mse") != mse or row.get("normalized_psnr") != expected or row.get("fixed_scale_data_range") != 1.0:
            changed += 1
        row["normalized_mse"] = float(mse)
        row["normalized_psnr"] = expected
        row["fixed_scale_data_range"] = 1.0
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
        temporary = Path(handle.name)
    os.replace(temporary, path)
    return changed


def main() -> None:
    args = parse_args()
    paths = sorted(args.root.glob("*/summary.json")) if args.root.is_dir() else [args.root]
    total = 0
    for path in paths:
        changed = update_summary(path)
        total += changed
        print(f"{path}: {changed} rows refreshed")
    print(f"total: {total}")


if __name__ == "__main__":
    main()
