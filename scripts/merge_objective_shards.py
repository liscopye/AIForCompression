#!/usr/bin/env python3
"""Merge independently written objective-v1 result shards by canonical point key."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument("sources", type=Path, nargs="+")
    return parser.parse_args()


def point_key(row: dict) -> tuple:
    return (
        row.get("dataset_id"),
        row.get("canonical_sample_id"),
        row.get("model_id"),
        str(row.get("control")),
        row.get("track_id"),
    )


def main() -> None:
    args = parse_args()
    rows = json.loads(args.target.read_text(encoding="utf-8")) if args.target.exists() else []
    merged = {point_key(row): row for row in rows}
    for source in args.sources:
        source_rows = json.loads(source.read_text(encoding="utf-8"))
        for row in source_rows:
            key = point_key(row)
            previous = merged.get(key)
            if previous is not None:
                identity = ("canonical_sha256", "normalized_canonical_sha256", "canonical_shape")
                if any(previous.get(field) != row.get(field) for field in identity):
                    raise ValueError(f"Canonical identity mismatch for {key} from {source}")
            merged[key] = row
    output_rows = sorted(
        merged.values(),
        key=lambda row: (
            str(row.get("dataset_id")), str(row.get("model_name")), str(row.get("model_id")),
            str(row.get("canonical_sample_id")), str(row.get("control")),
        ),
    )
    args.target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.target.parent, delete=False, encoding="utf-8") as handle:
        json.dump(output_rows, handle, indent=2)
        temporary = Path(handle.name)
    os.replace(temporary, args.target)
    print(f"{args.target}: {len(output_rows)} rows")


if __name__ == "__main__":
    main()
