#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compression_pipeline.objective_data import checksum, load_normalization, load_objective_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Export exact objective-v1 UVG RGB frames for native video codecs.")
    parser.add_argument("--root", type=Path, default=Path("unified_results/objective_v1"))
    args = parser.parse_args()
    dataset_dir = args.root / "uvg_twilight_1080p"
    manifests = json.loads((dataset_dir / "samples.json").read_text(encoding="utf-8"))
    sample = load_objective_samples("uvg_twilight_1080p")[0]
    normalization = load_normalization(dataset_dir / "normalization.json")
    normalized = normalization.normalize(sample.raw)
    expected = manifests[0]["normalized_canonical_sha256"]
    if checksum(normalized, sample.mask) != expected:
        raise ValueError("Normalized UVG checksum changed before frame export")

    frames_dir = dataset_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    restored = np.empty_like(normalized)
    paths = []
    for index in range(normalized.shape[1]):
        rgb = np.rint(normalized[:, index] * 255.0).astype(np.uint8)
        path = frames_dir / f"im{index + 1:05d}.png"
        Image.fromarray(np.moveaxis(rgb, 0, -1), mode="RGB").save(path, compress_level=1)
        decoded = np.moveaxis(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32), -1, 0) / 255.0
        restored[:, index] = decoded
        paths.append(str(path))
    restored_hash = checksum(restored, sample.mask)
    if restored_hash != expected:
        raise ValueError(f"PNG export is not canonical: {restored_hash} != {expected}")
    payload = {
        "protocol_id": "aifc-objective-v1",
        "dataset_id": "uvg_twilight_1080p",
        "canonical_sample_id": sample.sample_id,
        "normalized_canonical_sha256": expected,
        "frame_count": len(paths),
        "width": int(normalized.shape[-1]),
        "height": int(normalized.shape[-2]),
        "paths": paths,
    }
    (frames_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(frames_dir / "manifest.json")


if __name__ == "__main__":
    main()
