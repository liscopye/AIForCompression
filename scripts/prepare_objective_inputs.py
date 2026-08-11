#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compression_pipeline.objective_data import (
    checksum,
    derive_dataset_normalization,
    load_objective_samples,
    save_normalization,
)
from scripts.run_matched_codec_validation import DEFAULT_DATASETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze objective-v1 dataset inputs and normalization manifests.")
    parser.add_argument("--dataset", choices=DEFAULT_DATASETS, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("unified_results/objective_v1"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_root / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    samples = load_objective_samples(args.dataset)
    normalization = derive_dataset_normalization(args.dataset, samples)
    save_normalization(output / "normalization.json", normalization)
    manifests = []
    for sample in samples:
        normalized = normalization.normalize(sample.raw)
        external_input = normalization.to_json()
        if sample.mask is not None:
            external_input.update({
                "validity_mask_policy": "shared_benchmark_metadata",
                "validity_mask_sha256": checksum(sample.mask.astype("uint8")),
                "validity_mask_rate_bytes": 0,
            })
        manifests.append({
            "protocol_id": "aifc-objective-v1",
            "dataset_id": args.dataset,
            "canonical_sample_id": sample.sample_id,
            "canonical_shape": list(sample.raw.shape),
            "canonical_dtype": str(sample.raw.dtype),
            "canonical_symbol_count": int(sample.raw.size),
            "canonical_valid_symbol_count": int(sample.mask.sum()) if sample.mask is not None else int(sample.raw.size),
            "canonical_sha256": checksum(sample.raw, sample.mask),
            "normalized_canonical_sha256": checksum(normalized, sample.mask),
            "external_input_manifest": external_input,
            **sample.metadata,
        })
        print(f"{args.dataset} {sample.sample_id} {sample.raw.shape}", flush=True)
    (output / "samples.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print(output / "samples.json")


if __name__ == "__main__":
    main()
