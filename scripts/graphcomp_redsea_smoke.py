#!/usr/bin/env python3
"""Small RedSea smoke test for GraphComp-style prediction plus SZ residual.

The upstream GraphComp scripts are hard-coded for local paths. This runner keeps
the same core idea for a quick sanity check:

1. Read a small number of RedSea frames directly from the zip archive.
2. Segment the first frame with Felzenszwalb.
3. Predict every frame by replacing each segment with its per-frame mean.
4. Use GraphComp's SZ3 wrapper to compress the residual under relative EBs.

This is intentionally a smoke test, not a full fair accounting of GraphComp
metadata/latent side information.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
from skimage.segmentation import felzenszwalb
from skimage.util import img_as_float


REPO = Path(__file__).resolve().parents[1]
GRAPHCOMP_ROOT = REPO / "models" / "GraphComp"
sys.path.insert(0, str(GRAPHCOMP_ROOT))

from error_bounded import decompress, my_compress  # noqa: E402


def read_redsea_frames(zip_path: Path, member: str, n_frames: int, height: int, width: int) -> np.ndarray:
    n_values = n_frames * height * width
    n_bytes = n_values * np.dtype(np.float32).itemsize
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            raw = fh.read(n_bytes)
    if len(raw) != n_bytes:
        raise ValueError(f"Requested {n_bytes} bytes from {member}, got {len(raw)}")
    return np.frombuffer(raw, dtype=np.float32).copy().reshape(n_frames, height, width)


def segment_mean_predict(frames: np.ndarray, segments: np.ndarray) -> np.ndarray:
    labels = segments.reshape(-1).astype(np.int64)
    n_labels = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=n_labels).astype(np.float64)
    pred = np.empty_like(frames, dtype=np.float32)
    for i, frame in enumerate(frames):
        sums = np.bincount(labels, weights=frame.reshape(-1), minlength=n_labels)
        means = (sums / np.maximum(counts, 1.0)).astype(np.float32)
        pred[i] = means[labels].reshape(frame.shape)
    return pred


def psnr(data: np.ndarray, recon: np.ndarray) -> float:
    mse = float(np.mean((data.astype(np.float64) - recon.astype(np.float64)) ** 2))
    if mse == 0.0:
        return math.inf
    data_range = float(np.ptp(data))
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", type=Path, default=Path("/workspace/Redsea_t2_gan.zip"))
    parser.add_argument("--member", default="Redsea_t2_500_gan.dat")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--height", type=int, default=855)
    parser.add_argument("--width", type=int, default=1215)
    parser.add_argument("--scale", type=float, default=500.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--min_size", type=int, default=100)
    parser.add_argument("--ebs", default="1e-2,1e-3,1e-4")
    parser.add_argument("--output", type=Path, default=REPO / "unified_results" / "graphcomp_redsea_smoke.json")
    args = parser.parse_args()

    t0 = time.time()
    frames = read_redsea_frames(args.zip_path, args.member, args.frames, args.height, args.width)
    read_sec = time.time() - t0

    t1 = time.time()
    segments = felzenszwalb(
        img_as_float(frames[0]),
        scale=args.scale,
        sigma=args.sigma,
        min_size=args.min_size,
    )
    n_segments = int(np.unique(segments).size)
    seg_sec = time.time() - t1

    t2 = time.time()
    preds = segment_mean_predict(frames, segments)
    pred_sec = time.time() - t2
    predictor_psnr = psnr(frames, preds)

    flat_data = np.ascontiguousarray(frames.astype(np.float32).reshape(-1))
    flat_preds = np.ascontiguousarray(preds.astype(np.float32).reshape(-1))
    data_range = float(np.ptp(flat_data))
    results = []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="graphcomp_redsea_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for eb_text in args.ebs.split(","):
            eb = float(eb_text)
            cmp_path = tmpdir_path / f"redsea_eb{eb:g}.sz"
            t3 = time.time()
            cmp_size, _ = my_compress(flat_data.copy(), flat_preds.copy(), eb, str(cmp_path))
            comp_sec = time.time() - t3
            t4 = time.time()
            recon = decompress(str(cmp_path), flat_data.copy(), flat_preds.copy(), eb)
            dec_sec = time.time() - t4
            max_err = float(np.max(np.abs(flat_data - recon)))
            mse = float(np.mean((flat_data.astype(np.float64) - recon.astype(np.float64)) ** 2))
            results.append(
                {
                    "eb": eb,
                    "relative_bound": eb * data_range,
                    "cmp_size_bytes": int(cmp_size),
                    "residual_bpp": float(cmp_size * 8 / flat_data.size),
                    "psnr_db": psnr(flat_data, recon),
                    "mse": mse,
                    "max_abs_error": max_err,
                    "compress_sec": comp_sec,
                    "decompress_sec": dec_sec,
                }
            )

    summary = {
        "zip_path": str(args.zip_path),
        "member": args.member,
        "frames": args.frames,
        "shape": list(frames.shape),
        "dtype": str(frames.dtype),
        "data_min": float(np.min(frames)),
        "data_max": float(np.max(frames)),
        "data_range": data_range,
        "segment_params": {
            "scale": args.scale,
            "sigma": args.sigma,
            "min_size": args.min_size,
        },
        "n_segments": n_segments,
        "predictor_psnr_db": predictor_psnr,
        "timing_sec": {
            "read": read_sec,
            "segment": seg_sec,
            "predict": pred_sec,
        },
        "results": results,
    }
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
