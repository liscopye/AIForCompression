#!/usr/bin/env python3
"""
extract_caesar_npz_to_npy.py

Convert CAESAR scientific-data archives from .npz to mmap-friendly .npy files
without materializing the full array in RAM.

Expected input format:
    np.savez(..., data=array)    # array shape [V, S, T, H, W]

Why this script exists:
    np.load("large.npz", mmap_mode="r") does not provide true memory-mapped
    access to the member array. A standalone .npy file does.

Example:
    python extract_caesar_npz_to_npy.py \
        --input /workspace/Data/lysozyme_processed/lysozyme_train_nf16.npz \
                /workspace/Data/lysozyme_processed/lysozyme_test_nf16.npz \
        --output_dir /workspace/Data/lysozyme_processed/mmap
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm


def human_bytes(n: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(n)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def stream_extract_member(
    npz_path: Path,
    output_path: Path,
    key: str,
    overwrite: bool,
    chunk_mb: int,
) -> Path:
    """Extract <key>.npy from an .npz zip container using bounded RAM."""
    if npz_path.suffix.lower() != ".npz":
        raise ValueError(f"Only .npz input is supported, got: {npz_path}")
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    member_name = f"{key}.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        print(f"[skip] exists: {output_path}")
        return output_path

    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    with zipfile.ZipFile(npz_path, "r") as archive:
        try:
            member = archive.getinfo(member_name)
        except KeyError as exc:
            available = ", ".join(archive.namelist())
            raise KeyError(
                f"{member_name!r} not found in {npz_path}. Available entries: {available}"
            ) from exc

        total = member.file_size
        desc = f"extract {npz_path.name}:{member_name}"
        with archive.open(member, "r") as src, open(tmp_path, "wb") as dst:
            with tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar:
                while True:
                    chunk = src.read(chunk_mb * 1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
                    pbar.update(len(chunk))
            dst.flush()
            os.fsync(dst.fileno())

    os.replace(tmp_path, output_path)
    return output_path


def validate_npy(path: Path) -> None:
    """Validate shape and prove the output is mmap-backed."""
    array = np.load(path, mmap_mode="r")
    if not isinstance(array, np.memmap):
        raise RuntimeError(f"{path} was not opened as np.memmap.")
    if array.ndim != 5:
        raise ValueError(
            f"Expected [V, S, T, H, W] array, got shape={array.shape} in {path}"
        )
    print(
        f"[ok] {path}\n"
        f"     shape={tuple(array.shape)}, dtype={array.dtype}, "
        f"file_size={human_bytes(path.stat().st_size)}, mmap=True"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract data.npy from CAESAR .npz archive(s) for true mmap loading."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        type=Path,
        help="One or more .npz files containing the 'data' array.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory where standalone .npy files will be written.",
    )
    parser.add_argument(
        "--key",
        default="data",
        help="Array key inside each .npz file. Default: data",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional suffix appended to each output stem, e.g. '_mmap'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--chunk_mb",
        type=int,
        default=64,
        help="Streaming extraction buffer in MB. Default: 64.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_mb <= 0:
        raise ValueError("--chunk_mb must be positive.")

    for npz_path in args.input:
        out_name = f"{npz_path.stem}{args.suffix}.npy"
        out_path = args.output_dir / out_name
        written = stream_extract_member(
            npz_path=npz_path,
            output_path=out_path,
            key=args.key,
            overwrite=args.overwrite,
            chunk_mb=args.chunk_mb,
        )
        validate_npy(written)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise
