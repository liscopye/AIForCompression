from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import hdf5plugin  # register LZ4 filter for CHESS HDF5 files
import h5py
import numpy as np

from compression_pipeline.canonical import CanonicalSample


class LysozymeAdapter:
    """Reads CHESS lysozyme serial crystallography H5 files.

    Each H5 file is one diffraction frame with /entry/data/data of shape
    [1, H, W] uint32. The single channel is replicated to 3 channels.
    """

    def __init__(
        self,
        data_root: str | Path,
        dataset_id: str = "lysozyme",
    ) -> None:
        self.data_root = Path(data_root)
        self.dataset_id = dataset_id

    def _h5_files(self) -> list[Path]:
        return sorted(
            p for p in self.data_root.glob("*.h5") if p.is_file()
        )

    def _processed_array_path(self) -> Path | None:
        if self.data_root.is_file() and self.data_root.suffix in {".npy", ".npz"}:
            return self.data_root
        candidates = [
            self.data_root / "mmap" / "lysozyme_test_nf16.npy",
            self.data_root / "mmap" / "lysozyme_test_nf8.npy",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _load_processed(self):
        path = self._processed_array_path()
        if path is None:
            return None, None
        if path.suffix == ".npy":
            return np.load(path, mmap_mode="r"), path
        data = np.load(path)
        return data["data"], path

    def load_sequence(
        self,
        max_samples: int | None = None,
        resolution: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Stack frames as CAESAR sequence [V=1, T, H, W]."""
        processed, processed_path = self._load_processed()
        if processed is not None:
            # Processed lysozyme layout is [V, N_chunks, T, H, W].
            arr = processed[0]
            flat = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2], arr.shape[3]).astype(np.float32)
            if max_samples is not None and max_samples > 0:
                flat = flat[:max_samples]
            if resolution is not None:
                from compression_pipeline.adapters.era5 import center_crop_vthw
                flat = center_crop_vthw(flat[np.newaxis, ...], resolution)[0]
            sequence = flat[np.newaxis, ...]
            start = datetime(2024, 1, 1)
            timestamps = [(start + timedelta(hours=i)).isoformat() for i in range(flat.shape[0])]
            return sequence, timestamps

        files = self._h5_files()
        if max_samples is not None and max_samples > 0:
            files = files[:max_samples]
        frames = []
        for fp in files:
            with h5py.File(str(fp), "r") as f:
                img = f["/entry/data/data"][0].astype(np.float32)  # [H, W]
            if resolution is not None:
                from compression_pipeline.adapters.era5 import center_crop_hw
                img = center_crop_hw(img, resolution)
            frames.append(img)
        t = len(frames)
        data = np.stack(frames)[np.newaxis]  # [1, T, H, W]
        # Pad if needed (CAESAR-D)
        if max_samples is not None and max_samples > t:
            pad = np.repeat(data[:, -1:], max_samples - t, axis=1)
            data = np.concatenate([data, pad], axis=1)
            t = data.shape[1]
        start = datetime(2024, 1, 1)
        timestamps = [(start + timedelta(hours=i)).isoformat() for i in range(t)]
        return data, timestamps

    def iter_samples(self, max_samples: int = -1) -> Iterator[CanonicalSample]:
        processed, processed_path = self._load_processed()
        if processed is not None:
            arr = processed[0]
            count = arr.shape[0] if max_samples <= 0 else min(arr.shape[0], max_samples)
            for idx in range(count):
                chunk = np.asarray(arr[idx, :3], dtype=np.float32)
                if chunk.shape[0] < 3:
                    chunk = np.repeat(chunk[-1:], 3, axis=0)
                yield CanonicalSample(
                    dataset_id=self.dataset_id,
                    sample_id=f"chunk_{idx:04d}_frames000-002",
                    kind="lysozyme",
                    array=chunk,
                    layout="channel_height_width",
                    metadata={
                        "source_path": str(processed_path),
                        "source_format": processed_path.suffix.lstrip("."),
                        "dtype": "float32",
                        "height": int(chunk.shape[1]),
                        "width": int(chunk.shape[2]),
                        "channels": 3,
                        "chunk_index": int(idx),
                        "frame_range": [0, 3],
                    },
                )
            return

        files = self._h5_files()
        if not files:
            raise FileNotFoundError(f"No H5 files in {self.data_root}")
        if max_samples > 0:
            files = files[:max_samples]

        for fp in files:
            with h5py.File(str(fp), "r") as f:
                img = f["/entry/data/data"][0].astype(np.float32)  # [H, W]
            h, w = img.shape
            chw = np.stack([img, img, img], axis=0)  # replicate to 3 channels
            yield CanonicalSample(
                dataset_id=self.dataset_id,
                sample_id=fp.stem,
                kind="lysozyme",
                array=chw,
                layout="channel_height_width",
                metadata={
                    "source_path": str(fp),
                    "source_format": "h5",
                    "dtype": "float32",
                    "height": h,
                    "width": w,
                    "channels": 3,
                },
            )
