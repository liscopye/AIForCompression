from __future__ import annotations

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

    def load_sequence(
        self,
        max_samples: int | None = None,
        resolution: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Stack frames as CAESAR sequence [V=1, T, H, W]."""
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
        timestamps = [f"2024-01-01T{i:02d}:00:00" for i in range(t)]
        return data, timestamps

    def iter_samples(self, max_samples: int = -1) -> Iterator[CanonicalSample]:
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
