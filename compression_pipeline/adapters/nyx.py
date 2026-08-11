from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from compression_pipeline.canonical import CanonicalSample


class NYXAdapter:
    """Reads NYX cosmology .f32 volume, slices along Z, yields 3-slice groups.

    Each .f32 file is a [D, H, W] float32 volume. Three adjacent Z-slices are
    stacked to form a 3-channel input for image compression models.
    """

    def __init__(
        self,
        data_root: str | Path,
        field: str = "baryon_density",
        dataset_id: str = "nyx",
    ) -> None:
        self.data_root = Path(data_root)
        self.field = field
        self.dataset_id = dataset_id

    def _find_file(self) -> Path:
        candidates = list(self.data_root.rglob(f"{self.field}.f32"))
        if not candidates:
            raise FileNotFoundError(f"No .f32 file for field '{self.field}' in {self.data_root}")
        return candidates[0]

    def load_sequence(
        self,
        max_samples: int | None = None,
        resolution: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Load full volume as CAESAR sequence [V=1, T=Z, H, W]."""
        filepath = self._find_file()
        data = np.fromfile(filepath, dtype=np.float32)
        d, h, w = 512, 512, 512
        if data.size != d * h * w:
            raise ValueError(f"Expected {d}x{h}x{w}={d*h*w}, got {data.size}")
        data = data.reshape(d, h, w).astype(np.float32)
        if max_samples is not None and max_samples > 0:
            data = data[:max_samples]
            d = data.shape[0]
        if resolution is not None:
            from compression_pipeline.adapters.era5 import center_crop_vthw
            data = center_crop_vthw(data[np.newaxis, ...], resolution)[0]
        sequence = data[np.newaxis, ...]  # [1, T=Z, H, W]
        timestamps = [f"2024-01-01T{i:02d}:00:00" for i in range(d)]
        return sequence, timestamps

    def iter_samples(self, max_samples: int = -1) -> Iterator[CanonicalSample]:
        """Slice Z-axis in groups of 3, yielding [3, H, W] float32 samples."""
        filepath = self._find_file()
        data = np.fromfile(filepath, dtype=np.float32)
        # Expected: 512 x 512 x 512 = 134217728
        d, h, w = 512, 512, 512
        if data.size != d * h * w:
            raise ValueError(f"Expected {d}x{h}x{w}={d*h*w} elements, got {data.size}")
        data = data.reshape(d, h, w)

        group_size = 3
        num_groups = d // group_size
        if max_samples > 0:
            num_groups = min(num_groups, max_samples)

        for g in range(num_groups):
            start = g * group_size
            chunk = data[start:start + group_size]  # [3, H, W]
            yield CanonicalSample(
                dataset_id=self.dataset_id,
                sample_id=f"{self.field}_z{start:03d}-{start + group_size - 1:03d}",
                kind="nyx",
                array=chunk.astype(np.float32),
                layout="channel_height_width",
                metadata={
                    "source_path": str(filepath),
                    "source_format": "f32",
                    "dtype": "float32",
                    "height": h,
                    "width": w,
                    "channels": group_size,
                    "slice_range": [int(start), int(start + group_size)],
                    "field": self.field,
                },
            )
