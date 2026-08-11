from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from compression_pipeline.canonical import CanonicalSample


class HurricaneAdapter:
    """Reads hurricane .bin.f32 fields and yields 3-timestep grouped samples.

    Each .bin.f32 file is [T, H, W] float32. Three consecutive timesteps are
    stacked to form a 3-channel input for image compression models.
    """

    def __init__(self, data_root: str | Path, channel: str = "P", dataset_id: str = "hurricane") -> None:
        self.data_root = Path(data_root)
        self.channel = channel
        self.dataset_id = dataset_id

    def _find_file(self) -> Path:
        """Find the .bin.f32 file matching the requested channel."""
        candidates = list(self.data_root.rglob(f"{self.channel}*.bin.f32"))
        if not candidates:
            raise FileNotFoundError(f"No .bin.f32 file for channel '{self.channel}' in {self.data_root}")
        return candidates[0]

    def iter_samples(self, max_samples: int = -1) -> Iterator[CanonicalSample]:
        """Group every 3 consecutive timesteps into a [3, H, W] sample."""
        filepath = self._find_file()
        data = np.fromfile(filepath, dtype=np.float32)

        # Detect shape from known dataset layout: 100 x 500 x 500
        # But support arbitrary shape detection from filename or data size
        total = data.size
        # Try known sizes
        if total == 100 * 500 * 500:
            t, h, w = 100, 500, 500
        elif total == 500 * 500 * 100:
            t, h, w = 500, 500, 100
        else:
            raise ValueError(f"Cannot infer shape for {total} elements in {filepath}")

        data = data.reshape(t, h, w)
        group_size = 3
        num_groups = t // group_size
        if max_samples > 0:
            num_groups = min(num_groups, max_samples)

        for g in range(num_groups):
            start = g * group_size
            chunk = data[start:start + group_size]  # [3, H, W]
            yield CanonicalSample(
                dataset_id=self.dataset_id,
                sample_id=f"{self.channel}_t{start:03d}-{start + group_size - 1:03d}",
                kind="hurricane",
                array=chunk.astype(np.float32),
                layout="channel_height_width",
                metadata={
                    "source_path": str(filepath),
                    "source_format": "bin.f32",
                    "dtype": "float32",
                    "height": h,
                    "width": w,
                    "channels": group_size,
                    "timestep_range": [int(start), int(start + group_size)],
                    "channel_var": self.channel,
                },
            )

    def load_sequence(
        self,
        max_samples: int | None = None,
        resolution: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Load full time series as CAESAR sequence [V=1, T, H, W]."""
        filepath = self._find_file()
        data = np.fromfile(filepath, dtype=np.float32)
        total = data.size
        if total == 100 * 500 * 500:
            t, h, w = 100, 500, 500
        else:
            raise ValueError(f"Cannot infer shape for {total} elements")
        data = data.reshape(t, h, w).astype(np.float32)
        if max_samples is not None and max_samples > 0:
            data = data[:max_samples]
            t = data.shape[0]
        if resolution is not None:
            from compression_pipeline.adapters.era5 import center_crop_vthw
            data = center_crop_vthw(data[np.newaxis, ...], resolution)[0]
        # [T, H, W] -> [V=1, T, H, W]
        sequence = data[np.newaxis, ...]
        timestamps = [f"2024-01-01T{i:02d}:00:00" for i in range(t)]
        return sequence, timestamps
