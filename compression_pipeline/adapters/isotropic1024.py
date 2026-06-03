from __future__ import annotations

from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from compression_pipeline.canonical import CanonicalSample


class Isotropic1024Adapter:
    """Reads isotropic1024 turbulence velocity H5 data.

    Each H5 dataset is [X, Y, Z, 3] float32 — 3 velocity components (u,v,w).
    Each Z-slice is directly [3, H, W] with natural 3-channel structure.
    """

    def __init__(
        self,
        data_root: str | Path,
        dataset_id: str = "isotropic1024",
    ) -> None:
        self.data_root = Path(data_root)
        self.dataset_id = dataset_id

    def _data_keys(self, f: h5py.File) -> list[str]:
        for prefix in ("Velocity_", "Pressure_"):
            keys = sorted(k for k in f.keys() if k.startswith(prefix))
            if keys:
                return keys
        raise FileNotFoundError(f"No Velocity_* or Pressure_* in {self.data_root}")

    def load_sequence(
        self,
        max_samples: int | None = None,
        resolution: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Load all timesteps as CAESAR sequence [V=3, T, H, W] using mid-Z slice."""
        with h5py.File(str(self.data_root), "r") as f:
            keys = self._data_keys(f)
            frames = []
            for key in keys:
                vol = f[key][:]  # [X, Y, Z, 3]
                mid_z = vol[:, :, vol.shape[2] // 2, :]  # [256, 256, 3]
                frames.append(np.transpose(mid_z, (2, 0, 1)))  # [3, 256, 256]
            data = np.stack(frames, axis=1).astype(np.float32)  # [3, T, H, W]
        t = data.shape[1]
        # Pad by repeating last frame if requesting more than available (for CAESAR-D)
        if max_samples is not None and max_samples > t:
            pad_len = max_samples - t
            pad = np.repeat(data[:, -1:], pad_len, axis=1)
            data = np.concatenate([data, pad], axis=1)
            t = max_samples
        elif max_samples is not None and max_samples > 0:
            data = data[:, :max_samples]
            t = data.shape[1]
        if resolution is not None:
            from compression_pipeline.adapters.era5 import center_crop_vthw
            data = center_crop_vthw(data, resolution)
        timestamps = [f"2024-01-01T{i:02d}:00:00" for i in range(t)]
        return data, timestamps

    def iter_samples(self, max_samples: int = -1) -> Iterator[CanonicalSample]:
        with h5py.File(str(self.data_root), "r") as f:
            data_keys = self._data_keys(f)

            count = 0
            for key in data_keys:
                if max_samples > 0 and count >= max_samples:
                    return
                # shape: [X, Y, Z, 3] (256, 256, 256, 3)
                vol = f[key][:]

                d = vol.shape[2]
                for z in range(d):
                    if max_samples > 0 and count >= max_samples:
                        return
                    slice_3c = vol[:, :, z, :]  # (256, 256, 3)
                    chw = np.transpose(slice_3c, (2, 0, 1))  # (3, 256, 256)

                    yield CanonicalSample(
                        dataset_id=self.dataset_id,
                        sample_id=f"{key}_z{z:03d}",
                        kind="isotropic1024",
                        array=chw.astype(np.float32),
                        layout="channel_height_width",
                        metadata={
                            "source_path": str(self.data_root),
                            "source_format": "h5",
                            "dtype": "float32",
                            "height": int(chw.shape[1]),
                            "width": int(chw.shape[2]),
                            "channels": 3,
                            "timestep": key,
                            "slice_z": int(z),
                        },
                    )
                    count += 1
