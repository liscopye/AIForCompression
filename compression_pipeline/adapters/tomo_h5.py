from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from compression_pipeline.adapters.era5 import center_crop_chw
from compression_pipeline.canonical import CanonicalSample, DatasetManifest


class TomoH5Adapter:
    """Reads a tomopy-reconstructed HDF5 volume and yields each slice as a CHW sample.

    The reconstructed H5 contains ``data`` with shape (Z, H, W) in float32.

    When ``group_frames > 1`` consecutive slices are stacked as channels (e.g. group_frames=3
    produces [3, H, W] pseudo-RGB from three neighbouring slices).
    """

    def __init__(self, data_root: str | Path, dataset_id: str = "tomo", group_frames: int = 1) -> None:
        self.data_root = Path(data_root)
        self.dataset_id = dataset_id
        self.group_frames = group_frames
        if not self.data_root.exists():
            raise FileNotFoundError(f"Tomo H5 file does not exist: {self.data_root}")

    def manifest(self) -> DatasetManifest:
        import h5py
        with h5py.File(self.data_root, "r") as f:
            data = f["data"]
            n_slices, h, w = data.shape
        channels = self.group_frames
        sample_count = n_slices if channels == 1 else n_slices // channels
        return DatasetManifest(
            dataset_id=self.dataset_id,
            dataset_name="Tomography (reconstructed)",
            dataset_type="scientific_volume",
            source_format="hdf5",
            canonical_layout="channel_height_width",
            sample_count=sample_count,
            metadata={
                "data_root": str(self.data_root),
                "height": int(h),
                "width": int(w),
                "channels": channels,
                "dtype": "float32",
            },
        )

    def iter_samples(
        self,
        max_samples: int | None = None,
        resolution: tuple[int, int] | None = None,
    ) -> Iterator[CanonicalSample]:
        import h5py
        with h5py.File(self.data_root, "r") as f:
            data = f["data"]
            n_slices = data.shape[0]
            if max_samples is not None and max_samples > 0:
                n_slices = min(n_slices, max_samples)

            gf = self.group_frames
            if gf > 1:
                for start in range(0, n_slices - gf + 1, gf):
                    frames = []
                    for j in range(gf):
                        frame = data[start + j].astype(np.float32)
                        if resolution is not None:
                            frame = center_crop_chw(frame[None, ...], resolution)[0]
                        frames.append(frame)
                    chw = np.stack(frames, axis=0)
                    yield CanonicalSample(
                        dataset_id=self.dataset_id,
                        sample_id=f"slice_{start:04d}-{start+gf-1:04d}",
                        kind="scientific_field",
                        array=chw,
                        layout="channel_height_width",
                        metadata={
                            "source_path": str(self.data_root),
                            "dtype": "float32",
                            "source_dtype": "float32",
                            "height": int(chw.shape[1]),
                            "width": int(chw.shape[2]),
                            "channels": gf,
                            "slice_index": start,
                        },
                    )
                return

            for i in range(n_slices):
                frame = data[i].astype(np.float32)
                chw = frame[None, ...]
                if resolution is not None:
                    chw = center_crop_chw(chw, resolution)

                yield CanonicalSample(
                    dataset_id=self.dataset_id,
                    sample_id=f"slice_{i:04d}",
                    kind="scientific_field",
                    array=chw,
                    layout="channel_height_width",
                    metadata={
                        "source_path": str(self.data_root),
                        "dtype": "float32",
                        "source_dtype": "float32",
                        "height": int(chw.shape[1]),
                        "width": int(chw.shape[2]),
                        "channels": 1,
                        "slice_index": i,
                    },
                )

    def load_sequence(
        self,
        max_samples: int | None = None,
        resolution: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Load reconstructed slices as a sequence [C, T, H, W] for CAESAR/video models."""
        import h5py
        with h5py.File(self.data_root, "r") as f:
            data = f["data"]
            n_slices = data.shape[0]
            if max_samples is not None and max_samples > 0:
                n_slices = min(n_slices, max_samples)

            arrays = []
            timestamps = []
            for i in range(n_slices):
                frame = data[i].astype(np.float32)
                chw = frame[None, ...]
                if resolution is not None:
                    chw = center_crop_chw(chw, resolution)
                arrays.append(chw)
                timestamps.append(f"slice_{i:04d}")

            tchw = np.stack(arrays, axis=0)
            return np.transpose(tchw, (1, 0, 2, 3)), timestamps
