from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from compression_pipeline.adapters.era5 import center_crop_chw
from compression_pipeline.canonical import CanonicalSample, DatasetManifest


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class KodakAdapter:
    """Reads a Kodak-style RGB image directory into canonical CHW samples."""

    def __init__(self, data_root: str | Path, dataset_id: str = "kodak") -> None:
        self.data_root = Path(data_root)
        self.dataset_id = dataset_id

    def image_paths(self) -> list[Path]:
        if not self.data_root.exists():
            raise FileNotFoundError(f"Kodak data root does not exist: {self.data_root}")
        return sorted(
            path for path in self.data_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def manifest(self) -> DatasetManifest:
        paths = self.image_paths()
        return DatasetManifest(
            dataset_id=self.dataset_id,
            dataset_name="Kodak",
            dataset_type="image",
            source_format="image_folder",
            canonical_layout="channel_height_width",
            sample_count=len(paths),
            metadata={"data_root": str(self.data_root)},
        )

    def iter_samples(self, max_samples: int | None = None) -> Iterator[CanonicalSample]:
        paths = self.image_paths()
        if max_samples is not None and max_samples > 0:
            paths = paths[:max_samples]
        for path in paths:
            with Image.open(path) as img:
                rgb = img.convert("RGB")
                hwc = np.asarray(rgb, dtype=np.uint8)
            chw = np.transpose(hwc, (2, 0, 1))
            yield CanonicalSample(
                dataset_id=self.dataset_id,
                sample_id=path.stem,
                kind="image",
                array=chw,
                layout="channel_height_width",
                metadata={
                    "source_path": str(path),
                    "source_format": path.suffix.lower().lstrip("."),
                    "dtype": "uint8",
                    "height": int(chw.shape[1]),
                    "width": int(chw.shape[2]),
                    "channels": 3,
                },
            )

    def load_sequence(
        self,
        max_samples: int | None = None,
        resolution: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Load Kodak images as a pseudo-time sequence for CAESAR models.

        Returns:
            sequence_vthw: [C, T, H, W] float32 array (C=3 for RGB)
            timestamps: fake ISO timestamps with 1-hour intervals
        """
        samples = list(self.iter_samples(max_samples=max_samples))
        if not samples:
            raise ValueError(f"No Kodak images found in {self.data_root}")

        arrays = []
        for sample in samples:
            arr = sample.array.astype(np.float32)
            if resolution is not None:
                arr = center_crop_chw(arr, resolution)
            arrays.append(arr)

        start = datetime(2024, 1, 1)
        timestamps = [(start + timedelta(hours=i)).isoformat() for i in range(len(samples))]
        tchw = np.stack(arrays, axis=0)
        return np.transpose(tchw, (1, 0, 2, 3)), timestamps
