from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from compression_pipeline.canonical import CanonicalSample


class S2CAdapter:
    """Reads Sentinel-2 SAFE.zip and yields single-band images as 3-channel samples.

    Only supports bands stored as JP2 inside a SAFE-format zip archive.
    Defaults to B02 at 10m resolution.
    """

    def __init__(
        self,
        data_root: str | Path,
        band: str = "B02",
        resolution: str = "10m",
        dataset_id: str = "s2c",
        tile_size: int | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.band = band
        self.resolution = resolution
        self.dataset_id = dataset_id
        self.tile_size = tile_size

    def _resolve_path(self) -> tuple[zipfile.ZipFile | None, str]:
        """Return (zipfile_or_None, path_or_internal) for the requested band.
        Supports both SAFE.zip and extracted directory.
        """
        root = self.data_root
        if not root.exists():
            raise FileNotFoundError(f"Data root not found: {root}")

        jp2_band = f"{self.band}_{self.resolution}.jp2"

        # Mode 1: extracted directory — find JP2 directly
        if root.is_dir():
            for jp2 in root.rglob(jp2_band):
                if 'IMG_DATA' in str(jp2):
                    return None, str(jp2)
            # Also check if root is the SAFE directory itself
            img_data = root / "GRANULE"
            if img_data.exists():
                for jp2 in img_data.rglob(f"*{jp2_band}"):
                    return None, str(jp2)

        # Mode 2: zip file
        if root.suffix == '.zip':
            zf = zipfile.ZipFile(str(root), "r")
            for name in zf.namelist():
                if name.endswith(jp2_band):
                    return zf, name
            zf.close()

        raise FileNotFoundError(f"Band {self.band} at {self.resolution} not found in {root}")

    def iter_samples(self, max_samples: int = -1) -> Iterator[CanonicalSample]:
        zf, path = self._resolve_path()
        try:
            if zf is not None:
                raw = zf.read(path)
                img = Image.open(io.BytesIO(raw))
            else:
                img = Image.open(path)
            with img:
                hw = np.asarray(img, dtype=np.float32)

            h, w = hw.shape
            base_name = Path(path).stem

            if self.tile_size is None:
                chw = np.stack([hw, hw, hw], axis=0)
                yield self._make_sample(base_name, chw, path, h, w)
                return

            # Tiling mode: crop to multiples of tile_size, then split into tiles
            ts = self.tile_size
            valid_h = (h // ts) * ts
            valid_w = (w // ts) * ts
            offset_h = (h - valid_h) // 2
            offset_w = (w - valid_w) // 2
            cropped = hw[offset_h:offset_h + valid_h, offset_w:offset_w + valid_w]

            tiles_h = valid_h // ts
            tiles_w = valid_w // ts
            count = 0
            for th in range(tiles_h):
                for tw in range(tiles_w):
                    if max_samples > 0 and count >= max_samples:
                        return
                    r0, r1 = th * ts, (th + 1) * ts
                    c0, c1 = tw * ts, (tw + 1) * ts
                    tile = cropped[r0:r1, c0:c1]
                    # Skip tiles with near-zero variance (no-data margins, constant regions)
                    if tile.max() - tile.min() < 10:
                        continue
                    chw = np.stack([tile, tile, tile], axis=0)
                    yield self._make_sample(
                        f"{base_name}_t{th:03d}x{tw:03d}", chw, path, ts, ts,
                        tile_row=th, tile_col=tw,
                    )
                    count += 1
        finally:
            if zf is not None:
                zf.close()

    def load_sequence(
        self,
        max_samples: int | None = None,
        resolution: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Stack tiles as pseudo-sequence for CAESAR [V=1, T, H, W]."""
        samples = list(self.iter_samples(max_samples=-1))
        if max_samples is not None and max_samples > 0:
            samples = samples[:max_samples]
        if not samples:
            raise ValueError("No tiles available")
        # Extract mid-channel (all 3 identical) and stack as [T, H, W]
        frames = [s.array[0].astype(np.float32) for s in samples]  # take first channel
        t = len(frames)
        data = np.stack(frames)  # [T, H, W]
        if resolution is not None:
            from compression_pipeline.adapters.era5 import center_crop_vthw
            data = center_crop_vthw(data[np.newaxis, ...], resolution)[0]
        # Pad by repeating if needed (for CAESAR-D)
        if max_samples is not None and max_samples > t:
            pad = np.repeat(data[-1:], max_samples - t, axis=0)
            data = np.concatenate([data, pad], axis=0)
            t = data.shape[0]
        sequence = data[np.newaxis, ...]  # [1, T, H, W]
        timestamps = [f"2024-01-01T{i:02d}:00:00" for i in range(t)]
        return sequence, timestamps

    def _make_sample(self, sample_id, chw, source_path, h, w, **extra):
        meta = {
            "source_path": source_path,
            "source_format": "jp2",
            "dtype": "float32",
            "height": h,
            "width": w,
            "channels": 3,
            "band": self.band,
            "resolution": self.resolution,
        }
        meta.update(extra)
        return CanonicalSample(
            dataset_id=self.dataset_id,
            sample_id=sample_id,
            kind="s2c",
            array=chw,
            layout="channel_height_width",
            metadata=meta,
        )
