from __future__ import annotations

from datetime import datetime, timedelta
import io
import zipfile
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from compression_pipeline.canonical import CanonicalSample


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class S2CAdapter:
    """Reads Sentinel-2 SAFE.zip and yields multi-band 3-channel samples.

    Only supports bands stored as JP2 inside a SAFE-format zip archive.
    Defaults to the SAFE true-color RGB TCI product at 10m resolution.
    """

    def __init__(
        self,
        data_root: str | Path,
        band: str = "B02",
        bands: tuple[str, ...] | None = None,
        resolution: str = "10m",
        dataset_id: str = "s2c",
        tile_size: int | None = None,
        use_tci: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.bands = tuple(bands) if bands is not None else ("B04", "B03", "B02")
        self.band = band
        self.resolution = resolution
        self.dataset_id = dataset_id
        self.tile_size = tile_size
        self.use_tci = use_tci

    def _resolve_path(self, band: str) -> tuple[zipfile.ZipFile | None, str]:
        """Return (zipfile_or_None, path_or_internal) for the requested band.
        Supports both SAFE.zip and extracted directory.
        """
        root = self.data_root
        if not root.exists():
            raise FileNotFoundError(f"Data root not found: {root}")

        jp2_band = f"{band}_{self.resolution}.jp2"

        # Mode 1: extracted directory — find JP2 directly
        if root.is_dir():
            for jp2 in root.rglob(f"*{jp2_band}"):
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

        raise FileNotFoundError(f"Band {band} at {self.resolution} not found in {root}")

    def _image_tile_paths(self) -> list[Path]:
        if not self.data_root.is_dir():
            return []
        return sorted(
            path for path in self.data_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def iter_samples(self, max_samples: int = -1) -> Iterator[CanonicalSample]:
        image_tiles = self._image_tile_paths()
        if image_tiles:
            if max_samples > 0:
                image_tiles = image_tiles[:max_samples]
            for path in image_tiles:
                with Image.open(path) as img:
                    rgb = img.convert("RGB")
                    hwc = np.asarray(rgb, dtype=np.float32)
                chw = np.transpose(hwc, (2, 0, 1))
                yield self._make_sample(
                    path.stem,
                    chw,
                    str(path),
                    int(chw.shape[1]),
                    int(chw.shape[2]),
                    source_product="TCI_PRETILED",
                )
            return

        if self.use_tci:
            yield from self._iter_tci_samples(max_samples=max_samples)
            return

        readers = [self._resolve_path(band) for band in self.bands]
        images = []
        try:
            paths = []
            for zf, path in readers:
                if zf is not None:
                    raw = zf.read(path)
                    img = Image.open(io.BytesIO(raw))
                else:
                    img = Image.open(path)
                images.append(img)
                paths.append(path)

            shapes = {(img.height, img.width) for img in images}
            if len(shapes) != 1:
                raise ValueError(f"S2C bands have mismatched shapes: {sorted(shapes)}")

            h, w = next(iter(shapes))
            base_name = Path(paths[0]).stem

            if self.tile_size is None:
                planes = [np.asarray(img, dtype=np.float32) for img in images]
                chw_full = np.stack(planes, axis=0)
                yield self._make_sample(base_name, chw_full, paths, h, w)
                return

            # Tiling mode: crop each JP2 band before materializing numpy arrays.
            ts = self.tile_size
            valid_h = (h // ts) * ts
            valid_w = (w // ts) * ts
            offset_h = (h - valid_h) // 2
            offset_w = (w - valid_w) // 2

            tiles_h = valid_h // ts
            tiles_w = valid_w // ts
            count = 0
            for th in range(tiles_h):
                for tw in range(tiles_w):
                    if max_samples > 0 and count >= max_samples:
                        return
                    r0, r1 = th * ts, (th + 1) * ts
                    c0, c1 = tw * ts, (tw + 1) * ts
                    box = (offset_w + c0, offset_h + r0, offset_w + c1, offset_h + r1)
                    planes = [np.asarray(img.crop(box), dtype=np.float32) for img in images]
                    tile = np.stack(planes, axis=0)
                    # Skip tiles with near-zero variance (no-data margins, constant regions)
                    if tile.max() - tile.min() < 10:
                        continue
                    yield self._make_sample(
                        f"{base_name}_t{th:03d}x{tw:03d}", tile, paths, ts, ts,
                        tile_row=th, tile_col=tw,
                    )
                    count += 1
        finally:
            for img in images:
                img.close()
            for zf, _ in readers:
                if zf is not None:
                    zf.close()

    def _iter_tci_samples(self, max_samples: int = -1) -> Iterator[CanonicalSample]:
        zf, path = self._resolve_path("TCI")
        img = None
        try:
            if zf is not None:
                raw = zf.read(path)
                img = Image.open(io.BytesIO(raw))
            else:
                img = Image.open(path)
            if img.mode != "RGB":
                img = img.convert("RGB")

            h, w = img.height, img.width
            base_name = Path(path).stem
            if self.tile_size is None:
                hwc = np.asarray(img, dtype=np.float32)
                chw = np.transpose(hwc, (2, 0, 1))
                yield self._make_sample(base_name, chw, path, h, w, source_product="TCI")
                return

            ts = self.tile_size
            valid_h = (h // ts) * ts
            valid_w = (w // ts) * ts
            offset_h = (h - valid_h) // 2
            offset_w = (w - valid_w) // 2
            tiles_h = valid_h // ts
            tiles_w = valid_w // ts
            count = 0
            for th in range(tiles_h):
                for tw in range(tiles_w):
                    if max_samples > 0 and count >= max_samples:
                        return
                    r0, r1 = th * ts, (th + 1) * ts
                    c0, c1 = tw * ts, (tw + 1) * ts
                    box = (offset_w + c0, offset_h + r0, offset_w + c1, offset_h + r1)
                    hwc = np.asarray(img.crop(box), dtype=np.float32)
                    tile = np.transpose(hwc, (2, 0, 1))
                    if tile.max() - tile.min() < 10:
                        continue
                    yield self._make_sample(
                        f"{base_name}_t{th:03d}x{tw:03d}",
                        tile,
                        path,
                        ts,
                        ts,
                        tile_row=th,
                        tile_col=tw,
                        source_product="TCI",
                    )
                    count += 1
        finally:
            if img is not None:
                img.close()
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
        # Use the first selected band as pseudo-time frames for CAESAR.
        frames = [s.array[0].astype(np.float32) for s in samples]
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
        start = datetime(2024, 1, 1)
        timestamps = [(start + timedelta(hours=i)).isoformat() for i in range(t)]
        return sequence, timestamps

    def _make_sample(self, sample_id, chw, source_path, h, w, **extra):
        meta = {
            "source_path": source_path,
            "source_format": "jp2",
            "dtype": "float32",
            "height": h,
            "width": w,
            "channels": 3,
            "bands": list(self.bands) if not self.use_tci else ["TCI_R", "TCI_G", "TCI_B"],
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
