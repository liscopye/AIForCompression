from __future__ import annotations

from pathlib import Path
import re
from typing import Iterator

import numpy as np

from compression_pipeline.canonical import CanonicalSample


def _yuv420_to_rgb(yuv_data: bytes, height: int, width: int) -> np.ndarray:
    """Convert a single YUV420 (I420) frame to RGB [C, H, W] uint8."""
    # Y plane: H x W
    y_size = height * width
    y = np.frombuffer(yuv_data, dtype=np.uint8, count=y_size).reshape(height, width)

    # U, V planes: (H/2) x (W/2)
    uv_h, uv_w = height // 2, width // 2
    uv_size = uv_h * uv_w
    u = np.frombuffer(yuv_data, dtype=np.uint8, offset=y_size, count=uv_size).reshape(uv_h, uv_w)
    v = np.frombuffer(yuv_data, dtype=np.uint8, offset=y_size + uv_size, count=uv_size).reshape(uv_h, uv_w)

    # Upsample U, V to full resolution
    u_up = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
    v_up = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)

    # BT.601 full swing: YUV -> RGB
    yf = y.astype(np.float32)
    uf = (u_up.astype(np.float32) - 128.0)
    vf = (v_up.astype(np.float32) - 128.0)

    r = np.clip(np.rint(yf + 1.402 * vf), 0, 255).astype(np.uint8)
    g = np.clip(np.rint(yf - 0.344136 * uf - 0.714136 * vf), 0, 255).astype(np.uint8)
    b = np.clip(np.rint(yf + 1.772 * uf), 0, 255).astype(np.uint8)

    return np.stack([r, g, b], axis=0)


class UVGAdapter:
    """Reads a YUV420 video file and yields RGB frames as canonical CHW samples."""

    def __init__(self, data_root: str | Path, dataset_id: str = "uvg") -> None:
        self.data_root = Path(data_root)
        self.dataset_id = dataset_id
        self._yuv_paths: list[Path] = []

    def _find_yuv(self) -> Path:
        if self._yuv_paths:
            return self._yuv_paths[0]
        paths = sorted(self.data_root.glob("*.yuv"))
        if not paths:
            raise FileNotFoundError(f"No .yuv file found in {self.data_root}")
        self._yuv_paths = paths
        return paths[0]

    def _frame_size(self, height: int, width: int) -> int:
        return height * width * 3 // 2  # YUV420 = 1.5 bytes/pixel

    def _resolution(self) -> tuple[int, int]:
        yuv_path = self._find_yuv()
        match = re.search(r"(\d{3,5})x(\d{3,5})", yuv_path.name)
        if match:
            width = int(match.group(1))
            height = int(match.group(2))
            return height, width
        return 2160, 3840

    def load_sequence(
        self,
        max_samples: int | None = None,
        resolution: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Load video frames as CAESAR sequence [V=3, T, H, W] uint8→float32."""
        yuv_path = self._find_yuv()
        height, width = self._resolution()
        frame_size = self._frame_size(height, width)
        total_frames = min(30, max_samples or 30)
        frames = []
        with open(yuv_path, "rb") as f:
            for i in range(total_frames):
                raw = f.read(frame_size)
                if len(raw) < frame_size:
                    break
                rgb = _yuv420_to_rgb(raw, height, width)
                frames.append(rgb.astype(np.float32) / 255.0)
        arrays = np.stack(frames, axis=1)  # [C, T, H, W]
        t = arrays.shape[1]
        if resolution is not None:
            from compression_pipeline.adapters.era5 import center_crop_vthw
            arrays = center_crop_vthw(arrays, resolution)
        timestamps = [f"2024-01-01T{i:02d}:00:00" for i in range(t)]
        return arrays, timestamps

    def iter_samples(
        self,
        max_samples: int = 30,
        height: int | None = None,
        width: int | None = None,
    ) -> Iterator[CanonicalSample]:
        yuv_path = self._find_yuv()
        if height is None or width is None:
            height, width = self._resolution()
        frame_size = self._frame_size(height, width)
        file_size = yuv_path.stat().st_size
        total_frames = file_size // frame_size
        num_frames = min(max_samples, total_frames)

        with open(yuv_path, "rb") as f:
            for i in range(num_frames):
                raw = f.read(frame_size)
                if len(raw) < frame_size:
                    break
                rgb_chw = _yuv420_to_rgb(raw, height, width)
                yield CanonicalSample(
                    dataset_id=self.dataset_id,
                    sample_id=f"frame_{i:04d}",
                    kind="image",
                    array=rgb_chw,
                    layout="channel_height_width",
                    metadata={
                        "source_path": str(yuv_path),
                        "source_format": "yuv420",
                        "dtype": "uint8",
                        "height": height,
                        "width": width,
                        "channels": 3,
                        "frame_index": i,
                    },
                )
