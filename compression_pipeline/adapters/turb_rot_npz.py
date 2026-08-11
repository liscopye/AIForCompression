from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from compression_pipeline.canonical import CanonicalSample


class TurbRotNPZAdapter:
    """Reads paper-style turbulence NPZ data in [V,S,T,H,W] layout."""

    def __init__(
        self,
        data_root: str | Path,
        dataset_id: str = "turb_rot_npz",
        section_index: int = 0,
        section_start: int = 0,
        time_start: int = 0,
        image_group_mode: str = "auto",
        image_channel_count: int | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.dataset_id = dataset_id
        self.section_index = int(section_index)
        self.section_start = int(section_start)
        self.time_start = int(time_start)
        if image_group_mode not in {"auto", "variables", "sections"}:
            raise ValueError(f"Unsupported image_group_mode={image_group_mode!r}")
        self.image_group_mode = image_group_mode
        self.image_channel_count = None if image_channel_count is None or image_channel_count <= 0 else int(image_channel_count)

    def _load_npz(self):
        if self.data_root.is_dir():
            candidates = sorted(self.data_root.glob("*.npz"))
            if not candidates:
                raise FileNotFoundError(f"No .npz files found in {self.data_root}")
            path = candidates[0]
        else:
            path = self.data_root
        if not path.exists():
            raise FileNotFoundError(path)
        return path, np.load(path, allow_pickle=False)

    def _metadata(self, path: Path, data: np.ndarray, variable_name: np.ndarray | None) -> dict:
        variables = [] if variable_name is None else [str(v) for v in variable_name.tolist()]
        return {
            "source_path": str(path),
            "source_format": "npz",
            "source_layout": "V,S,T,H,W",
            "dtype": str(data.dtype),
            "variable_name": variables,
            "variables_in_data": int(data.shape[0]),
            "sections": int(data.shape[1]),
            "timesteps": int(data.shape[2]),
            "height": int(data.shape[3]),
            "width": int(data.shape[4]),
        }

    def _data_and_metadata(self):
        path, handle = self._load_npz()
        try:
            if "data" not in handle:
                raise KeyError(f"{path} does not contain a 'data' array")
            data = handle["data"]
            if data.ndim != 5:
                raise ValueError(f"Turb_Rot data must be [V,S,T,H,W], got {data.shape}")
            variable_name = handle["variable_name"] if "variable_name" in handle else None
            return path, data, self._metadata(path, data, variable_name)
        finally:
            handle.close()

    def load_sequence(
        self,
        max_samples: int | None = None,
        resolution: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Return a CAESAR sequence [V,T,H,W] from one section slice."""
        _, data, _ = self._data_and_metadata()
        mode = self._resolved_image_group_mode(data)
        if mode == "sections":
            section = self._checked_section_index(data.shape[1], self.section_index)
            section_indices = list(range(section, min(section + 3, data.shape[1])))
            while len(section_indices) < 3:
                section_indices.append(section_indices[-1])
            sequence = data[0, section_indices].astype(np.float32, copy=False)
        else:
            section = self._checked_section_index(data.shape[1], self.section_index)
            variable_indices = self._variable_indices(data.shape[0])
            sequence = data[variable_indices, section].astype(np.float32, copy=False)
        time_start = self._checked_time_index(sequence.shape[1], self.time_start)
        if max_samples is not None and max_samples > 0:
            sequence = sequence[:, time_start : time_start + max_samples]
        else:
            sequence = sequence[:, time_start:]
        if resolution is not None:
            from compression_pipeline.adapters.era5 import center_crop_vthw

            sequence = center_crop_vthw(sequence, resolution)
        timestamps = [self._timestamp(i) for i in range(sequence.shape[1])]
        return sequence, timestamps

    def iter_samples(self, max_samples: int = -1) -> Iterator[CanonicalSample]:
        """Yield [3,H,W] image samples.

        Paper-style turbulence/E3SM data stores physical variables in V.  When
        multiple variables are present, keep all variables from the same
        section/time; image codecs split them into 3-channel groups internally.
        Reduced files with one stored variable can still fall back to neighboring
        section slices so RGB-style image codecs receive three channels.
        """
        path, data, base_metadata = self._data_and_metadata()
        section_start = self._checked_section_index(data.shape[1], self.section_start)
        mode = self._resolved_image_group_mode(data)
        count = 0
        time_start = self._checked_time_index(data.shape[2], self.time_start)
        for t in range(time_start, data.shape[2]):
            if max_samples > 0 and count >= max_samples:
                return
            metadata = dict(base_metadata)
            if mode == "variables":
                section = section_start
                variable_indices = self._variable_indices(data.shape[0])
                chunk = data[variable_indices, section, t].astype(np.float32, copy=False)
                sample_id = f"section{section:03d}_vars000-{variable_indices[-1]:03d}_t{t:04d}"
                metadata.update(
                    {
                        "time_index": int(t),
                        "section_index": int(section),
                        "variable_indices": variable_indices,
                        "image_group_mode": mode,
                    }
                )
            else:
                section_indices = list(range(section_start, min(section_start + 3, data.shape[1])))
                while len(section_indices) < 3:
                    section_indices.append(section_indices[-1])
                chunk = data[0, section_indices, t].astype(np.float32, copy=False)
                first = section_indices[0]
                last_real = min(section_start + 2, data.shape[1] - 1)
                sample_id = f"section{first:03d}-{last_real:03d}_t{t:04d}"
                metadata.update(
                    {
                        "time_index": int(t),
                        "variable_index": 0,
                        "section_indices": [int(i) for i in section_indices],
                        "image_group_mode": mode,
                    }
                )
            metadata["source_path"] = str(path)
            yield CanonicalSample(
                dataset_id=self.dataset_id,
                sample_id=sample_id,
                kind="turb_rot_npz",
                array=chunk,
                layout="channel_height_width",
                metadata=metadata,
            )
            count += 1

    def _resolved_image_group_mode(self, data: np.ndarray) -> str:
        if self.image_group_mode == "variables":
            return "variables"
        if self.image_group_mode == "sections":
            return "sections"
        return "variables" if data.shape[0] >= 3 else "sections"

    def _timestamp(self, index: int) -> str:
        return f"turb_rot_t{index:04d}"

    def _variable_indices(self, variable_count: int) -> list[int]:
        count = variable_count if self.image_channel_count is None else min(self.image_channel_count, variable_count)
        if count <= 0:
            raise ValueError("NPZ data must contain at least one variable")
        return list(range(count))

    @staticmethod
    def _checked_section_index(section_count: int, section_index: int) -> int:
        if section_index < 0 or section_index >= section_count:
            raise ValueError(f"section index {section_index} out of range for S={section_count}")
        return section_index

    @staticmethod
    def _checked_time_index(time_count: int, time_index: int) -> int:
        if time_index < 0 or time_index >= time_count:
            raise ValueError(f"time index {time_index} out of range for T={time_count}")
        return time_index
