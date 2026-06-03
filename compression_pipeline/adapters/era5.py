from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import numpy as np
import xarray as xr

from compression_pipeline.canonical import CanonicalSample, DatasetManifest
from compression_pipeline.era5_constants import ERA5_CHANNELS, PRESSURE_LEVELS, VNAMES, era5_channel_names


class ERA5Adapter:
    """Reads paired ERA5 pressure/single NetCDF files into canonical CHW samples.
    Uses two-stage normalization: z-score + per-channel minmax (like CRA5/CompressAI).
    """

    def __init__(self, data_root: str | Path, dataset_id: str = "era5_2024") -> None:
        self.data_root = Path(data_root)
        self.dataset_id = dataset_id

    def _get_day_zscore(self, date_str: str):
        import json
        norm_dir = Path("/data/run01/scxj523/zsh/project/AIForCompression/normalization")
        date_key = date_str.replace("-", "_") + "_000000"
        if not hasattr(self, '_zscore_cache'):
            self._zscore_cache = {}
        if date_str in self._zscore_cache:
            return self._zscore_cache[date_str]
        with open(norm_dir / f"mean_std_{date_key}.json") as fp:
            p = json.load(fp)
        with open(norm_dir / f"mean_std_single_{date_key}.json") as fp:
            s = json.load(fp)
        mean_list, std_list = [], []
        for var in ['z', 'q', 't', 'u', 'v', 'w', 'r']:
            mean_list.extend(p['mean'].get(var, []))
            std_list.extend(p['std'].get(var, []))
        for var in ['t2m', 'u10', 'v10', 'msl', 'tp', 'tcwv', 'tcc', 'u100', 'v100']:
            mean_list.append(s['mean'].get(var, 0))
            std_list.append(s['std'].get(var, 1))
        self._zscore_cache[date_str] = (mean_list, std_list)
        return mean_list, std_list

    def find_pairs(self) -> list[tuple[Path, Path, str]]:
        if not self.data_root.exists():
            raise FileNotFoundError(f"ERA5 data root does not exist: {self.data_root}")
        pairs: list[tuple[Path, Path, str]] = []
        for root, _, files in os.walk(self.data_root):
            for name in sorted(files):
                if not name.endswith("_pressure.nc"):
                    continue
                timestamp = name.replace("_pressure.nc", "")
                pressure = Path(root) / name
                single = Path(root) / f"{timestamp}_single.nc"
                if single.exists():
                    pairs.append((pressure, single, timestamp))
        return sorted(pairs, key=lambda item: item[2])

    def manifest(self) -> DatasetManifest:
        pairs = self.find_pairs()
        return DatasetManifest(
            dataset_id=self.dataset_id,
            dataset_name="ERA5",
            dataset_type="scientific_field",
            source_format="netcdf",
            canonical_layout="channel_height_width",
            sample_count=len(pairs),
            metadata={
                "data_root": str(self.data_root),
                "channels": ERA5_CHANNELS,
                "channel_names": era5_channel_names(),
            },
        )

    def read_pair(self, pressure_file: str | Path, single_file: str | Path, max_channels: int | None = None) -> np.ndarray:
        fields: list[np.ndarray] = []
        pressure_data = xr.open_dataset(pressure_file, engine="netcdf4")
        single_data = xr.open_dataset(single_file, engine="netcdf4")
        try:
            available_levels = list(pressure_data.pressure_level.data)
            level_mapping = [available_levels.index(level) for level in PRESSURE_LEVELS]
            for vname in VNAMES["pressure"]:
                data = pressure_data[vname].data
                for level_idx in level_mapping:
                    fields.append(data[0][level_idx][None])
                    if _has_enough_channels(fields, max_channels):
                        return np.concatenate(fields, axis=0).astype(np.float32)
            for vname in VNAMES["single"]:
                data = single_data[vname].data
                if vname == "tp":
                    data = data * 1000
                fields.append(data)
                if _has_enough_channels(fields, max_channels):
                    return np.concatenate(fields, axis=0).astype(np.float32)
        finally:
            pressure_data.close()
            single_data.close()
        return np.concatenate(fields, axis=0).astype(np.float32)

    def iter_samples(
        self,
        max_samples: int | None = None,
        max_channels: int | None = None,
        resolution: tuple[int, int] | None = None,
    ) -> Iterator[CanonicalSample]:
        pairs = self.find_pairs()
        if max_samples is not None and max_samples > 0:
            pairs = pairs[:max_samples]
        for pressure, single, timestamp in pairs:
            array = limit_channels(self.read_pair(pressure, single, max_channels=max_channels), max_channels)
            array = center_crop_chw(array, resolution)
            meta = {
                "pressure_path": str(pressure),
                "single_path": str(single),
                "timestamp": timestamp,
                "dtype": "float32",
                "channel_names": era5_channel_names()[: array.shape[0]],
                "height": int(array.shape[1]),
                "width": int(array.shape[2]),
                "channels": int(array.shape[0]),
            }
            # Two-stage normalization: add per-day z-score params
            date_str = timestamp[:10]
            z_mean, z_std = self._get_day_zscore(date_str)
            n_ch = array.shape[0]
            meta["zscore_mean"] = z_mean[:n_ch]
            meta["zscore_std"] = z_std[:n_ch]
            yield CanonicalSample(
                dataset_id=self.dataset_id,
                sample_id=timestamp,
                kind="scientific_field",
                array=array,
                layout="channel_height_width",
                metadata=meta,
            )

    def load_sequence(
        self,
        max_samples: int | None = None,
        max_channels: int | None = None,
        resolution: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        samples = list(self.iter_samples(max_samples=max_samples, max_channels=max_channels, resolution=resolution))
        if not samples:
            raise ValueError(f"No ERA5 samples found in {self.data_root}")
        timestamps = [sample.sample_id for sample in samples]
        tchw = np.stack([sample.array for sample in samples], axis=0)
        return np.transpose(tchw, (1, 0, 2, 3)), timestamps


def limit_channels(array: np.ndarray, max_channels: int | None) -> np.ndarray:
    if max_channels is None or max_channels < 0:
        return array
    if max_channels <= 0:
        raise ValueError(f"max_channels must be positive, got {max_channels}")
    if max_channels > array.shape[0]:
        raise ValueError(f"max_channels {max_channels} exceeds available channels {array.shape[0]}")
    return array[:max_channels]


def _has_enough_channels(fields: list[np.ndarray], max_channels: int | None) -> bool:
    return max_channels is not None and max_channels > 0 and len(fields) >= max_channels


def center_crop_chw(array: np.ndarray, resolution: tuple[int, int] | None) -> np.ndarray:
    if resolution is None:
        return array
    target_h, target_w = resolution
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"resolution values must be positive, got {resolution}")
    _, height, width = array.shape
    if target_h > height or target_w > width:
        raise ValueError(f"resolution {resolution} exceeds data size {(height, width)}")
    start_h = (height - target_h) // 2
    start_w = (width - target_w) // 2
    return array[:, start_h:start_h + target_h, start_w:start_w + target_w]


def center_crop_hw(array: np.ndarray, resolution: tuple[int, int] | None) -> np.ndarray:
    if resolution is None:
        return array
    target_h, target_w = resolution
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"resolution values must be positive, got {resolution}")
    height, width = array.shape
    if target_h > height or target_w > width:
        raise ValueError(f"resolution {resolution} exceeds data size {(height, width)}")
    start_h = (height - target_h) // 2
    start_w = (width - target_w) // 2
    return array[start_h:start_h + target_h, start_w:start_w + target_w]


def center_crop_vthw(array: np.ndarray, resolution: tuple[int, int] | None) -> np.ndarray:
    if resolution is None:
        return array
    if array.ndim != 4:
        raise ValueError(f"Expected VTHW array, got shape {array.shape}")
    target_h, target_w = resolution
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"resolution values must be positive, got {resolution}")
    _, _, height, width = array.shape
    if target_h > height or target_w > width:
        raise ValueError(f"resolution {resolution} exceeds data size {(height, width)}")
    start_h = (height - target_h) // 2
    start_w = (width - target_w) // 2
    return array[:, :, start_h:start_h + target_h, start_w:start_w + target_w]
