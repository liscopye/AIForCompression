"""Streaming ERA5 NetCDF dataset for CAESAR fine-tuning."""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import netCDF4
import numpy as np
import torch
from torch.utils.data import Dataset


PRESSURE_VARIABLES = ("z", "q", "u", "v", "t", "r", "w")
PRESSURE_LEVELS = (
    1000.0, 975.0, 950.0, 925.0, 900.0, 875.0, 850.0, 825.0, 800.0,
    775.0, 750.0, 700.0, 650.0, 600.0, 550.0, 500.0, 450.0, 400.0,
    350.0, 300.0, 250.0, 225.0, 200.0, 175.0, 150.0, 125.0, 100.0,
    70.0, 50.0, 30.0, 20.0, 10.0, 7.0, 5.0, 3.0, 2.0, 1.0,
)
SINGLE_VARIABLES = ("v10", "u10", "v100", "u100", "t2m", "tcc", "sp", "tp", "msl")
N_CHANNELS = len(PRESSURE_VARIABLES) * len(PRESSURE_LEVELS) + len(SINGLE_VARIABLES)


@dataclass(frozen=True)
class ERA5FrameRef:
    pressure_path: str
    single_path: str
    local_time_index: int


@dataclass(frozen=True)
class ERA5NpyFrameRef:
    shard_path: str
    local_time_index: int


def load_cra5_channel_stats(stats_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    stats_dir = Path(stats_dir)
    pressure = json.loads((stats_dir / "mean_std.json").read_text(encoding="utf-8"))
    single = json.loads((stats_dir / "mean_std_single.json").read_text(encoding="utf-8"))

    means: list[float] = []
    stds: list[float] = []
    for variable in PRESSURE_VARIABLES:
        means.extend(float(value) for value in pressure["mean"][variable])
        stds.extend(float(value) for value in pressure["std"][variable])
    for variable in SINGLE_VARIABLES:
        means.append(float(single["mean"][variable]))
        stds.append(float(single["std"][variable]))
    result = np.asarray(means, dtype=np.float32), np.asarray(stds, dtype=np.float32)
    if result[0].shape != (N_CHANNELS,) or result[1].shape != (N_CHANNELS,):
        raise ValueError(f"Expected {N_CHANNELS} CRA5 channel statistics, got {result[0].shape}")
    return result


def discover_netcdf_frames(data_dir: str | Path) -> list[ERA5FrameRef]:
    data_dir = Path(data_dir)
    frames: list[ERA5FrameRef] = []
    for pressure_path in sorted(data_dir.glob("*_pressure.nc")):
        single_path = pressure_path.with_name(
            pressure_path.name.replace("_pressure.nc", "_single.nc")
        )
        if not single_path.is_file():
            continue
        with netCDF4.Dataset(pressure_path, "r") as pressure_ds:
            pressure_times = len(pressure_ds.dimensions["valid_time"])
        with netCDF4.Dataset(single_path, "r") as single_ds:
            single_times = len(single_ds.dimensions["valid_time"])
        if pressure_times != single_times:
            raise ValueError(
                f"Time dimension mismatch: {pressure_path}={pressure_times}, "
                f"{single_path}={single_times}"
            )
        frames.extend(
            ERA5FrameRef(str(pressure_path), str(single_path), local_index)
            for local_index in range(pressure_times)
        )
    return frames


def discover_npy_shard_frames(data_dir: str | Path) -> list[ERA5NpyFrameRef]:
    frames: list[ERA5NpyFrameRef] = []
    for shard_path in sorted(Path(data_dir).glob("*_hourly.npy")):
        shard = np.load(shard_path, mmap_mode="r")
        if shard.ndim != 4 or shard.shape[0] != N_CHANNELS:
            raise ValueError(
                f"Expected [{N_CHANNELS},T,H,W] shard, got {shard.shape}: {shard_path}"
            )
        frames.extend(
            ERA5NpyFrameRef(str(shard_path), local_index)
            for local_index in range(int(shard.shape[1]))
        )
        del shard
    return frames


class ERA5NetCDFDataset(Dataset):
    """Read CAESAR patches directly from chronological ERA5 NetCDF frame references."""

    def __init__(
        self,
        frames: Sequence[ERA5FrameRef],
        means: np.ndarray,
        stds: np.ndarray,
        n_frame: int,
        *,
        train: bool,
        train_size: int = 256,
        temporal_stride: int | None = None,
        frame_step: int = 1,
        crop_multiplier: int = 1,
        norm_type: str = "mean_range",
        channels: Sequence[int] | None = None,
        max_open_file_pairs: int = 4,
    ) -> None:
        super().__init__()
        if not frames:
            raise ValueError("At least one ERA5 frame is required.")
        self.frames = tuple(frames)
        self.means = np.asarray(means, dtype=np.float32)
        self.stds = np.asarray(stds, dtype=np.float32)
        if self.means.shape != (N_CHANNELS,) or self.stds.shape != (N_CHANNELS,):
            raise ValueError(f"means/stds must have shape ({N_CHANNELS},)")
        if np.any(self.stds <= 0):
            raise ValueError("All CRA5 channel standard deviations must be positive.")

        self.n_frame = int(n_frame)
        self.train = bool(train)
        self.train_size = int(train_size)
        self.t_stride = int(temporal_stride if temporal_stride is not None else n_frame)
        self.frame_step = int(frame_step)
        self.crop_mult = max(1, int(crop_multiplier))
        self.norm_type = str(norm_type)
        self.channels = tuple(range(N_CHANNELS) if channels is None else channels)
        if not self.channels or min(self.channels) < 0 or max(self.channels) >= N_CHANNELS:
            raise ValueError(f"channels must be non-empty values in [0, {N_CHANNELS})")
        self.C = len(self.channels)
        self.T_full = len(self.frames)
        self.max_open_file_pairs = max(1, int(max_open_file_pairs))

        with netCDF4.Dataset(self.frames[0].pressure_path, "r") as pressure_ds:
            self.H = len(pressure_ds.dimensions["latitude"])
            self.W = len(pressure_ds.dimensions["longitude"])
            available_levels = np.asarray(pressure_ds.variables["pressure_level"][:])
        self.level_indices = tuple(
            int(np.flatnonzero(np.isclose(available_levels, level))[0])
            for level in PRESSURE_LEVELS
        )

        if self.n_frame <= 0 or self.t_stride <= 0 or self.frame_step <= 0:
            raise ValueError("n_frame, temporal_stride, and frame_step must be positive.")
        if self.t_stride > self.n_frame:
            raise ValueError("temporal_stride must be <= n_frame.")
        self.temporal_span = (self.n_frame - 1) * self.frame_step + 1
        if self.T_full < self.temporal_span:
            raise ValueError(
                f"T_full={self.T_full} < temporal_span={self.temporal_span}."
            )
        if self.H < self.train_size or self.W < self.train_size:
            raise ValueError(
                f"Data ({self.H}x{self.W}) smaller than train_size ({self.train_size})."
            )

        self.t_windows = (
            math.ceil((self.T_full - self.temporal_span) / self.t_stride) + 1
        )
        padded_t = (self.t_windows - 1) * self.t_stride + self.temporal_span
        self.pad_t = padded_t - self.T_full
        self.n_h = math.ceil(self.H / self.train_size)
        self.n_w = math.ceil(self.W / self.train_size)
        h_target = self.n_h * self.train_size
        w_target = self.n_w * self.train_size
        dh, dw = h_target - self.H, w_target - self.W
        self.pad_top, self.pad_bottom = dh // 2, dh - dh // 2
        self.pad_left, self.pad_right = dw // 2, dw - dw // 2
        self.spatial_blocks = self.n_h * self.n_w
        self.length = (
            self.C * self.t_windows * self.crop_mult
            if self.train
            else self.C * self.t_windows * self.spatial_blocks
        )
        self._open_pairs: OrderedDict[
            tuple[str, str], tuple[netCDF4.Dataset, netCDF4.Dataset]
        ] = OrderedDict()

    def __len__(self) -> int:
        return self.length

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_open_pairs"] = OrderedDict()
        return state

    def close(self) -> None:
        while self._open_pairs:
            _, pair = self._open_pairs.popitem(last=False)
            pair[0].close()
            pair[1].close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _datasets(self, frame: ERA5FrameRef) -> tuple[netCDF4.Dataset, netCDF4.Dataset]:
        key = (frame.pressure_path, frame.single_path)
        pair = self._open_pairs.pop(key, None)
        if pair is None:
            pair = (
                netCDF4.Dataset(frame.pressure_path, "r"),
                netCDF4.Dataset(frame.single_path, "r"),
            )
        self._open_pairs[key] = pair
        while len(self._open_pairs) > self.max_open_file_pairs:
            _, old_pair = self._open_pairs.popitem(last=False)
            old_pair[0].close()
            old_pair[1].close()
        return pair

    def _decode_index(self, idx: int) -> tuple[int, int, int | None, int | None]:
        if self.train:
            base = idx // self.crop_mult
            c_idx = base // self.t_windows
            return self.channels[c_idx], base % self.t_windows, None, None
        blocks_per_channel = self.t_windows * self.spatial_blocks
        c_idx = idx // blocks_per_channel
        within_channel = idx % blocks_per_channel
        t_idx = within_channel // self.spatial_blocks
        block_idx = within_channel % self.spatial_blocks
        return self.channels[c_idx], t_idx, block_idx // self.n_w, block_idx % self.n_w

    def _read_channel_crop(
        self,
        frame: ERA5FrameRef,
        channel: int,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
    ) -> np.ndarray:
        pressure_ds, single_ds = self._datasets(frame)
        if channel < len(PRESSURE_VARIABLES) * len(PRESSURE_LEVELS):
            variable_index, level_index = divmod(channel, len(PRESSURE_LEVELS))
            values = pressure_ds.variables[PRESSURE_VARIABLES[variable_index]][
                frame.local_time_index,
                self.level_indices[level_index],
                y0:y1,
                x0:x1,
            ]
        else:
            single_index = channel - len(PRESSURE_VARIABLES) * len(PRESSURE_LEVELS)
            variable = SINGLE_VARIABLES[single_index]
            values = single_ds.variables[variable][
                frame.local_time_index, y0:y1, x0:x1
            ]
            if variable == "tp":
                values = values * 1000.0
        values = np.asarray(values, dtype=np.float32)
        return (values - self.means[channel]) / self.stds[channel]

    def _temporal_slice(
        self, channel: int, t0: int, y0: int, y1: int, x0: int, x1: int
    ) -> np.ndarray:
        indices = [
            t0 + index * self.frame_step
            for index in range(self.n_frame)
            if t0 + index * self.frame_step < self.T_full
        ]
        missing = self.n_frame - len(indices)
        if missing:
            indices.extend(range(self.T_full - 1, self.T_full - missing - 1, -1))
        return np.stack(
            [
                self._read_channel_crop(self.frames[index], channel, y0, y1, x0, x1)
                for index in indices
            ],
            axis=0,
        )

    def _train_patch(self, channel: int, t0: int) -> torch.Tensor:
        y0 = torch.randint(0, self.H - self.train_size + 1, (1,)).item()
        x0 = torch.randint(0, self.W - self.train_size + 1, (1,)).item()
        return torch.from_numpy(
            self._temporal_slice(
                channel, t0, y0, y0 + self.train_size, x0, x0 + self.train_size
            )
        )

    def _validation_patch(
        self, channel: int, t0: int, block_h: int, block_w: int
    ) -> torch.Tensor:
        source_y0 = block_h * self.train_size - self.pad_top
        source_x0 = block_w * self.train_size - self.pad_left
        source_y1 = source_y0 + self.train_size
        source_x1 = source_x0 + self.train_size
        y0, y1 = max(0, source_y0), min(self.H, source_y1)
        x0, x1 = max(0, source_x0), min(self.W, source_x1)
        data = self._temporal_slice(channel, t0, y0, y1, x0, x1)
        padding = (
            max(0, -source_x0),
            max(0, source_x1 - self.W),
            max(0, -source_y0),
            max(0, source_y1 - self.H),
        )
        if any(padding):
            left, right, top, bottom = padding
            data = np.pad(
                data,
                ((0, 0), (top, bottom), (left, right)),
                mode="reflect",
            )
        return torch.from_numpy(np.ascontiguousarray(data))

    def _normalize(
        self, data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = torch.finfo(data.dtype).eps
        if self.norm_type == "mean_range":
            offset = data.mean().view(1, 1, 1)
            scale = (data.max() - data.min()).view(1, 1, 1)
        elif self.norm_type == "mean_range_hw":
            offset = data.mean(dim=(-2, -1), keepdim=True)
            scale = (
                data.amax(dim=(-2, -1), keepdim=True)
                - data.amin(dim=(-2, -1), keepdim=True)
            )
        elif self.norm_type == "min_max":
            dmin, dmax = data.min(), data.max()
            offset = ((dmax + dmin) / 2).view(1, 1, 1)
            scale = ((dmax - dmin) / 2).view(1, 1, 1)
        else:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}")
        scale = torch.where(scale.abs() > eps, scale, torch.ones_like(scale))
        return (data - offset) / scale, offset, scale

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        channel, t_idx, block_h, block_w = self._decode_index(idx)
        t0 = t_idx * self.t_stride
        if self.train:
            data = self._train_patch(channel, t0)
        else:
            assert block_h is not None and block_w is not None
            data = self._validation_patch(channel, t0, block_h, block_w)
        data, offset, scale = self._normalize(data)
        return {"input": data.unsqueeze(0), "offset": offset, "scale": scale}


class TemporalWindowBatchSampler:
    """Yield shuffled channel batches that share one temporal window."""

    def __init__(self, dataset: ERA5NetCDFDataset, batch_size: int, seed: int) -> None:
        if not dataset.train:
            raise ValueError("TemporalWindowBatchSampler requires a training dataset.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.generator = torch.Generator().manual_seed(int(seed))
        self.samples_per_window = dataset.C * dataset.crop_mult
        self.batches_per_window = self.samples_per_window // self.batch_size
        if self.batches_per_window == 0:
            raise ValueError(
                f"batch_size={batch_size} exceeds samples/window={self.samples_per_window}"
            )

    def __len__(self) -> int:
        return self.dataset.t_windows * self.batches_per_window

    def __iter__(self):
        window_order = torch.randperm(
            self.dataset.t_windows, generator=self.generator
        ).tolist()
        for t_idx in window_order:
            sample_order = torch.randperm(
                self.samples_per_window, generator=self.generator
            ).tolist()
            usable = self.batches_per_window * self.batch_size
            for start in range(0, usable, self.batch_size):
                batch = []
                for sample_index in sample_order[start:start + self.batch_size]:
                    channel_index, repeat = divmod(
                        sample_index, self.dataset.crop_mult
                    )
                    base = channel_index * self.dataset.t_windows + t_idx
                    batch.append(base * self.dataset.crop_mult + repeat)
                yield batch


class ERA5NpyShardDataset(ERA5NetCDFDataset):
    """CAESAR patch dataset backed by daily normalized ``[268,T,H,W]`` mmap shards."""

    def __init__(
        self,
        frames: Sequence[ERA5NpyFrameRef],
        n_frame: int,
        *,
        train: bool,
        train_size: int = 256,
        temporal_stride: int | None = None,
        frame_step: int = 1,
        crop_multiplier: int = 1,
        norm_type: str = "mean_range",
        channels: Sequence[int] | None = None,
        max_open_shards: int = 4,
    ) -> None:
        Dataset.__init__(self)
        if not frames:
            raise ValueError("At least one ERA5 npy shard frame is required.")
        self.frames = tuple(frames)
        self.means = np.zeros(N_CHANNELS, dtype=np.float32)
        self.stds = np.ones(N_CHANNELS, dtype=np.float32)
        self.n_frame = int(n_frame)
        self.train = bool(train)
        self.train_size = int(train_size)
        self.t_stride = int(temporal_stride if temporal_stride is not None else n_frame)
        self.frame_step = int(frame_step)
        self.crop_mult = max(1, int(crop_multiplier))
        self.norm_type = str(norm_type)
        self.channels = tuple(range(N_CHANNELS) if channels is None else channels)
        if not self.channels or min(self.channels) < 0 or max(self.channels) >= N_CHANNELS:
            raise ValueError(f"channels must be non-empty values in [0, {N_CHANNELS})")
        self.C = len(self.channels)
        self.T_full = len(self.frames)
        self.max_open_file_pairs = max(1, int(max_open_shards))
        probe = np.load(self.frames[0].shard_path, mmap_mode="r")
        self.H, self.W = map(int, probe.shape[-2:])
        del probe
        self.level_indices = tuple(range(len(PRESSURE_LEVELS)))

        if self.n_frame <= 0 or self.t_stride <= 0 or self.frame_step <= 0:
            raise ValueError("n_frame, temporal_stride, and frame_step must be positive.")
        if self.t_stride > self.n_frame:
            raise ValueError("temporal_stride must be <= n_frame.")
        self.temporal_span = (self.n_frame - 1) * self.frame_step + 1
        if self.T_full < self.temporal_span:
            raise ValueError(
                f"T_full={self.T_full} < temporal_span={self.temporal_span}."
            )
        if self.H < self.train_size or self.W < self.train_size:
            raise ValueError(
                f"Data ({self.H}x{self.W}) smaller than train_size ({self.train_size})."
            )
        self.t_windows = (
            math.ceil((self.T_full - self.temporal_span) / self.t_stride) + 1
        )
        padded_t = (self.t_windows - 1) * self.t_stride + self.temporal_span
        self.pad_t = padded_t - self.T_full
        self.n_h = math.ceil(self.H / self.train_size)
        self.n_w = math.ceil(self.W / self.train_size)
        h_target = self.n_h * self.train_size
        w_target = self.n_w * self.train_size
        dh, dw = h_target - self.H, w_target - self.W
        self.pad_top, self.pad_bottom = dh // 2, dh - dh // 2
        self.pad_left, self.pad_right = dw // 2, dw - dw // 2
        self.spatial_blocks = self.n_h * self.n_w
        self.length = (
            self.C * self.t_windows * self.crop_mult
            if self.train
            else self.C * self.t_windows * self.spatial_blocks
        )
        self._open_shards: OrderedDict[str, np.memmap] = OrderedDict()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_open_shards"] = OrderedDict()
        return state

    def close(self) -> None:
        self._open_shards.clear()

    def _shard(self, path: str) -> np.memmap:
        shard = self._open_shards.pop(path, None)
        if shard is None:
            shard = np.load(path, mmap_mode="r")
        self._open_shards[path] = shard
        while len(self._open_shards) > self.max_open_file_pairs:
            self._open_shards.popitem(last=False)
        return shard

    def _read_channel_crop(
        self,
        frame: ERA5NpyFrameRef,
        channel: int,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
    ) -> np.ndarray:
        values = self._shard(frame.shard_path)[
            channel, frame.local_time_index, y0:y1, x0:x1
        ]
        return np.asarray(values, dtype=np.float32)
