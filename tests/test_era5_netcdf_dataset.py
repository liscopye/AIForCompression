from pathlib import Path

import netCDF4
import numpy as np
import pytest

from utils.era5_netcdf_dataset import (
    ERA5NetCDFDataset,
    ERA5NpyShardDataset,
    N_CHANNELS,
    PRESSURE_LEVELS,
    PRESSURE_VARIABLES,
    SINGLE_VARIABLES,
    TemporalWindowBatchSampler,
    discover_netcdf_frames,
    discover_npy_shard_frames,
)
from utils.prepare_era5_hourly_shards import (
    available_pairs,
    convert_pair,
    output_path_for,
)


def write_pair(root: Path) -> None:
    pressure_path = root / "2024-03-01_hourly_pressure.nc"
    single_path = root / "2024-03-01_hourly_single.nc"
    with netCDF4.Dataset(pressure_path, "w") as ds:
        ds.createDimension("valid_time", 2)
        ds.createDimension("pressure_level", len(PRESSURE_LEVELS))
        ds.createDimension("latitude", 5)
        ds.createDimension("longitude", 5)
        ds.createVariable("pressure_level", "f4", ("pressure_level",))[:] = PRESSURE_LEVELS
        for variable_index, name in enumerate(PRESSURE_VARIABLES):
            values = np.empty((2, len(PRESSURE_LEVELS), 5, 5), dtype=np.float32)
            for time_index in range(2):
                for level_index in range(len(PRESSURE_LEVELS)):
                    values[time_index, level_index] = (
                        variable_index * 1000 + time_index * 100 + level_index
                    )
            ds.createVariable(
                name, "f4", ("valid_time", "pressure_level", "latitude", "longitude")
            )[:] = values

    with netCDF4.Dataset(single_path, "w") as ds:
        ds.createDimension("valid_time", 2)
        ds.createDimension("latitude", 5)
        ds.createDimension("longitude", 5)
        for variable_index, name in enumerate(SINGLE_VARIABLES):
            values = np.empty((2, 5, 5), dtype=np.float32)
            for time_index in range(2):
                values[time_index] = variable_index * 10 + time_index
            ds.createVariable(name, "f4", ("valid_time", "latitude", "longitude"))[:] = values


def test_streaming_dataset_channel_order_and_tp_conversion(tmp_path):
    write_pair(tmp_path)
    frames = discover_netcdf_frames(tmp_path)
    assert len(frames) == 2

    dataset = ERA5NetCDFDataset(
        frames,
        means=np.zeros(N_CHANNELS, dtype=np.float32),
        stds=np.ones(N_CHANNELS, dtype=np.float32),
        n_frame=2,
        train=True,
        train_size=4,
    )
    try:
        z_1000 = dataset._temporal_slice(0, 0, 0, 4, 0, 4)
        q_1000 = dataset._temporal_slice(len(PRESSURE_LEVELS), 0, 0, 4, 0, 4)
        w_1 = dataset._temporal_slice(
            len(PRESSURE_VARIABLES) * len(PRESSURE_LEVELS) - 1, 0, 0, 4, 0, 4
        )
        v10 = dataset._temporal_slice(
            len(PRESSURE_VARIABLES) * len(PRESSURE_LEVELS), 0, 0, 4, 0, 4
        )
        tp = dataset._temporal_slice(
            len(PRESSURE_VARIABLES) * len(PRESSURE_LEVELS)
            + SINGLE_VARIABLES.index("tp"),
            0,
            0,
            4,
            0,
            4,
        )

        np.testing.assert_allclose(z_1000[:, 0, 0], [0, 100])
        np.testing.assert_allclose(q_1000[:, 0, 0], [1000, 1100])
        np.testing.assert_allclose(w_1[:, 0, 0], [6036, 6136])
        np.testing.assert_allclose(v10[:, 0, 0], [0, 1])
        np.testing.assert_allclose(tp[:, 0, 0], [70_000, 71_000])

        sample = dataset[0]
        assert sample["input"].shape == (1, 2, 4, 4)
        assert np.isfinite(sample["input"].numpy()).all()
    finally:
        dataset.close()


def test_temporal_batch_sampler_keeps_each_batch_in_one_window(tmp_path):
    write_pair(tmp_path)
    frames = discover_netcdf_frames(tmp_path)
    dataset = ERA5NetCDFDataset(
        frames,
        means=np.zeros(N_CHANNELS, dtype=np.float32),
        stds=np.ones(N_CHANNELS, dtype=np.float32),
        n_frame=1,
        train=True,
        train_size=4,
        crop_multiplier=2,
    )
    sampler = TemporalWindowBatchSampler(dataset, batch_size=32, seed=7)
    try:
        assert len(sampler) == dataset.t_windows * (N_CHANNELS * 2 // 32)
        for batch in sampler:
            windows = {
                (index // dataset.crop_mult) % dataset.t_windows
                for index in batch
            }
            assert len(batch) == 32
            assert len(windows) == 1
    finally:
        dataset.close()


def test_daily_shard_conversion_and_dataset(tmp_path):
    write_pair(tmp_path)
    output_path = tmp_path / "2024-03-01_hourly.npy"
    convert_pair(
        tmp_path / "2024-03-01_hourly_pressure.nc",
        tmp_path / "2024-03-01_hourly_single.nc",
        output_path,
        np.zeros(N_CHANNELS, dtype=np.float32),
        np.ones(N_CHANNELS, dtype=np.float32),
    )
    shard = np.load(output_path, mmap_mode="r")
    assert shard.shape == (N_CHANNELS, 2, 5, 5)
    np.testing.assert_allclose(shard[0, :, 0, 0], [0, 100])
    tp_channel = (
        len(PRESSURE_VARIABLES) * len(PRESSURE_LEVELS)
        + SINGLE_VARIABLES.index("tp")
    )
    np.testing.assert_allclose(shard[tp_channel, :, 0, 0], [70_000, 71_000])
    del shard

    dataset = ERA5NpyShardDataset(
        discover_npy_shard_frames(tmp_path),
        n_frame=2,
        train=True,
        train_size=4,
    )
    try:
        np.testing.assert_allclose(
            dataset._temporal_slice(tp_channel, 0, 0, 4, 0, 4)[:, 0, 0],
            [70_000, 71_000],
        )
    finally:
        dataset.close()

    validation = ERA5NpyShardDataset(
        discover_npy_shard_frames(tmp_path),
        n_frame=2,
        train=False,
        train_size=4,
        channels=[0],
    )
    try:
        edge = validation[len(validation) - 1]
        assert edge["input"].shape == (1, 2, 4, 4)
    finally:
        validation.close()


def test_daily_shard_conversion_rejects_unexpected_shape(tmp_path):
    write_pair(tmp_path)

    with pytest.raises(ValueError, match="Expected NetCDF shape"):
        convert_pair(
            tmp_path / "2024-03-01_hourly_pressure.nc",
            tmp_path / "2024-03-01_hourly_single.nc",
            tmp_path / "unexpected.npy",
            np.zeros(N_CHANNELS, dtype=np.float32),
            np.ones(N_CHANNELS, dtype=np.float32),
            expected_shape=(24, 513, 513),
        )


def test_npy_shard_dataset_supports_daily_frame_step(tmp_path):
    for day in range(2):
        values = np.empty((N_CHANNELS, 24, 5, 5), dtype=np.float32)
        for hour in range(24):
            values[:, hour] = day * 100 + hour
        np.save(tmp_path / f"2024-03-{day + 1:02d}_hourly.npy", values)

    dataset = ERA5NpyShardDataset(
        discover_npy_shard_frames(tmp_path),
        n_frame=2,
        frame_step=24,
        temporal_stride=1,
        train=True,
        train_size=4,
        channels=[0],
    )
    try:
        assert dataset.temporal_span == 25
        assert dataset.t_windows == 24
        np.testing.assert_allclose(
            dataset._temporal_slice(0, 7, 0, 4, 0, 4)[:, 0, 0],
            [7, 107],
        )
    finally:
        dataset.close()


def test_grib_pair_discovery_and_output_name(tmp_path):
    grib = tmp_path / "2024-03-01_hourly_pressure.grib"
    grib.touch()
    single = tmp_path / "2024-03-01_hourly_single.nc"
    single.touch()

    assert output_path_for(grib, tmp_path / "output") == (
        tmp_path / "output" / "2024-03-01_hourly.npy"
    )
    assert available_pairs(tmp_path) == [(grib, single)]

    netcdf = tmp_path / "2024-03-01_hourly_pressure.nc"
    netcdf.touch()
    assert available_pairs(tmp_path) == [(netcdf, single)]
