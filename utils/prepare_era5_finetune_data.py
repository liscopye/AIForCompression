#!/usr/bin/env python3
"""
Prepare ERA5 NC files for CAESAR fine-tuning.

Reads all pressure.nc + single.nc pairs from --input_dir, concatenates into
268 channels, z-score normalizes per channel using CRA5 statistics, then saves
[C, T, H, W] float32 .npy files suitable for mmap access.

By default this now uses a chronological split: the first N timestamps become
training data and the following timestamps become validation data. The previous
longitude split is still available with ``--split_mode spatial``.

Usage:
  python utils/prepare_era5_finetune_data.py \
    --input_dir /workspace/Data/ERA5/finetune \
    --output_dir /workspace/Data/ERA5/finetune_processed
"""

import argparse
import json
import os
import sys
import numpy as np
import xarray as xr
from pathlib import Path

# ─── ERA5 variable definitions (must match all test_era5.py) ───
VNAMES = dict(
    pressure=['z', 'q', 'u', 'v', 't', 'r', 'w'],
    single=['v10', 'u10', 'v100', 'u100', 't2m', 'tcc', 'sp', 'tp', 'msl'],
)

PRESSURE_LEVELS = [
    1000., 975., 950., 925., 900., 875., 850., 825., 800.,
    775., 750., 700., 650., 600., 550., 500., 450., 400.,
    350., 300., 250., 225., 200., 175., 150., 125., 100.,
    70., 50., 30., 20., 10., 7., 5., 3., 2., 1.,
]

N_CHANNELS = len(VNAMES['pressure']) * len(PRESSURE_LEVELS) + len(VNAMES['single'])  # 268
H, W = 721, 1440


def load_mean_std(mean_std_dir: str):
    with open(os.path.join(mean_std_dir, 'mean_std.json'), 'r') as f:
        mean_std = json.load(f)
    with open(os.path.join(mean_std_dir, 'mean_std_single.json'), 'r') as f:
        mean_std_single = json.load(f)
    return mean_std, mean_std_single


def build_channel_stats(mean_std: dict, mean_std_single: dict):
    """Return (means, stds) as float32 arrays of length 268."""
    level_mapping = [PRESSURE_LEVELS.index(v) for v in PRESSURE_LEVELS]
    mean_list, std_list = [], []
    for vname in VNAMES['pressure']:
        mean_list += [mean_std['mean'][vname][idx] for idx in level_mapping]
        std_list += [mean_std['std'][vname][idx] for idx in level_mapping]
    for vname in VNAMES['single']:
        mean_list.append(mean_std_single['mean'][vname])
        std_list.append(mean_std_single['std'][vname])
    return (
        np.array(mean_list, dtype=np.float32),
        np.array(std_list, dtype=np.float32),
    )


def read_one_step(pressure_file: str, single_file: str) -> np.ndarray:
    """Read one time step, return (268, 721, 1440) float32 z-score normalized."""
    pressure_data = xr.open_dataset(pressure_file, engine='netcdf4')
    single_data = xr.open_dataset(single_file, engine='netcdf4')

    pha_levels = list(pressure_data.pressure_level.data)
    level_mapping = [pha_levels.index(v) for v in PRESSURE_LEVELS if v in pha_levels]

    channels = []
    for vname in VNAMES['pressure']:
        D = pressure_data[vname].data
        for level in level_mapping:
            channels.append(D[0][level].astype(np.float32))

    for vname in VNAMES['single']:
        D = single_data[vname].data.astype(np.float32)
        if vname == 'tp':
            D = D * 1000
        channels.append(D[0])

    pressure_data.close()
    single_data.close()

    return np.stack(channels, axis=0)  # (268, 721, 1440)


def list_valid_pairs(input_dir: str):
    nc_files = sorted(Path(input_dir).glob('*_pressure.nc'))
    valid = []
    for pf in nc_files:
        sf = Path(str(pf).replace('_pressure.nc', '_single.nc'))
        if sf.exists():
            timestamp = pf.name.replace('_pressure.nc', '')
            valid.append((str(pf), str(sf), timestamp))
    return valid


def process_to_mmap(
    input_dir: str,
    output_dir: str,
    means: np.ndarray,
    stds: np.ndarray,
    split_mode: str,
    val_lon_split: int,
    train_days: int,
    val_days: int,
) -> None:
    """Process NC files incrementally, writing directly to mmap-backed .npy files.

    One time step ~1.1 GB in RAM at a time instead of loading all 30 days.
    """
    valid = list_valid_pairs(input_dir)

    if not valid:
        raise FileNotFoundError(f"No valid pressure+single pairs found in {input_dir}")

    T = len(valid)
    train_path = os.path.join(output_dir, 'era5_train.npy')
    val_path = os.path.join(output_dir, 'era5_val.npy')
    meta_path = os.path.join(output_dir, 'meta.json')

    means_ = means.reshape(N_CHANNELS, 1, 1)
    stds_ = np.maximum(stds.reshape(N_CHANNELS, 1, 1), 1e-8)

    if split_mode == 'spatial':
        W_train = val_lon_split
        W_val = W - val_lon_split
        train_items = valid
        val_items = valid
        train_mmap = np.lib.format.open_memmap(
            train_path, mode='w+', dtype=np.float32,
            shape=(N_CHANNELS, T, H, W_train),
        )
        val_mmap = np.lib.format.open_memmap(
            val_path, mode='w+', dtype=np.float32,
            shape=(N_CHANNELS, T, H, W_val),
        )
        print(f"Processing {T} time steps with spatial split → train:{train_mmap.shape} val:{val_mmap.shape}")
        for t, (pf, sf, _) in enumerate(valid):
            print(f"  [{t+1}/{T}] {Path(pf).name} ...", end=' ', flush=True)
            data = (read_one_step(pf, sf) - means_) / stds_
            train_mmap[:, t] = data[:, :, :W_train]
            val_mmap[:, t] = data[:, :, W_train:]
            print("done")
    elif split_mode == 'time':
        total = len(valid)
        train_days = train_days if train_days > 0 else int(total * 0.8)
        if train_days <= 0 or train_days >= total:
            raise ValueError(f"train_days must be in [1, {total - 1}], got {train_days}")
        val_days = val_days if val_days > 0 else total - train_days
        if train_days + val_days > total:
            raise ValueError(
                f"train_days + val_days exceeds available timestamps: "
                f"{train_days} + {val_days} > {total}"
            )
        train_items = valid[:train_days]
        val_items = valid[train_days:train_days + val_days]
        train_mmap = np.lib.format.open_memmap(
            train_path, mode='w+', dtype=np.float32,
            shape=(N_CHANNELS, len(train_items), H, W),
        )
        val_mmap = np.lib.format.open_memmap(
            val_path, mode='w+', dtype=np.float32,
            shape=(N_CHANNELS, len(val_items), H, W),
        )
        print(
            f"Processing {total} time steps with chronological split → "
            f"train:{train_mmap.shape} val:{val_mmap.shape}"
        )
        for out_t, (pf, sf, _) in enumerate(train_items):
            print(f"  [train {out_t+1}/{len(train_items)}] {Path(pf).name} ...", end=' ', flush=True)
            train_mmap[:, out_t] = (read_one_step(pf, sf) - means_) / stds_
            print("done")
        for out_t, (pf, sf, _) in enumerate(val_items):
            print(f"  [val {out_t+1}/{len(val_items)}] {Path(pf).name} ...", end=' ', flush=True)
            val_mmap[:, out_t] = (read_one_step(pf, sf) - means_) / stds_
            print("done")
    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    train_mmap.flush()
    val_mmap.flush()

    meta = {
        'C': N_CHANNELS, 'T': T, 'H': H, 'W': W,
        'split_mode': split_mode,
        'val_lon_split': val_lon_split,
        'train_shape': list(train_mmap.shape),
        'val_shape': list(val_mmap.shape),
        'train_timestamps': [item[2] for item in train_items],
        'val_timestamps': [item[2] for item in val_items],
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f)

    print(f"Train: {train_mmap.shape} ({train_mmap.nbytes / 1e9:.2f} GB)")
    print(f"Val:   {val_mmap.shape} ({val_mmap.nbytes / 1e9:.2f} GB)")
    print(f"Meta:  {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare ERA5 data for CAESAR fine-tuning")
    parser.add_argument('--input_dir', default='/workspace/Data/ERA5/finetune')
    parser.add_argument('--output_dir', default='/workspace/Data/ERA5/finetune_processed')
    parser.add_argument('--mean_std_dir',
                        default='/workspace/AIForCompression/models/CRA5/cra5/dataset')
    parser.add_argument('--split_mode', default='time', choices=['time', 'spatial'])
    parser.add_argument('--train_days', type=int, default=49,
                        help='Number of earliest timestamps for train when --split_mode=time.')
    parser.add_argument('--val_days', type=int, default=-1,
                        help='Number of following timestamps for val; default uses all remaining timestamps.')
    parser.add_argument('--val_lon_split', type=int, default=1152,
                        help='Longitude index for --split_mode=spatial (default 1152/1440 = 80/20).')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    mean_std, mean_std_single = load_mean_std(args.mean_std_dir)
    means, stds = build_channel_stats(mean_std, mean_std_single)

    process_to_mmap(
        args.input_dir,
        args.output_dir,
        means,
        stds,
        args.split_mode,
        args.val_lon_split,
        args.train_days,
        args.val_days,
    )
    print("Done!")


if __name__ == '__main__':
    main()
