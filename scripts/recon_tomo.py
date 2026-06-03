#!/usr/bin/env python3
"""Stream tomopy reconstruction of tomo_00001.h5, chunk by chunk."""

import h5py, tomopy, numpy as np, os, sys

data_path = '/data/run01/scxj523/zsh/project/Data/tomo/tomo_00001.h5'
out_path = '/data/run01/scxj523/zsh/project/Data/tomo/tomo_00001_rec/recon.h5'
CHUNK = 32
ROT_CENTER = 1024.0

os.makedirs(os.path.dirname(out_path), exist_ok=True)

with h5py.File(data_path, 'r') as f:
    n_angles, n_rows, n_cols = f['exchange/data'].shape
    theta = f['exchange/theta'][:].astype(np.float32)

print(f"Total: {n_angles} angles, {n_rows} rows, {n_cols} cols")

with h5py.File(out_path, 'w') as fout:
    dset = fout.create_dataset('data', shape=(n_rows, n_cols, n_cols), dtype=np.float32,
                                chunks=(CHUNK, n_cols, n_cols), compression='gzip', compression_opts=4)

    n_chunks = (n_rows + CHUNK - 1) // CHUNK
    for ci in range(n_chunks):
        start = ci * CHUNK
        end = min(start + CHUNK, n_rows)
        print(f"[{start}:{end}] ({ci+1}/{n_chunks}) reading...", end=' ', flush=True)

        with h5py.File(data_path, 'r') as fin:
            sub_proj = fin['exchange/data'][:, start:end, :].astype(np.float32)
            sub_dark = fin['exchange/data_dark'][:, start:end, :].astype(np.float32)
            sub_white = fin['exchange/data_white'][:, start:end, :].astype(np.float32)

        print("norm...", end=' ', flush=True)
        sub_proj = tomopy.normalize(sub_proj, sub_white, sub_dark, cutoff=1.0)
        sub_proj = tomopy.minus_log(sub_proj)
        sub_proj = tomopy.misc.corr.remove_nan(sub_proj, val=0.0)

        print("recon...", end=' ', flush=True)
        recon_chunk = tomopy.recon(sub_proj, theta, center=ROT_CENTER, algorithm='gridrec')

        print("save...", end=' ', flush=True)
        dset[start:end] = recon_chunk
        print(f"done [{recon_chunk.min():.4f}, {recon_chunk.max():.4f}]")
        sys.stdout.flush()

    fout.create_dataset('theta', data=theta)
    fout.attrs['rotation_axis'] = ROT_CENTER
    fout.attrs['algorithm'] = 'gridrec'

print(f"\nDone: {out_path}  [{n_rows}x{n_cols}x{n_cols}]")
