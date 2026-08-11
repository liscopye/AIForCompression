"""
Convert lysozyme_chip3 HDF5 diffraction images to CAESAR .npz format [V, S, T, H, W].
Splits the sequence into train and test sets (80/20 by time order).
"""
import os
import sys
import argparse
import numpy as np
import h5py
import hdf5plugin  # needed for crystallography HDF5 filters
from tqdm import tqdm


def load_h5_files(data_dir, max_files=None):
    """Load all HDF5 diffraction frames sorted by filename."""
    files = sorted([
        f for f in os.listdir(data_dir) if f.endswith('.h5')
    ])
    if max_files:
        files = files[:max_files]

    images = []
    for fname in tqdm(files, desc='Loading HDF5 files'):
        fpath = os.path.join(data_dir, fname)
        try:
            with h5py.File(fpath, 'r') as f:
                img = f['entry/data/data'][:]
            images.append(img.astype(np.float32))
        except Exception as e:
            print(f"Warning: skipping {fname}: {e}")

    data = np.stack(images, axis=1)  # [1, N, 1, H, W] broadcast -> will reshape
    return data  # [1, N, 1, H, W] -> needs to become [V=1, S, T, H, W]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/workspace/Data/nfs/chess/raw/2018-1/g3/finke-707-2/20180305/lysozyme_chip3')
    parser.add_argument('--output_dir', type=str,
                        default='/workspace/Data/lysozyme_processed')
    parser.add_argument('--n_frame', type=int, default=16,
                        help='Number of frames per sample (8 for CAESAR-V, 16 for CAESAR-D)')
    parser.add_argument('--train_frac', type=float, default=0.8)
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help='Validation fraction (taken from non-train portion)')
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--crop', type=int, default=1024,
                        help='Crop spatial dims to this size (must be multiple of 256)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    n_frame = args.n_frame

    # Load all images
    files = sorted([f for f in os.listdir(args.data_dir) if f.endswith('.h5')])
    if args.max_files:
        files = files[:args.max_files]
    print(f"Found {len(files)} HDF5 files")

    all_images = []
    skipped = 0
    for fname in tqdm(files, desc='Loading'):
        fpath = os.path.join(args.data_dir, fname)
        try:
            with h5py.File(fpath, 'r') as f:
                img = f['entry/data/data'][:]  # [1, H, W]
            all_images.append(img[0].astype(np.float32))  # -> [H, W]
        except Exception:
            skipped += 1
    print(f"Loaded {len(all_images)} images, skipped {skipped}")

    # Stack as [T_total, H, W]
    all_data = np.stack(all_images, axis=0)  # [T_total, H, W]

    # Center crop to args.crop
    H, W = all_data.shape[1], all_data.shape[2]
    if args.crop:
        h_start = (H - args.crop) // 2
        w_start = (W - args.crop) // 2
        all_data = all_data[:, h_start:h_start+args.crop, w_start:w_start+args.crop]
        print(f"After crop: {all_data.shape}")

    T_total = len(all_data)

    # Make T divisible by n_frame
    usable = (T_total // n_frame) * n_frame
    all_data = all_data[:usable]
    print(f"Using {usable} frames ({usable // n_frame} chunks of {n_frame})")

    # Reshape to [T_chunks, n_frame, H, W] -> [V=1, S=n_chunks, T=n_frame, H, W]
    n_chunks = usable // n_frame
    data = all_data.reshape(n_chunks, n_frame, all_data.shape[1], all_data.shape[2])
    data = data[np.newaxis, :, :, :, :]  # [V=1, S=n_chunks, T=n_frame, H, W]
    print(f"Data shape: {data.shape}")

    # Split train/val/test
    np.random.seed(42)
    indices = np.random.permutation(n_chunks)
    n_train = int(n_chunks * args.train_frac)
    n_val = int(n_chunks * args.val_frac)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_data = data[:, train_idx, :, :, :]
    val_data = data[:, val_idx, :, :, :]
    test_data = data[:, test_idx, :, :, :]

    # Save
    train_path = os.path.join(args.output_dir, f'lysozyme_train_nf{n_frame}.npz')
    val_path = os.path.join(args.output_dir, f'lysozyme_val_nf{n_frame}.npz')
    test_path = os.path.join(args.output_dir, f'lysozyme_test_nf{n_frame}.npz')

    print(f"Train: {train_data.shape}, saving to {train_path}")
    np.savez(train_path, data=train_data)

    print(f"Val:   {val_data.shape}, saving to {val_path}")
    np.savez(val_path, data=val_data)

    print(f"Test:  {test_data.shape}, saving to {test_path}")
    np.savez(test_path, data=test_data)

    # Save metadata
    meta = {
        'dataset': 'lysozyme_chip3',
        'original_shape': data.shape,
        'train_shape': train_data.shape,
        'val_shape': val_data.shape,
        'test_shape': test_data.shape,
        'n_frame': n_frame,
        'dtype': 'float32',
        'crop': args.crop,
        'n_train_chunks': n_train,
        'n_val_chunks': n_val,
        'n_test_chunks': n_chunks - n_train - n_val,
        'data_path': args.data_dir,
    }
    import json
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print("Done!")


if __name__ == '__main__':
    main()
