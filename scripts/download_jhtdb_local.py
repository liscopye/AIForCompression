#!/usr/bin/env python3
import os
import re
import json
import time
import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getCutout


def sort_data_vars(names):
    def key(name):
        nums = re.findall(r"\d+", name)
        return int(nums[-1]) if nums else name
    return sorted(names, key=key)


def cutout_to_cthw(ds):
    """
    getCutout 返回 xarray.Dataset。
    velocity 每个 time step 的数组通常是 [z, y, x, 3]。
    这里一次只请求一个 z slice，所以转换成 [3, time, y, x]。
    """
    names = sort_data_vars(list(ds.data_vars.keys()))
    arrs = []

    for name in names:
        arr = ds[name].values.astype(np.float32)  # [z, y, x, component]
        if arr.ndim != 4:
            raise RuntimeError(f"Unexpected shape for {name}: {arr.shape}")
        arrs.append(arr)

    data = np.stack(arrs, axis=0)  # [time, z, y, x, component]

    if data.shape[1] != 1:
        raise RuntimeError(f"Expected one z-slice, got z dimension = {data.shape[1]}")

    data = data[:, 0, :, :, :]          # [time, y, x, component]
    data = np.transpose(data, (3, 0, 1, 2))  # [component, time, y, x]
    return data


def make_z_indices(mode, n_regions, z_min, z_max, z_list=None):
    if mode == "spaced":
        return np.linspace(z_min, z_max, n_regions, dtype=int).tolist()

    if mode == "first":
        return list(range(z_min, z_min + n_regions))

    if mode == "custom":
        if not z_list:
            raise ValueError("custom z mode needs --z-list")
        vals = [int(v.strip()) for v in z_list.split(",")]
        if len(vals) != n_regions:
            raise ValueError(f"--z-list length {len(vals)} != n_regions {n_regions}")
        return vals

    raise ValueError(f"Unknown z mode: {mode}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="isotropic1024coarse")
    parser.add_argument("--var", default="velocity")

    parser.add_argument("--out", default="jhtdb_caesar_shape.h5")

    parser.add_argument("--n-regions", type=int, default=64)
    parser.add_argument("--n-times", type=int, default=256)
    parser.add_argument("--time-start", type=int, default=1)

    # 本地请求建议小一点，8 或 16 都可以。8 更稳，16 更快。
    parser.add_argument("--time-block", type=int, default=8)

    # 512 x 512 slice
    parser.add_argument("--xs", type=int, default=1)
    parser.add_argument("--xe", type=int, default=512)
    parser.add_argument("--ys", type=int, default=1)
    parser.add_argument("--ye", type=int, default=512)

    # 论文没有给 exact 64 regions，所以默认从 z 方向均匀取 64 个 2D slices
    parser.add_argument("--z-mode", choices=["spaced", "first", "custom"], default="spaced")
    parser.add_argument("--z-min", type=int, default=1)
    parser.add_argument("--z-max", type=int, default=1024)
    parser.add_argument("--z-list", default=None)

    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.time_start < 1:
        raise ValueError(
            "--time-start must be 1 or greater for JHTDB GetCutout time ranges. "
            "For isotropic1024coarse, the valid inclusive range is [1, 5024]."
        )

    token = os.environ.get("JHTDB_TOKEN")
    if not token:
        raise RuntimeError("Please set JHTDB_TOKEN first.")

    out_path = Path(args.out)
    done_path = Path(str(out_path) + ".done.json")

    height = args.ye - args.ys + 1
    width = args.xe - args.xs + 1

    z_indices = make_z_indices(
        args.z_mode,
        args.n_regions,
        args.z_min,
        args.z_max,
        args.z_list,
    )

    print("Local JHTDB GetCutout download")
    print("dataset:", args.dataset)
    print("variable:", args.var)
    print("output:", out_path)
    print("target shape:", (3, args.n_regions, args.n_times, height, width))
    print("z indices preview:", z_indices[:8], "...", z_indices[-8:])
    print("time range:", args.time_start, "to", args.time_start + args.n_times - 1)

    cube = turb_dataset(
        dataset_title=args.dataset,
        output_path=str(out_path.parent),
        auth_token=token,
    )

    completed = set()
    if done_path.exists():
        completed = set(json.loads(done_path.read_text()))

    with h5py.File(out_path, "a") as f:
        if "velocity" not in f:
            dset = f.create_dataset(
                "velocity",
                shape=(3, args.n_regions, args.n_times, height, width),
                dtype="float32",
                chunks=(3, 1, args.time_block, height, width),
                compression=None,
            )
            dset.attrs["layout"] = "[component, region, time, y, x]"
            dset.attrs["dataset"] = args.dataset
            dset.attrs["variable"] = args.var
            dset.attrs["x_range_1based_inclusive"] = [args.xs, args.xe]
            dset.attrs["y_range_1based_inclusive"] = [args.ys, args.ye]
            dset.attrs["z_indices_1based"] = z_indices
            dset.attrs["time_start"] = args.time_start
            dset.attrs["n_times"] = args.n_times
            dset.attrs["note"] = (
                "Paper-like JHTDB subset. Paper gives shape but not exact region/time indices."
            )
        else:
            dset = f["velocity"]

            existing_time_start = int(dset.attrs.get("time_start", args.time_start))
            if existing_time_start != args.time_start:
                raise RuntimeError(
                    f"Existing output {out_path} has time_start={existing_time_start}, "
                    f"but this run requested time_start={args.time_start}. "
                    "Delete the existing file or choose a new --out path."
                )

        n_blocks_per_region = int(np.ceil(args.n_times / args.time_block))
        total = args.n_regions * n_blocks_per_region

        with tqdm(total=total) as pbar:
            for region_id, z in enumerate(z_indices):
                for local_t0 in range(0, args.n_times, args.time_block):
                    local_t1 = min(local_t0 + args.time_block, args.n_times) - 1

                    t0 = args.time_start + local_t0
                    t1 = args.time_start + local_t1

                    block_id = f"r{region_id}_z{z}_t{t0}_{t1}"

                    if block_id in completed:
                        pbar.update(1)
                        continue

                    xyzt_ranges = np.array(
                        [
                            [args.xs, args.xe],
                            [args.ys, args.ye],
                            [z, z],
                            [t0, t1],
                        ],
                        dtype=np.int32,
                    )

                    xyzt_strides = np.array([1, 1, 1, 1], dtype=np.int32)

                    if args.verbose:
                        print("Request:", block_id, xyzt_ranges.tolist())

                    ds = getCutout(
                        cube,
                        args.var,
                        xyzt_ranges,
                        xyzt_strides,
                        verbose=args.verbose,
                    )

                    block = cutout_to_cthw(ds)  # [3, time_block, 512, 512]
                    expected_t = local_t1 - local_t0 + 1

                    if block.shape != (3, expected_t, height, width):
                        raise RuntimeError(
                            f"Bad block shape {block.shape}, expected {(3, expected_t, height, width)}"
                        )

                    dset[:, region_id, local_t0:local_t1 + 1, :, :] = block
                    f.flush()

                    completed.add(block_id)
                    done_path.write_text(json.dumps(sorted(completed), indent=2))

                    pbar.update(1)
                    time.sleep(args.sleep)

    print("Done:", out_path)


if __name__ == "__main__":
    main()