#!/usr/bin/env python3
"""
Evaluate CAESAR-V or CAESAR-D on ERA5 test data (original vs finetuned).

Test .npy: [C=268, T, H=721, W=1440], CRA5 z-score, mmap-readable.
Each channel is treated independently. Spatial field is padded to 256 multiples
and split into 256×256 blocks.

Usage:
  python scripts/eval_caesar_era5.py --model_type V --ckpt both --max_channels 10
  python scripts/eval_caesar_era5.py --model_type D --ckpt both --max_channels 5 --max_blocks 20
"""

import os, sys, argparse, json, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/workspace/AIForCompression/models/CAESAR")
from CAESAR.compressor import CAESAR

PAD = 256


class ERA5BlockDataset(Dataset):
    """Blocks of size [1, n_frame, 256, 256] from one ERA5 channel.

    The spatial field is reflection-padded to multiples of 256,
    then split into non-overlapping 256×256 blocks × all temporal windows.
    """

    def __init__(self, npy_path: str, channel: int, n_frame: int):
        self._arr = np.load(npy_path, mmap_mode="r")
        self.C, self.T, self.H_raw, self.W_raw = map(int, self._arr.shape)
        self.ch = channel
        self.nf = n_frame
        self.ts = self.T - n_frame + 1

        self.pad_h = (PAD - (self.H_raw % PAD)) % PAD
        self.pad_w = (PAD - (self.W_raw % PAD)) % PAD
        self.H_pad = self.H_raw + self.pad_h
        self.W_pad = self.W_raw + self.pad_w
        self.h_blks = self.H_pad // PAD
        self.w_blks = self.W_pad // PAD
        self.blk_count = self.h_blks * self.w_blks
        self.length = self.ts * self.blk_count

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        t0 = idx // self.blk_count
        blk_id = idx % self.blk_count
        bh, bw = divmod(blk_id, self.w_blks)

        h0, h1 = bh * PAD, min(bh * PAD + PAD, self.H_raw)
        w0, w1 = bw * PAD, min(bw * PAD + PAD, self.W_raw)

        d = torch.from_numpy(
            np.array(self._arr[self.ch, t0:t0 + self.nf, h0:h1, w0:w1], dtype=np.float32)
        ).unsqueeze(0)

        ph, pw = PAD - d.shape[-2], PAD - d.shape[-1]
        if ph or pw:
            d = F.pad(d, (0, pw, 0, ph), mode="reflect")
        return d


def resolve_device(arg: str) -> torch.device:
    d = torch.device(arg)
    if d.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA not available")
    return d


def compute_bpp(compressed_bytes: int, total_elements: int) -> float:
    return compressed_bytes * 8 / total_elements if total_elements > 0 else 0.0


def evaluate(ckpt_path, test_npy, n_frame, use_diffusion, device, eb,
             batch_size, num_workers, max_channels, max_blocks):
    tag = "D" if use_diffusion else "V"
    print(f"\n{'=' * 60}")
    print(f"CAESAR-{tag}  ckpt={ckpt_path}")
    print(f"eb={eb}  n_frame={n_frame}  diffusion={use_diffusion}")
    print(f"{'=' * 60}")

    compressor = CAESAR(model_path=ckpt_path, use_diffusion=use_diffusion,
                        device=str(device), n_frame=n_frame, interpo_rate=3)

    probe = np.load(test_npy, mmap_mode="r")
    total_C = int(probe.shape[0])
    del probe

    channels = list(range(total_C))
    if max_channels:
        channels = channels[:max_channels]

    total_enc = 0.0
    total_dec = 0.0
    total_comp_bytes = 0
    total_elems = 0

    for ci, c in enumerate(channels):
        ds = ERA5BlockDataset(test_npy, channel=c, n_frame=n_frame)
        n_total = len(ds)

        if max_blocks and max_blocks < n_total:
            stride = max(1, n_total // max_blocks)
            indices = list(range(0, n_total, stride))[:max_blocks]
            ds = torch.utils.data.Subset(ds, indices)

        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True,
                           persistent_workers=num_workers > 0)

        print(f"  ch {c+1}/{len(channels)}: {len(ds)} blocks  ", end="", flush=True)

        try:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            compressed, comp_size = compressor.compress(loader, eb=eb)
            if device.type == "cuda":
                torch.cuda.synchronize()
            enc_t = time.perf_counter() - t0

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = compressor.decompress(compressed)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dec_t = time.perf_counter() - t0

        except Exception as e:
            print(f"ERR: {e}")
            continue

        total_enc += enc_t
        total_dec += dec_t
        total_comp_bytes += comp_size
        total_elems += len(ds) * n_frame * PAD * PAD

        print(f"enc={enc_t:.1f}s  dec={dec_t:.1f}s  comp={comp_size/1e6:.1f}MB")

    bpp = compute_bpp(total_comp_bytes, total_elems)
    orig_bytes = total_elems * 4
    cr = orig_bytes / total_comp_bytes if total_comp_bytes > 0 else float("inf")

    return {
        "bpp": bpp,
        "compression_ratio": cr,
        "encode_time_total": total_enc,
        "decode_time_total": total_dec,
        "compressed_size_bytes": int(total_comp_bytes),
        "original_size_bytes": int(orig_bytes),
        "channels_evaluated": len(channels),
        "total_elements": total_elems,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", default="V", choices=["V", "D"])
    p.add_argument("--ckpt", default="both", choices=["original", "era5_tuned", "both"])
    p.add_argument("--test_npy", default="/workspace/Data/ERA5/finetune_processed/era5_test.npy")
    p.add_argument("--ckpt_base", default="/workspace/AIForCompression/checkpoints/caesar")
    p.add_argument("--output_dir", default="/workspace/AIForCompression/results_era5")
    p.add_argument("--eb", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_channels", type=int, default=None)
    p.add_argument("--max_blocks", type=int, default=None)
    args = p.parse_args()

    device = resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    is_v = args.model_type == "V"
    tag = "v" if is_v else "d"
    nf = 8 if is_v else 16
    use_diff = not is_v

    variants = []
    if args.ckpt in ("original", "both"):
        variants.append(("original", f"{args.ckpt_base}/caesar_{tag}.pt"))
    if args.ckpt in ("era5_tuned", "both"):
        variants.append(("era5_tuned", f"{args.ckpt_base}/caesar_{tag}_tuning_era5.pt"))

    for ckpt_name, ckpt_path in variants:
        if not os.path.exists(ckpt_path):
            print(f"SKIP: {ckpt_path}")
            continue

        m = evaluate(ckpt_path, args.test_npy, nf, use_diff, device, args.eb,
                     args.batch_size, args.num_workers,
                     args.max_channels, args.max_blocks)

        variant_name = f"CAESAR-{args.model_type}_{ckpt_name}"
        m["variant"] = variant_name
        m["eb"] = args.eb

        pth = os.path.join(args.output_dir, f"{variant_name}.json")
        with open(pth, "w") as f:
            json.dump(m, f, indent=2)

        print(f"\n{variant_name}:  BPP={m['bpp']:.5f}  CR={m['compression_ratio']:.1f}x  "
              f"enc={m['encode_time_total']:.1f}s  dec={m['decode_time_total']:.1f}s  "
              f"ch={m['channels_evaluated']}")


if __name__ == "__main__":
    main()
