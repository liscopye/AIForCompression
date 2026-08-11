#!/usr/bin/env python3
"""Fast eb sweep: uses subset of test sections so PCA fits on GPU.
Runs both original and finetuned CAESAR-V, saves per-eb + sweep summary.
"""
import os, sys, json, time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '/workspace/AIForCompression/models/CAESAR')
from CAESAR.compressor import CAESAR
from dataset import ScientificDataset


EB_VALUES = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 1e0]  # 1e0 = skip PCA
SECTION_LIMIT = 50  # out of 400, 1/8 of full test set
OUTPUT_BASE = "/workspace/AIForCompression/results/eb_sweep"
GPU_ID = 1
CKPT_BASE = "/workspace/AIForCompression/checkpoints/caesar"
TEST_DATA = "/workspace/Data/lysozyme_processed/lysozyme_test_nf8.npz"

os.makedirs(OUTPUT_BASE, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} (physical GPU {GPU_ID}), section_limit={SECTION_LIMIT}, eb values: {EB_VALUES}")

all_results = []

for ckpt_name, ckpt_file in [
    ("CAESAR-V_original", f"{CKPT_BASE}/caesar_v.pt"),
    ("CAESAR-V_finetuned", f"{CKPT_BASE}/caesar_v_tuning_lysozyme.pt"),
]:
    print(f"\n{'='*60}")
    print(f"Loading {ckpt_name}: {ckpt_file}")

    compressor = CAESAR(
        model_path=ckpt_file, use_diffusion=False,
        device=str(device), n_frame=8, interpo_rate=4,
    )

    for eb in EB_VALUES:
        eb_str = f"{eb:.0e}".replace("e-0", "e-").replace("e-", "em")

        data_arg = {
            "data_path": TEST_DATA,
            "variable_idx": [0],
            "section_range": [0, SECTION_LIMIT],
            "frame_range": None,
            "n_frame": 8,
            "train": False,
            "train_size": 256,
            "test_size": (256, 256),
            "inst_norm": True,
            "norm_type": "mean_range",
            "augment_type": {},
            "resolution": None,
            "downsampling": 1,
            "n_overlap": 0,
        }

        dataset = ScientificDataset(data_arg)
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

        print(f"  eb={eb:.1e} ...", end=" ", flush=True)

        t0 = time.time()
        compressed, compressed_size = compressor.compress(loader, eb=eb)
        enc_t = time.time() - t0

        t0 = time.time()
        recons_data = compressor.decompress(compressed)
        dec_t = time.time() - t0

        orig = dataset.input_data()
        recon = dataset.recons_data(recons_data)

        mse = ((orig - recon) ** 2).mean().item()
        rmse = mse ** 0.5
        dr = orig.max() - orig.min()
        nrmse = rmse / dr.item() if dr > 0 else 0
        psnr = (20 * torch.log10(dr / rmse)).item() if rmse > 0 else float("inf")
        bpp = float(compressed_size * 8 / orig.numel())
        cr = float(orig.nbytes / compressed_size) if compressed_size > 0 else float("inf")
        enc_thru = float(orig.nbytes / (1024 * 1024) / enc_t) if enc_t > 0 else 0.0
        dec_thru = float(orig.nbytes / (1024 * 1024) / dec_t) if dec_t > 0 else 0.0

        result = {
            "eb": float(eb), "variant": str(ckpt_name),
            "mse": float(mse), "rmse": float(rmse), "nrmse": float(nrmse), "psnr": float(psnr),
            "bpp": bpp, "compression_ratio": cr,
            "compressed_size_bytes": int(compressed_size),
            "original_size_bytes": int(orig.nbytes),
            "encode_time_avg": float(enc_t), "decode_time_avg": float(dec_t),
            "encode_throughput_mbps": enc_thru,
            "decode_throughput_mbps": dec_thru,
        }
        all_results.append(result)

        print(f"PSNR={psnr:.2f} BPP={bpp:.5f} Enc={enc_t:.1f}s Dec={dec_t:.1f}s")

        # Save per-eb result
        out_dir = os.path.join(OUTPUT_BASE, f"eb_{eb_str}")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{ckpt_name}.json"), "w") as f:
            json.dump(result, f, indent=2)

# Save combined
sweep_path = os.path.join(OUTPUT_BASE, "sweep_results.json")
with open(sweep_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*60}")
print(f"Done! {len(all_results)} results -> {sweep_path}")
for r in all_results:
    print(f"  {r['variant']:30s} eb={r['eb']:.1e}  PSNR={r['psnr']:.2f}  BPP={r['bpp']:.5f}  CR={r['compression_ratio']:.1f}")
