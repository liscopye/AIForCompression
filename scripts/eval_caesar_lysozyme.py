"""
Evaluate original and fine-tuned CAESAR models on lysozyme test set.

Examples:
  python scripts/eval_caesar_lysozyme.py --model_type D --ckpt original
  python scripts/eval_caesar_lysozyme.py --model_type D --ckpt finetuned
  python scripts/eval_caesar_lysozyme.py --model_type D --ckpt both
  python scripts/eval_caesar_lysozyme.py --all
"""

import os
import sys
import argparse
import json
import time
import math
import numpy as np

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/workspace/AIForCompression/models/CAESAR")
from dataset import ScientificDataset
from CAESAR.compressor import CAESAR


def cuda_sync(device):
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()


def compute_global_metrics(original, reconstructed, compressed_size_bytes, encode_time, decode_time):
    original = original.numpy() if isinstance(original, torch.Tensor) else original
    reconstructed = reconstructed.numpy() if isinstance(reconstructed, torch.Tensor) else reconstructed

    orig_flat = original.ravel()
    recon_flat = reconstructed.ravel()

    mse = np.mean((orig_flat - recon_flat) ** 2)
    rmse = np.sqrt(mse)
    data_range = orig_flat.max() - orig_flat.min()

    nrmse = rmse / data_range if data_range > 0 else 0.0
    psnr = 20 * math.log10(data_range / rmse) if data_range > 0 and rmse > 0 else float("inf")

    original_size_bytes = original.nbytes
    compression_ratio = (
        original_size_bytes / compressed_size_bytes
        if compressed_size_bytes > 0
        else float("inf")
    )
    bpp = compressed_size_bytes * 8 / original.size if original.size > 0 else 0.0

    original_size_mb = original_size_bytes / (1024 * 1024)
    encode_throughput = original_size_mb / encode_time if encode_time > 0 else 0.0
    decode_throughput = original_size_mb / decode_time if decode_time > 0 else 0.0

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "nrmse": float(nrmse),
        "psnr": float(psnr),
        "compression_ratio": float(compression_ratio),
        "bpp": float(bpp),
        "original_size_bytes": int(original_size_bytes),
        "compressed_size_bytes": int(compressed_size_bytes),
        "encode_time_total": float(encode_time),
        "decode_time_total": float(decode_time),
        "encode_throughput_mbps": float(encode_throughput),
        "decode_throughput_mbps": float(decode_throughput),
    }


def compute_per_sample_metrics(original, reconstructed):
    original = original.numpy() if isinstance(original, torch.Tensor) else original
    reconstructed = reconstructed.numpy() if isinstance(reconstructed, torch.Tensor) else reconstructed

    # expected shape: [V, S, T, H, W]
    V, S, T, H, W = original.shape

    sample_metrics = []
    for v in range(V):
        for s in range(S):
            orig = original[v, s]
            recon = reconstructed[v, s]

            mse = np.mean((orig - recon) ** 2)
            rmse = np.sqrt(mse)
            data_range = orig.max() - orig.min()

            nrmse = rmse / data_range if data_range > 0 else 0.0
            psnr = 20 * math.log10(data_range / rmse) if data_range > 0 and rmse > 0 else float("inf")

            sample_metrics.append({
                "variable": int(v),
                "sample": int(s),
                "mse": float(mse),
                "rmse": float(rmse),
                "nrmse": float(nrmse),
                "psnr": float(psnr),
            })

    finite_psnr = [m["psnr"] for m in sample_metrics if math.isfinite(m["psnr"])]

    return {
        "per_sample_mean_mse": float(np.mean([m["mse"] for m in sample_metrics])),
        "per_sample_mean_rmse": float(np.mean([m["rmse"] for m in sample_metrics])),
        "per_sample_mean_nrmse": float(np.mean([m["nrmse"] for m in sample_metrics])),
        "per_sample_mean_psnr": float(np.mean(finite_psnr)) if finite_psnr else float("inf"),
        "per_sample": sample_metrics[:10],
    }


def evaluate_model(
    ckpt_path,
    test_data_path,
    n_frame,
    use_diffusion,
    device,
    variable_idx=None,
    interpo_rate=3,
    diffusion_steps=32,
    eb=1e-3,
    batch_size=64,
    num_workers=2,
    max_blocks=None,
):
    print(f"\n{'=' * 70}")
    print(f"Evaluating checkpoint: {ckpt_path}")
    print(f"Test data: {test_data_path}")
    print(f"n_frame={n_frame}, use_diffusion={use_diffusion}, eb={eb}")
    print(f"{'=' * 70}")

    compressor = CAESAR(
        model_path=ckpt_path,
        use_diffusion=use_diffusion,
        device=str(device),
        n_frame=n_frame,
        interpo_rate=interpo_rate,
        diffusion_steps=diffusion_steps,
    )

    data_arg = {
        "data_path": test_data_path,
        "variable_idx": [0] if variable_idx is None else variable_idx,
        "section_range": [0, max_blocks] if max_blocks else None,
        "frame_range": None,
        "n_frame": n_frame,
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
    print(f"Blocked test data shape: {dataset.data_input.shape}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print("Compressing...")
    cuda_sync(device)
    t0 = time.perf_counter()
    compressed, compressed_size = compressor.compress(dataloader, eb=eb)
    cuda_sync(device)
    encode_time = time.perf_counter() - t0

    print("Decompressing...")
    cuda_sync(device)
    t0 = time.perf_counter()
    recons_data = compressor.decompress(compressed)
    cuda_sync(device)
    decode_time = time.perf_counter() - t0

    original_data = dataset.input_data()
    recons_data = dataset.recons_data(recons_data)

    print(f"Original shape:      {original_data.shape}")
    print(f"Reconstructed shape: {recons_data.shape}")
    print(f"Compressed size:     {compressed_size} bytes")

    global_metrics = compute_global_metrics(
        original_data,
        recons_data,
        compressed_size,
        encode_time,
        decode_time,
    )

    per_sample_metrics = compute_per_sample_metrics(original_data, recons_data)

    metrics = {
        **global_metrics,
        **per_sample_metrics,
    }

    return metrics


def build_variants(args):
    ckpt_base = args.ckpt_base

    if args.checkpoint is not None:
        if args.all or args.ckpt != "both":
            raise ValueError("--checkpoint cannot be combined with --all or --ckpt other than 'both'.")
        is_v = args.model_type == "V"
        nf = 8 if is_v else 16
        use_diffusion = not is_v
        default_test_data = f"/workspace/Data/lysozyme_processed/lysozyme_test_nf{nf}.npz"
        test_data = args.test_data if args.test_data is not None else default_test_data
        return [
            (args.model_type, args.variant_name, args.checkpoint,
             test_data, nf, use_diffusion),
        ]

    if args.all:
        return [
            ("V", "original", f"{ckpt_base}/caesar_v.pt",
             "/workspace/Data/lysozyme_processed/lysozyme_test_nf8.npz", 8, False),
            ("V", "finetuned", f"{ckpt_base}/caesar_v_tuning_lysozyme.pt",
             "/workspace/Data/lysozyme_processed/lysozyme_test_nf8.npz", 8, False),
            ("D", "original", f"{ckpt_base}/caesar_d.pt",
             "/workspace/Data/lysozyme_processed/lysozyme_test_nf16.npz", 16, True),
            ("D", "finetuned", f"{ckpt_base}/caesar_d_tuning_lysozyme.pt",
             "/workspace/Data/lysozyme_processed/lysozyme_test_nf16.npz", 16, True),
        ]

    is_v = args.model_type == "V"
    nf = 8 if is_v else 16
    use_diffusion = not is_v

    default_test_data = f"/workspace/Data/lysozyme_processed/lysozyme_test_nf{nf}.npz"
    test_data = args.test_data if args.test_data is not None else default_test_data

    model_tag = "v" if is_v else "d"

    if args.ckpt == "both":
        return [
            (args.model_type, "original",
             f"{ckpt_base}/caesar_{model_tag}.pt",
             test_data, nf, use_diffusion),
            (args.model_type, "finetuned",
             f"{ckpt_base}/caesar_{model_tag}{args.tuned_suffix}.pt",
             test_data, nf, use_diffusion),
        ]

    suffix = "" if args.ckpt == "original" else args.tuned_suffix

    return [
        (args.model_type, args.ckpt,
         f"{ckpt_base}/caesar_{model_tag}{suffix}.pt",
         test_data, nf, use_diffusion),
    ]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", type=str, default="D", choices=["V", "D"])
    parser.add_argument("--ckpt", type=str, default="both",
                        choices=["original", "finetuned", "both"])
    parser.add_argument("--tuned_suffix", type=str, default="_tuning_lysozyme",
                        help="Suffix for finetuned checkpoint: _tuning_lysozyme or _tuning_era5")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Evaluate one explicit checkpoint path.")
    parser.add_argument("--variant_name", type=str, default="custom",
                        help="Output label used with --checkpoint.")

    parser.add_argument("--test_data", type=str, default=None)
    parser.add_argument("--variable_idx", type=int, nargs="+", default=[0],
                        help="Variable indices to evaluate from npz data[V,S,T,H,W].")
    parser.add_argument("--interpo_rate", type=int, default=3,
                        help="CAESAR-D keyframe interval; keyframes are 0, rate, 2*rate, ...")
    parser.add_argument("--diffusion_steps", type=int, default=32,
                        help="Number of CAESAR-D diffusion denoising steps.")
    parser.add_argument("--ckpt_base", type=str,
                        default="/workspace/AIForCompression/checkpoints/caesar")
    parser.add_argument("--output_dir", type=str,
                        default="/workspace/AIForCompression/results_lysozyme")

    parser.add_argument("--eb", type=float, default=1e-3)
    parser.add_argument("--max_blocks", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    variants = build_variants(args)
    all_results = {}

    for model_type, ckpt_name, ckpt_path, test_path, nf, use_diffusion in variants:
        if not os.path.exists(ckpt_path):
            print(f"SKIP: checkpoint not found: {ckpt_path}")
            continue

        if not os.path.exists(test_path):
            print(f"SKIP: test data not found: {test_path}")
            continue

        variant_name = f"CAESAR-{model_type}_{ckpt_name}"

        try:
            metrics = evaluate_model(
                ckpt_path=ckpt_path,
                test_data_path=test_path,
                n_frame=nf,
                use_diffusion=use_diffusion,
                device=device,
                variable_idx=args.variable_idx,
                interpo_rate=args.interpo_rate,
                diffusion_steps=args.diffusion_steps,
                eb=args.eb,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_blocks=args.max_blocks,
            )

            all_results[variant_name] = metrics

            result_path = os.path.join(args.output_dir, f"{variant_name}.json")
            with open(result_path, "w") as f:
                json.dump(metrics, f, indent=2)

            print(f"Saved: {result_path}")

        except Exception as e:
            print(f"ERROR evaluating {variant_name}: {e}")
            import traceback
            traceback.print_exc()

    if len(all_results) > 0:
        print(f"\n{'=' * 90}")
        print("SUMMARY")
        print(f"{'=' * 90}")
        print(
            f"{'Model':<28} "
            f"{'Global NRMSE':>14} "
            f"{'Sample NRMSE':>14} "
            f"{'Global PSNR':>12} "
            f"{'Sample PSNR':>12} "
            f"{'CR':>10} "
            f"{'BPP':>10}"
        )
        print("-" * 90)

        for name, m in all_results.items():
            print(
                f"{name:<28} "
                f"{m['nrmse']:>14.6f} "
                f"{m['per_sample_mean_nrmse']:>14.6f} "
                f"{m['psnr']:>12.2f} "
                f"{m['per_sample_mean_psnr']:>12.2f} "
                f"{m['compression_ratio']:>10.2f} "
                f"{m['bpp']:>10.4f}"
            )

    summary_filename = (
        f"all_results_{args.variant_name}.json"
        if args.checkpoint is not None
        else "all_results.json"
    )
    summary_path = os.path.join(args.output_dir, summary_filename)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to: {summary_path}")


if __name__ == "__main__":
    main()
