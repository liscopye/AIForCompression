#!/usr/bin/env python3
"""Sweep eb values for CAESAR-D on lysozyme test set.
Runs both original and finetuned models per eb, saves results per eb dir.

Usage:
  # Full sweep
  python scripts/sweep_caesar_eb_lysozyme.py --model_type D --gpu 1

  # Test with single eb
  python scripts/sweep_caesar_eb_lysozyme.py --model_type D --gpu 1 --test_eb 1e-3

  # Both V and D
  python scripts/sweep_caesar_eb_lysozyme.py --model_type both --gpu 0
"""
import os, sys, json, time, subprocess
from pathlib import Path

EB_VALUES = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
SCRIPT = "/workspace/AIForCompression/scripts/eval_caesar_lysozyme.py"


def run_sweep(model_type: str, gpu: int, test_eb: float | None = None):
    is_v = model_type == "V"
    nf = 8 if is_v else 16
    test_data = f"/workspace/Data/lysozyme_processed/lysozyme_test_nf{nf}.npz"

    output_base = f"/workspace/AIForCompression/results/eb_sweep_{model_type}"
    os.makedirs(output_base, exist_ok=True)

    eb_values = [test_eb] if test_eb is not None else EB_VALUES
    all_metrics = []

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    for eb in eb_values:
        eb_str = f"{eb:.0e}".replace("e-0", "e-").replace("e-", "em")
        out_dir = os.path.join(output_base, f"eb_{eb_str}")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"CAESAR-{model_type}  eb={eb}  ->  {out_dir}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, SCRIPT,
            "--model_type", model_type,
            "--ckpt", "both",
            "--device", "cuda:0",
            "--output_dir", out_dir,
            "--eb", str(eb),
            "--batch_size", "8",
            "--max_blocks", "10",
        ]

        t0 = time.time()
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        elapsed = time.time() - t0

        print(result.stdout)
        if result.returncode != 0:
            print(f"ERROR at eb={eb}:")
            print(result.stderr)
            continue

        print(f"eb={eb} completed in {elapsed:.1f}s")

        for variant in [f"CAESAR-{model_type}_original", f"CAESAR-{model_type}_finetuned"]:
            json_path = os.path.join(out_dir, f"{variant}.json")
            if os.path.exists(json_path):
                with open(json_path) as f:
                    m = json.load(f)
                m["eb"] = eb
                m["variant"] = variant
                all_metrics.append(m)

    sweep_path = os.path.join(output_base, "sweep_results.json")
    with open(sweep_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSweep done! {len(all_metrics)} results saved to {sweep_path}")
    return sweep_path


def main():
    import argparse
    p = argparse.ArgumentParser(description="Sweep EB values for CAESAR on lysozyme")
    p.add_argument("--model_type", default="D", choices=["V", "D", "both"])
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--test_eb", type=float, default=None,
                   help="Run only this eb value (for testing)")
    args = p.parse_args()

    if args.model_type == "both":
        for mt in ["V", "D"]:
            run_sweep(mt, args.gpu, args.test_eb)
    else:
        run_sweep(args.model_type, args.gpu, args.test_eb)


if __name__ == "__main__":
    main()
