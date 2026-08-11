#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pair a fine-tuned CAESAR-D stage-1 VAE with a released diffusion checkpoint."
    )
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    vae = load(args.vae)
    base = load(args.base)
    if not isinstance(base, dict) or "diffusion" not in base:
        raise ValueError(f"{args.base} is not a full CAESAR-D checkpoint")
    if isinstance(vae, dict) and "vae" in vae:
        vae = vae["vae"]
    if not isinstance(vae, dict):
        raise ValueError(f"{args.vae} does not contain a VAE state dict")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vae": vae,
            "diffusion": base["diffusion"],
            "provenance": {
                "stage1_vae": str(args.vae.resolve()),
                "paired_diffusion": str(args.base.resolve()),
            },
        },
        args.output,
    )
    print(args.output)


if __name__ == "__main__":
    main()
