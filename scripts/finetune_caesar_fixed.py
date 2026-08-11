#!/usr/bin/env python3
"""
finetune_caesar_fixed.py

Fine-tune CAESAR-V or CAESAR-D on mmap-backed scientific data stored as
standalone .npy arrays of shape [V, S, T, H, W].

Important fixes compared with the original draft:
  1) Refuses .npz input; use extract_caesar_npz_to_npy.py first for true mmap.
  2) Does not silently fall back from requested CUDA to CPU.
  3) Uses pin_memory/non_blocking only when CUDA is active.
  4) Restores train() after VAE validation.
  5) Logs architecture-aware bpp while preserving configurable RD rate terms.
  6) Reloads/saves the best checkpoint without overwriting it with final weights.
  7) CAESAR-D stage 2 uses quantized VAE latents and inference-aligned latent
     normalization derived from the condition/keyframes.

Official-code alignment:
  - The public CAESAR dataset defaults to "mean_range" instance normalization
    after crop/padding. This script keeps that default.
  - CAESAR-D compresses uniformly selected keyframes. Stage 1 therefore trains
    keyframes only by default; pass --d_stage1_all_frames to disable this.
  - CAESAR-D stage 2 defaults to sequence length 16, interval 3, giving 6
    condition/keyframes and 10 synthesized latent frames.

Examples:
  CAESAR-V smoke test:
    python finetune_caesar_fixed.py --model_type V --device cuda \
      --iterations 2 --batch_size 1 --num_workers 0 --val_interval 1 \
      --save_interval 2 --no_wandb

  CAESAR-V full fine-tuning:
    python finetune_caesar_fixed.py --model_type V --device cuda \
      --iterations 100000 --batch_size 32 --lr 1e-4

  CAESAR-D stage 1:
    python finetune_caesar_fixed.py --model_type D --stage 1 --device cuda \
      --iterations 100000 --batch_size 32 --lr 1e-4

  CAESAR-D stage 2:
    python finetune_caesar_fixed.py --model_type D --stage 2 --device cuda \
      --iterations 200000 --batch_size 64 --lr 1e-4
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def remove_module_prefix(state_dict: dict[str, torch.Tensor]) -> OrderedDict:
    clean = OrderedDict()
    for key, value in state_dict.items():
        clean[key.removeprefix("module.")] = value
    return clean


def safe_torch_load(path: str | Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def atomic_torch_save(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def step_checkpoint_path(path: str | Path, step: int) -> Path:
    path = Path(path)
    return path.with_name(f"{path.stem}_step{step}{path.suffix}")


def resolve_device(device_arg: str) -> torch.device:
    requested = torch.device(device_arg)
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"--device {device_arg!r} was requested, but torch.cuda.is_available() is False.\n"
                f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}.\n"
                "Check `nvidia-smi`, container launch flags such as `--gpus all`, "
                "and whether a CUDA-enabled PyTorch build is installed."
            )
        if requested.index is not None:
            torch.cuda.set_device(requested.index)
    return requested


class MetricLogger:
    def __init__(self, disabled: bool, project: str, name: str, config: dict[str, Any]):
        self._wandb = None
        if disabled:
            return
        try:
            import wandb
            self._wandb = wandb
            self._wandb.init(project=project, name=name, config=config)
        except Exception as exc:
            print(f"wandb init failed ({exc}); continuing without wandb logging.")
            self._wandb = None

    def log(self, metrics: dict[str, float | int]) -> None:
        if self._wandb is not None:
            self._wandb.log(metrics)

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MMapTemporalDataset(Dataset):
    """Read [V, S, T, H, W] .npy data and yield [1, n_frame, size, size] patches.

    Default normalization mirrors the public CAESAR dataset implementation:
      mean_range: (patch - patch.mean()) / (patch.max() - patch.min())

    For train samples larger than `train_size`, only the selected patch is read
    from disk, instead of loading an entire high-resolution temporal block.
    """

    def __init__(
        self,
        npy_path: str | Path,
        n_frame: int,
        train: bool,
        train_size: int = 256,
        temporal_stride: int | None = None,
        section_limit: int = -1,
        norm_type: str = "mean_range",
    ) -> None:
        super().__init__()
        self.npy_path = str(npy_path)
        if Path(self.npy_path).suffix.lower() != ".npy":
            raise ValueError(
                f"MMapTemporalDataset requires a standalone .npy file, got {self.npy_path}. "
                "Run extract_caesar_npz_to_npy.py on the .npz archive first."
            )
        if not Path(self.npy_path).is_file():
            raise FileNotFoundError(self.npy_path)

        probe = np.load(self.npy_path, mmap_mode="r")
        if not isinstance(probe, np.memmap):
            raise RuntimeError(f"Expected mmap-backed array, got {type(probe)}")
        if probe.ndim != 5:
            raise ValueError(f"Expected array shape [V, S, T, H, W], got {probe.shape}")

        self.V, full_s, self.T_full, self.H, self.W = map(int, probe.shape)
        self.dtype = probe.dtype
        del probe

        self.S = min(full_s, section_limit) if section_limit > 0 else full_s
        self.n_frame = int(n_frame)
        self.train = bool(train)
        self.train_size = int(train_size)
        self.temporal_stride = int(temporal_stride or n_frame)
        self.norm_type = norm_type
        self._data: np.memmap | None = None

        if self.n_frame <= 0 or self.train_size <= 0 or self.temporal_stride <= 0:
            raise ValueError("n_frame, train_size, and temporal_stride must be positive.")
        if self.T_full < self.n_frame:
            raise ValueError(f"T_full={self.T_full} is smaller than n_frame={self.n_frame}.")
        if norm_type not in {"mean_range", "mean_range_hw", "min_max"}:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        self.t_samples = (self.T_full - self.n_frame) // self.temporal_stride + 1
        self.total = self.V * self.S * self.t_samples

    def __len__(self) -> int:
        return self.total

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_data"] = None
        return state

    def _array(self) -> np.memmap:
        if self._data is None:
            self._data = np.load(self.npy_path, mmap_mode="r")
        return self._data

    def _spatial_window(self, length: int) -> tuple[int, int]:
        if length <= self.train_size:
            return 0, length
        if self.train:
            start = int(torch.randint(0, length - self.train_size + 1, (1,)).item())
        else:
            start = (length - self.train_size) // 2
        return start, start + self.train_size

    def _pad_to_size(self, data: torch.Tensor) -> torch.Tensor:
        h, w = data.shape[-2:]
        pad_h = max(self.train_size - h, 0)
        pad_w = max(self.train_size - w, 0)
        if pad_h == 0 and pad_w == 0:
            return data

        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        # Reflection requires each pad extent to be smaller than its input dim.
        can_reflect = (
            h > 1 and w > 1 and top < h and bottom < h and left < w and right < w
        )
        mode = "reflect" if can_reflect else "replicate"
        return F.pad(data.unsqueeze(0), (left, right, top, bottom), mode=mode).squeeze(0)

    def _normalize(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = torch.finfo(data.dtype).eps
        if self.norm_type == "mean_range":
            offset = data.mean().view(1, 1, 1)
            scale = (data.max() - data.min()).view(1, 1, 1)
        elif self.norm_type == "mean_range_hw":
            offset = data.mean(dim=(-2, -1), keepdim=True)
            scale = (
                data.amax(dim=(-2, -1), keepdim=True)
                - data.amin(dim=(-2, -1), keepdim=True)
            )
        else:  # min_max -> [-1, 1]
            dmin = data.min()
            dmax = data.max()
            offset = ((dmax + dmin) / 2).view(1, 1, 1)
            scale = ((dmax - dmin) / 2).view(1, 1, 1)

        scale = torch.where(scale.abs() > eps, scale, torch.ones_like(scale))
        return (data - offset) / scale, offset, scale

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        v = idx // (self.S * self.t_samples)
        remain = idx % (self.S * self.t_samples)
        s = remain // self.t_samples
        t0 = (remain % self.t_samples) * self.temporal_stride

        h0, h1 = self._spatial_window(self.H)
        w0, w1 = self._spatial_window(self.W)

        arr = np.array(
            self._array()[v, s, t0:t0 + self.n_frame, h0:h1, w0:w1],
            dtype=np.float32,
            copy=True,
        )
        data = self._pad_to_size(torch.from_numpy(arr))
        data, offset, scale = self._normalize(data)

        return {
            "input": data.unsqueeze(0),  # [C=1, T, H, W]
            "offset": offset,
            "scale": scale,
        }


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_caesar_v(ckpt_path: str | Path, device: torch.device):
    from CAESAR.models import compress_modules3d_mid_SR as cm

    model = cm.CompressorMix(
        dim=16,
        dim_mults=[1, 2, 3, 4],
        reverse_dim_mults=[4, 3, 2],
        hyper_dims_mults=[4, 4, 4],
        channels=1,
        out_channels=1,
        d3=True,
        sr_dim=16,
    )
    state = remove_module_prefix(safe_torch_load(ckpt_path, device))
    model.load_state_dict(state)
    return model.to(device)


def load_caesar_d_vae(ckpt_path: str | Path, device: torch.device):
    from CAESAR.models import keyframe_compressor as kc

    checkpoint = safe_torch_load(ckpt_path, device)
    state = checkpoint["vae"] if isinstance(checkpoint, dict) and "vae" in checkpoint else checkpoint
    state = remove_module_prefix(state)

    model = kc.ResnetCompressor(
        dim=16,
        dim_mults=[1, 2, 3, 4],
        reverse_dim_mults=[4, 3, 2, 1],
        hyper_dims_mults=[4, 4, 4],
        channels=1,
        out_channels=1,
    )
    model.load_state_dict(state)
    return model.to(device)


def load_caesar_d_diffusion(ckpt_path: str | Path, device: torch.device, diffusion_steps: int = 32):
    from CAESAR.models.video_diffusion_interpo import GaussianDiffusion, Unet3D

    checkpoint = safe_torch_load(ckpt_path, device)
    state = (
        checkpoint["diffusion"]
        if isinstance(checkpoint, dict) and "diffusion" in checkpoint
        else checkpoint
    )
    state = remove_module_prefix(state)

    unet = Unet3D(
        dim=64,
        out_dim=64,
        channels=64,
        dim_mults=(1, 2, 4, 8),
        use_bert_text_cond=False,
    )
    diffusion = GaussianDiffusion(
        unet,
        image_size=16,
        num_frames=10,
        channels=64,
        timesteps=diffusion_steps,
        loss_type="l2",
    )
    diffusion.load_state_dict(state)
    return diffusion.to(device)


# ---------------------------------------------------------------------------
# VAE stage helpers
# ---------------------------------------------------------------------------

def select_vae_training_input(
    full_x: torch.Tensor, args: argparse.Namespace
) -> torch.Tensor:
    """For D stage 1, train on the keyframes actually compressed at inference."""
    if args.model_type == "D" and args.stage == 1 and not args.d_stage1_all_frames:
        indices = torch.arange(0, full_x.shape[2], args.interpo_rate, device=full_x.device)
        return full_x.index_select(2, indices)
    return full_x


def vae_rate_metrics(
    result: dict[str, torch.Tensor],
    full_x: torch.Tensor,
    model_x: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (rate_term_for_loss, displayed_bpp, mean_bits_from_model).

    --rate_mode bits preserves the scale of the original draft's lambda_rate.
    Displayed bpp is made codec-aware for D stage 1 keyframe-only training.
    """
    mean_bits = result["frame_bit"].mean()

    if args.model_type == "D" and args.stage == 1 and not args.d_stage1_all_frames:
        batch_size = full_x.shape[0]
        n_keyframes = model_x.shape[2]
        bits_per_sequence = result["frame_bit"].reshape(batch_size, n_keyframes).sum(dim=1)
        displayed_bpp = bits_per_sequence.mean() / (
            full_x.shape[2] * full_x.shape[-2] * full_x.shape[-1]
        )
    else:
        displayed_bpp = result["bpp"].mean()

    rate_term = mean_bits if args.rate_mode == "bits" else displayed_bpp
    return rate_term, displayed_bpp, mean_bits


@torch.no_grad()
def evaluate_vae(
    model: torch.nn.Module,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    non_blocking: bool,
) -> dict[str, float]:
    was_training = model.training
    model.eval()

    sum_loss = 0.0
    sum_mse = 0.0
    sum_bpp = 0.0
    sum_bits = 0.0
    n_samples = 0

    for batch in val_loader:
        full_x = batch["input"].to(device, non_blocking=non_blocking)
        model_x = select_vae_training_input(full_x, args)
        result = model(model_x)
        mse = F.mse_loss(result["output"], model_x)
        rate_term, bpp, mean_bits = vae_rate_metrics(result, full_x, model_x, args)
        rd_loss = mse + args.lambda_rate * rate_term

        batch_n = full_x.shape[0]
        n_samples += batch_n
        sum_loss += rd_loss.item() * batch_n
        sum_mse += mse.item() * batch_n
        sum_bpp += bpp.item() * batch_n
        sum_bits += mean_bits.item() * batch_n

    if was_training:
        model.train()

    if n_samples == 0:
        raise RuntimeError("Validation loader produced zero batches.")

    return {
        "loss": sum_loss / n_samples,
        "mse": sum_mse / n_samples,
        "bpp": sum_bpp / n_samples,
        "mean_bits": sum_bits / n_samples,
    }


def finetune_vae(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    logger: MetricLogger,
    label: str,
) -> torch.nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    non_blocking = device.type == "cuda"

    best_val = float("inf")
    step = 0
    pbar = tqdm(total=args.iterations, desc=f"{label} fine-tune")
    start_time = time.time()

    model.train()
    optimizer.zero_grad(set_to_none=True)

    while step < args.iterations:
        for batch in train_loader:
            if step >= args.iterations:
                break

            full_x = batch["input"].to(device, non_blocking=non_blocking)
            model_x = select_vae_training_input(full_x, args)
            result = model(model_x)

            mse = F.mse_loss(result["output"], model_x)
            rate_term, bpp, mean_bits = vae_rate_metrics(result, full_x, model_x, args)
            rd_loss = mse + args.lambda_rate * rate_term
            (rd_loss / args.gradient_accumulation_steps).backward()

            step += 1
            if step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            pbar.update(1)
            pbar.set_postfix(
                loss=f"{rd_loss.item():.5f}",
                mse=f"{mse.item():.6f}",
                bpp=f"{bpp.item():.5f}",
            )

            if step % args.log_interval == 0:
                logger.log(
                    {
                        "train/loss": rd_loss.item(),
                        "train/mse": mse.item(),
                        "train/bpp": bpp.item(),
                        "train/mean_bits": mean_bits.item(),
                        "train/step": step,
                        "train/lr": args.lr,
                    }
                )

            if step % args.val_interval == 0 or step == args.iterations:
                metrics = evaluate_vae(model, val_loader, args, device, non_blocking)
                logger.log(
                    {
                        "val/loss": metrics["loss"],
                        "val/mse": metrics["mse"],
                        "val/bpp": metrics["bpp"],
                        "val/mean_bits": metrics["mean_bits"],
                        "val/step": step,
                    }
                )

                if metrics["loss"] < best_val:
                    best_val = metrics["loss"]
                    atomic_torch_save(model.state_dict(), args.output_ckpt)
                    pbar.write(
                        f"step {step}: best saved "
                        f"(val_loss={metrics['loss']:.6f}, val_bpp={metrics['bpp']:.6f})"
                    )

            if step % args.save_interval == 0:
                atomic_torch_save(model.state_dict(), step_checkpoint_path(args.output_ckpt, step))

    pbar.close()
    if not Path(args.output_ckpt).exists():
        raise RuntimeError("No best checkpoint was saved; check val_interval and validation data.")
    model.load_state_dict(safe_torch_load(args.output_ckpt, device))

    elapsed = time.time() - start_time
    print(f"{label} done: {step} iters in {elapsed / 60:.1f} min, best_val={best_val:.6f}")
    print(f"Best checkpoint: {args.output_ckpt}")
    return model


# ---------------------------------------------------------------------------
# Diffusion stage helpers
# ---------------------------------------------------------------------------

def condition_aligned_normalize_latent(
    full_quantized_latent: torch.Tensor, interpo_rate: int
) -> torch.Tensor:
    """Normalize latents with statistics obtainable at decoding time.

    Official decompression forms a zero-filled sequence containing decoded
    keyframe latents, derives scale/offset from it, then samples missing frames.
    Here targets are normalized using those same condition-derived statistics.
    """
    cond_idx = torch.arange(
        0, full_quantized_latent.shape[2], interpo_rate, device=full_quantized_latent.device
    )
    condition_latent = torch.zeros_like(full_quantized_latent)
    condition_latent.index_copy_(
        2, cond_idx, full_quantized_latent.index_select(2, cond_idx)
    )

    x_min = condition_latent.amin(dim=(1, 2, 3, 4), keepdim=True)
    x_max = condition_latent.amax(dim=(1, 2, 3, 4), keepdim=True)
    scale = (x_max - x_min + 1e-8) / 2
    offset = x_min + scale
    return (full_quantized_latent - offset) / scale


@torch.no_grad()
def make_diffusion_latent(
    vae: torch.nn.Module, x: torch.Tensor, interpo_rate: int
) -> torch.Tensor:
    # Official keyframe_compressor exposes inference_qlatent(), which applies
    # VAE encoding and hyperprior-guided quantization to every frame.
    latent = vae.inference_qlatent(x)
    return condition_aligned_normalize_latent(latent, interpo_rate)


@torch.no_grad()
def evaluate_diffusion(
    diffusion: torch.nn.Module,
    vae: torch.nn.Module,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    non_blocking: bool,
) -> float:
    was_training = diffusion.training
    diffusion.eval()
    total = 0.0
    count = 0

    for batch in val_loader:
        x = batch["input"].to(device, non_blocking=non_blocking)
        latent = make_diffusion_latent(vae, x, args.interpo_rate)
        loss = diffusion(latent, interpo_rate=args.interpo_rate)
        batch_n = x.shape[0]
        total += loss.item() * batch_n
        count += batch_n

    if was_training:
        diffusion.train()

    if count == 0:
        raise RuntimeError("Validation loader produced zero batches.")
    return total / count


def finetune_diffusion(
    diffusion: torch.nn.Module,
    vae: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    logger: MetricLogger,
) -> torch.nn.Module:
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    diffusion.train()
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.lr)
    non_blocking = device.type == "cuda"

    best_val = float("inf")
    step = 0
    pbar = tqdm(total=args.iterations, desc="CAESAR-D Diffusion fine-tune")
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    while step < args.iterations:
        for batch in train_loader:
            if step >= args.iterations:
                break

            x = batch["input"].to(device, non_blocking=non_blocking)
            with torch.no_grad():
                latent = make_diffusion_latent(vae, x, args.interpo_rate)

            diff_loss = diffusion(latent, interpo_rate=args.interpo_rate)
            (diff_loss / args.gradient_accumulation_steps).backward()

            step += 1
            if step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            pbar.update(1)
            pbar.set_postfix(loss=f"{diff_loss.item():.6f}")

            if step % args.log_interval == 0:
                logger.log({"train/diff_loss": diff_loss.item(), "train/step": step})

            if step % args.val_interval == 0 or step == args.iterations:
                val_loss = evaluate_diffusion(
                    diffusion, vae, val_loader, args, device, non_blocking
                )
                logger.log({"val/diff_loss": val_loss, "val/step": step})
                if val_loss < best_val:
                    best_val = val_loss
                    payload = {
                        "vae": vae.state_dict(),
                        "diffusion": diffusion.state_dict(),
                    }
                    atomic_torch_save(payload, args.output_ckpt)
                    pbar.write(f"step {step}: best full CAESAR-D saved (val={val_loss:.6f})")

            if step % args.save_interval == 0:
                payload = {"vae": vae.state_dict(), "diffusion": diffusion.state_dict()}
                atomic_torch_save(payload, step_checkpoint_path(args.output_ckpt, step))

    pbar.close()
    if not Path(args.output_ckpt).exists():
        raise RuntimeError("No best checkpoint was saved; check validation settings.")
    best_payload = safe_torch_load(args.output_ckpt, device)
    diffusion.load_state_dict(remove_module_prefix(best_payload["diffusion"]))

    elapsed = time.time() - start_time
    print(f"Diffusion done: {step} iters in {elapsed / 60:.1f} min, best_val={best_val:.6f}")
    print(f"Best full CAESAR-D checkpoint: {args.output_ckpt}")
    return diffusion


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune CAESAR-V or CAESAR-D on mmap .npy data.")

    parser.add_argument("--model_type", default="V", choices=["V", "D"])
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="For CAESAR-D: 1=keyframe VAE; 2=latent diffusion.")
    parser.add_argument("--n_frame", type=int, default=None)
    parser.add_argument("--interpo_rate", type=int, default=3)
    parser.add_argument("--diffusion_steps", type=int, default=32)

    parser.add_argument(
        "--data_path",
        default="/workspace/Data/lysozyme_processed/mmap/lysozyme_train_nf16.npy",
    )
    parser.add_argument(
        "--val_data_path",
        default="/workspace/Data/lysozyme_processed/mmap/lysozyme_test_nf16.npy",
    )
    parser.add_argument("--train_size", type=int, default=256)
    parser.add_argument("--temporal_stride", type=int, default=None)
    parser.add_argument("--norm_type", default="mean_range",
                        choices=["mean_range", "mean_range_hw", "min_max"])
    parser.add_argument("--train_sections", type=int, default=-1)
    parser.add_argument("--val_sections", type=int, default=-1)

    parser.add_argument("--caesar_root", default="/workspace/AIForCompression/models/CAESAR")
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--vae_ckpt_path", default=None)
    parser.add_argument("--output_ckpt", default=None)

    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_rate", type=float, default=1e-5)
    parser.add_argument(
        "--rate_mode",
        choices=["bits", "bpp"],
        default="bits",
        help="Loss rate term. 'bits' preserves the original lambda scale; "
             "'bpp' requires retuning lambda_rate.",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--val_interval", type=int, default=2000)
    parser.add_argument("--save_interval", type=int, default=10000)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--d_stage1_all_frames", action="store_true",
                        help="Train CAESAR-D stage-1 VAE on all frames rather than only "
                             "the uniformly selected keyframes.")
    parser.add_argument("--wandb_project", default="caesar-finetune")
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.model_type == "V" and args.stage != 1:
        raise ValueError("CAESAR-V has only stage 1; use --stage 1.")
    if args.n_frame is None:
        args.n_frame = 8 if args.model_type == "V" else 16
    if args.n_frame <= 0 or args.interpo_rate <= 0:
        raise ValueError("n_frame and interpo_rate must be positive.")
    if args.model_type == "V" and args.n_frame % 4 != 0:
        raise ValueError("CAESAR-V temporal length should be divisible by 4.")
    if args.model_type == "D" and args.stage == 2:
        n_condition = len(range(0, args.n_frame, args.interpo_rate))
        n_predicted = args.n_frame - n_condition
        if n_predicted != 10:
            raise ValueError(
                "The public CAESAR-D diffusion checkpoint is configured for 10 synthesized "
                f"frames, but n_frame={args.n_frame}, interpo_rate={args.interpo_rate} "
                f"would synthesize {n_predicted}. Use n_frame=16 and interpo_rate=3."
            )
    if args.iterations <= 0 or args.batch_size <= 0:
        raise ValueError("iterations and batch_size must be positive.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if args.iterations % args.gradient_accumulation_steps != 0:
        raise ValueError("--iterations must be divisible by --gradient_accumulation_steps.")
    if args.val_interval % args.gradient_accumulation_steps != 0:
        raise ValueError("--val_interval must be divisible by --gradient_accumulation_steps.")
    if args.num_workers < 0:
        raise ValueError("--num_workers cannot be negative.")


def default_paths(args: argparse.Namespace) -> None:
    base_ckpt = Path("/workspace/AIForCompression/checkpoints/caesar")
    if args.ckpt_path is None:
        args.ckpt_path = str(base_ckpt / ("caesar_v.pt" if args.model_type == "V" else "caesar_d.pt"))

    if args.output_ckpt is None:
        if args.model_type == "V":
            args.output_ckpt = str(base_ckpt / "caesar_v_tuning_lysozyme.pt")
        elif args.stage == 1:
            args.output_ckpt = str(base_ckpt / "caesar_d_tuning_lysozyme_vae.pt")
        else:
            args.output_ckpt = str(base_ckpt / "caesar_d_tuning_lysozyme.pt")

    if args.model_type == "D" and args.stage == 2 and args.vae_ckpt_path is None:
        args.vae_ckpt_path = str(base_ckpt / "caesar_d_tuning_lysozyme_vae.pt")


def make_loaders(args: argparse.Namespace, use_pin_memory: bool) -> tuple[DataLoader, DataLoader]:
    train_ds = MMapTemporalDataset(
        args.data_path,
        n_frame=args.n_frame,
        train=True,
        train_size=args.train_size,
        temporal_stride=args.temporal_stride,
        section_limit=args.train_sections,
        norm_type=args.norm_type,
    )
    val_ds = MMapTemporalDataset(
        args.val_data_path,
        n_frame=args.n_frame,
        train=False,
        train_size=args.train_size,
        temporal_stride=args.temporal_stride,
        section_limit=args.val_sections,
        norm_type=args.norm_type,
    )
    print(
        f"Train: {len(train_ds)} items | Val: {len(val_ds)} items | "
        f"train_shape=[{train_ds.V},{train_ds.S},{train_ds.T_full},{train_ds.H},{train_ds.W}]"
    )

    common: dict[str, Any] = {
        "num_workers": args.num_workers,
        "pin_memory": use_pin_memory,
        "persistent_workers": args.num_workers > 0,
        "worker_init_fn": worker_init_fn if args.num_workers > 0 else None,
    }
    if args.num_workers > 0:
        common["prefetch_factor"] = args.prefetch_factor

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    if len(train_loader) == 0:
        raise RuntimeError(
            f"No training batches: dataset has {len(train_ds)} items but batch_size={args.batch_size} "
            "with drop_last=True."
        )
    return train_loader, val_loader


def main() -> None:
    args = parse_args()
    validate_args(args)
    default_paths(args)

    sys.path.insert(0, args.caesar_root)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    use_pin_memory = device.type == "cuda"

    print(
        f"CAESAR-{args.model_type} Stage {args.stage} | device={device} | "
        f"n_frame={args.n_frame} | interpo_rate={args.interpo_rate} | norm={args.norm_type}"
    )
    print(f"Input train: {args.data_path}")
    print(f"Input val:   {args.val_data_path}")
    print(f"Checkpoint:  {args.ckpt_path}")
    print(f"Output:      {args.output_ckpt}")

    if args.model_type == "D" and args.stage == 1:
        mode = "all frames" if args.d_stage1_all_frames else "uniform keyframes only"
        print(f"CAESAR-D stage-1 VAE input mode: {mode}")
    if args.rate_mode == "bits":
        print("RD rate term: mean model bits (preserves original lambda_rate scale); bpp is logged separately.")
    else:
        print("RD rate term: codec-aware bpp; retune lambda_rate for this scale.")

    run_name = (
        f"CAESAR-{args.model_type}_S{args.stage}_lysozyme_"
        f"lr{args.lr}_b{args.batch_size}_{args.rate_mode}"
    )
    logger = MetricLogger(args.no_wandb, args.wandb_project, run_name, vars(args))
    train_loader, val_loader = make_loaders(args, use_pin_memory)

    try:
        if args.model_type == "V":
            model = load_caesar_v(args.ckpt_path, device)
            print(f"CAESAR-V params: {sum(p.numel() for p in model.parameters()):,}")
            finetune_vae(model, train_loader, val_loader, args, device, logger, "CAESAR-V")

        elif args.stage == 1:
            vae = load_caesar_d_vae(args.ckpt_path, device)
            print(f"CAESAR-D keyframe VAE params: {sum(p.numel() for p in vae.parameters()):,}")
            finetune_vae(vae, train_loader, val_loader, args, device, logger, "CAESAR-D-VAE")

        else:
            vae = load_caesar_d_vae(args.vae_ckpt_path, device)
            diffusion = load_caesar_d_diffusion(args.ckpt_path, device, args.diffusion_steps)
            n_v = sum(p.numel() for p in vae.parameters())
            n_d = sum(p.numel() for p in diffusion.parameters())
            print(f"CAESAR-D VAE: {n_v:,} | Diffusion: {n_d:,} | Total: {n_v + n_d:,}")
            finetune_diffusion(diffusion, vae, train_loader, val_loader, args, device, logger)
    finally:
        logger.finish()

    print("Done!")


if __name__ == "__main__":
    main()
