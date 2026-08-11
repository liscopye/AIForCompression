#!/usr/bin/env python3
"""
Fine-tune CAESAR-V or CAESAR-D on preprocessed ERA5 mmap ``.npy`` data.

Expected data layout
--------------------
  era5_train.npy: [C, T, H, W]
  era5_val.npy:   [C, T, H, W]

``C`` may represent flattened physical-variable / vertical-level channels; each
sample is still passed to CAESAR as a single-channel spatiotemporal patch.

This version intentionally aligns the data/training control flow with the
public CAESAR implementation:
  * default temporal windows do not overlap (stride = n_frame);
  * a final incomplete temporal window is reflection-padded from tail frames;
  * validation covers the complete spatial field with reflected 256x256 blocks;
  * ``--iterations``, ``--log_interval``, ``--val_interval`` and
    ``--save_interval`` count optimizer updates, not micro-batches.

Examples
--------
CAESAR-V (paper/official-code style batch size 32):
  CUDA_VISIBLE_DEVICES=0 python scripts/finetune_caesar_era5_fixed.py \
    --model_type V --device cuda:0 --iterations 100000 --batch_size 32 \
    --lr 1e-4

CAESAR-D stage 1, keyframe VAE:
  CUDA_VISIBLE_DEVICES=0 python scripts/finetune_caesar_era5_fixed.py \
    --model_type D --stage 1 --device cuda:0 --iterations 100000 \
    --batch_size 32 --lr 1e-4

CAESAR-D stage 2, if batch size 64 does not fit GPU memory:
  CUDA_VISIBLE_DEVICES=0 python scripts/finetune_caesar_era5_fixed.py \
    --model_type D --stage 2 --device cuda:0 \
    --vae_ckpt_path /workspace/AIForCompression/checkpoints/caesar/caesar_d_tuning_era5_vae.pt \
    --iterations 200000 --batch_size 32 --gradient_accumulation_steps 2 \
    --lr 1e-4

With the stage-2 command above, each optimizer update uses an effective batch
size of 64 while ``--iterations 200000`` remains 200,000 parameter updates.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
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
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def step_ckpt_path(path: str | Path, update_step: int) -> Path:
    path = Path(path)
    return path.with_name(f"{path.stem}_update{update_step}{path.suffix}")


def resolve_device(device_arg: str) -> torch.device:
    requested = torch.device(device_arg)
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"--device {device_arg!r} requested but torch.cuda.is_available() is False. "
                f"torch={torch.__version__}, cuda={torch.version.cuda}."
            )
        if requested.index is not None:
            if requested.index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Requested {device_arg!r}, but only {torch.cuda.device_count()} CUDA device(s) are visible."
                )
            torch.cuda.set_device(requested.index)
    return requested


def infinite_batches(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        yield from loader


def percent_change(current: float, initial: float) -> float:
    if initial == 0.0:
        return 0.0 if current == 0.0 else float("nan")
    return (current / initial - 1.0) * 100.0


def count_predicted_frames(n_frame: int, interpo_rate: int) -> int:
    if n_frame <= 0 or interpo_rate <= 0:
        raise ValueError("n_frame and interpo_rate must be positive.")
    return n_frame - len(range(0, n_frame, interpo_rate))


def resolve_channel_indices(
    start: int,
    end: int | None,
    total_channels: int,
) -> tuple[int, ...]:
    """Resolve an inclusive/exclusive physical-channel training range."""
    if total_channels <= 0:
        raise ValueError("total_channels must be positive.")
    if start < 0:
        raise ValueError("--train_channel_start cannot be negative.")
    resolved_end = total_channels if end is None else end
    if resolved_end <= start:
        raise ValueError(
            "--train_channel_end must be greater than --train_channel_start."
        )
    if resolved_end > total_channels:
        raise ValueError(f"--train_channel_end must be <= {total_channels}.")
    return tuple(range(start, resolved_end))


class MetricLogger:
    def __init__(
        self,
        disabled: bool,
        project: str,
        name: str,
        config: dict[str, Any],
        *,
        group: str | None = None,
        tags: list[str] | None = None,
        required: bool = False,
    ):
        self._wandb = None
        if disabled:
            return
        try:
            import wandb

            self._wandb = wandb
            self._wandb.init(project=project, name=name, group=group, tags=tags, config=config)
        except Exception as exc:  # logging must not abort training
            if required:
                raise RuntimeError(f"W&B initialization is required but failed: {exc}") from exc
            print(f"wandb init failed ({exc}); continuing without logging.")

    def log(self, metrics: dict[str, float | int]) -> None:
        if self._wandb is not None:
            self._wandb.log(metrics)

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()


# ---------------------------------------------------------------------------
# ERA5 dataset
# ---------------------------------------------------------------------------


class ERA5MmapDataset(Dataset):
    """Read mmap ``[C, T, H, W]`` ERA5 arrays and yield ``[1, T, S, S]`` patches.

    Training follows the official CAESAR sampling spirit: each channel and
    temporal window maps to a random spatial crop. ``crop_multiplier`` repeats
    every (channel, temporal-window) pair with independently sampled crops.

    Validation is deterministic and covers the full spatial field: the field is
    reflection-padded to multiples of ``train_size`` and enumerated as
    non-overlapping blocks, matching the public dataset's ``block_hw`` behavior.
    """

    def __init__(
        self,
        npy_path: str | Path,
        n_frame: int,
        train: bool = True,
        train_size: int = 256,
        temporal_stride: int | None = None,
        crop_multiplier: int = 1,
        norm_type: str = "mean_range",
    ) -> None:
        super().__init__()
        self.npy_path = str(npy_path)
        if not Path(self.npy_path).is_file():
            raise FileNotFoundError(self.npy_path)

        probe = np.load(self.npy_path, mmap_mode="r")
        if not isinstance(probe, np.memmap):
            raise ValueError(f"Expected mmap-able .npy input, got {type(probe).__name__}: {self.npy_path}")
        if probe.ndim != 4:
            raise ValueError(f"Expected [C, T, H, W] array, got shape {probe.shape}")
        self.C, self.T_full, self.H, self.W = map(int, probe.shape)
        del probe

        self.n_frame = int(n_frame)
        self.train = bool(train)
        self.train_size = int(train_size)
        self.t_stride = int(temporal_stride if temporal_stride is not None else n_frame)
        self.crop_mult = max(1, int(crop_multiplier))
        self.norm_type = str(norm_type)

        if self.n_frame <= 0 or self.t_stride <= 0:
            raise ValueError("n_frame and temporal_stride must be positive.")
        if self.t_stride > self.n_frame:
            raise ValueError("temporal_stride must be <= n_frame to avoid unrepresented time gaps.")
        if self.H < self.train_size or self.W < self.train_size:
            raise ValueError(f"Data ({self.H}x{self.W}) smaller than train_size ({self.train_size}).")
        if self.T_full < self.n_frame:
            raise ValueError(f"T_full={self.T_full} < n_frame={self.n_frame}.")

        # Same temporal-window rule as official ScientificDataset: include the
        # tail window and reflection-pad it when the final window is incomplete.
        self.t_windows = math.ceil((self.T_full - self.n_frame) / self.t_stride) + 1
        padded_t = (self.t_windows - 1) * self.t_stride + self.n_frame
        self.pad_t = padded_t - self.T_full

        # Deterministic full-spatial-field validation block layout.
        self.n_h = math.ceil(self.H / self.train_size)
        self.n_w = math.ceil(self.W / self.train_size)
        h_target = self.n_h * self.train_size
        w_target = self.n_w * self.train_size
        dh, dw = h_target - self.H, w_target - self.W
        self.pad_top, self.pad_bottom = dh // 2, dh - dh // 2
        self.pad_left, self.pad_right = dw // 2, dw - dw // 2
        self.spatial_blocks = self.n_h * self.n_w

        if self.train:
            self.length = self.C * self.t_windows * self.crop_mult
        else:
            self.length = self.C * self.t_windows * self.spatial_blocks

        self._data: np.memmap | None = None

    def __len__(self) -> int:
        return self.length

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_data"] = None
        return state

    def _array(self) -> np.memmap:
        if self._data is None:
            self._data = np.load(self.npy_path, mmap_mode="r")
        return self._data

    def _decode_index(self, idx: int) -> tuple[int, int, int | None, int | None]:
        if self.train:
            base = idx // self.crop_mult
            c = base // self.t_windows
            t_idx = base % self.t_windows
            return c, t_idx, None, None

        blocks_per_channel = self.t_windows * self.spatial_blocks
        c = idx // blocks_per_channel
        within_c = idx % blocks_per_channel
        t_idx = within_c // self.spatial_blocks
        block_idx = within_c % self.spatial_blocks
        return c, t_idx, block_idx // self.n_w, block_idx % self.n_w

    def _temporal_slice(self, c: int, t0: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        t1 = min(t0 + self.n_frame, self.T_full)
        arr = np.asarray(self._array()[c, t0:t1, y0:y1, x0:x1], dtype=np.float32)
        missing = self.n_frame - arr.shape[0]
        if missing > 0:
            tail = np.asarray(
                self._array()[c, self.T_full - missing:self.T_full, y0:y1, x0:x1],
                dtype=np.float32,
            )[::-1]
            arr = np.concatenate((arr, tail), axis=0)
        return np.array(arr, dtype=np.float32, copy=True)

    def _train_patch(self, c: int, t0: int) -> torch.Tensor:
        h0 = torch.randint(0, self.H - self.train_size + 1, (1,)).item()
        w0 = torch.randint(0, self.W - self.train_size + 1, (1,)).item()
        arr = self._temporal_slice(c, t0, h0, h0 + self.train_size, w0, w0 + self.train_size)
        return torch.from_numpy(arr)

    def _validation_patch(self, c: int, t0: int, block_h: int, block_w: int) -> torch.Tensor:
        # Coordinates in the reflection-padded image, then mapped to the source field.
        padded_y0 = block_h * self.train_size
        padded_x0 = block_w * self.train_size
        source_y0 = padded_y0 - self.pad_top
        source_x0 = padded_x0 - self.pad_left
        source_y1 = source_y0 + self.train_size
        source_x1 = source_x0 + self.train_size

        y0, y1 = max(0, source_y0), min(self.H, source_y1)
        x0, x1 = max(0, source_x0), min(self.W, source_x1)
        arr = self._temporal_slice(c, t0, y0, y1, x0, x1)
        data = torch.from_numpy(arr)

        pad_top = max(0, -source_y0)
        pad_bottom = max(0, source_y1 - self.H)
        pad_left = max(0, -source_x0)
        pad_right = max(0, source_x1 - self.W)
        if pad_top or pad_bottom or pad_left or pad_right:
            data = F.pad(
                data.unsqueeze(0),
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="reflect",
            ).squeeze(0)
        return data

    def _normalize(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = torch.finfo(data.dtype).eps
        if self.norm_type == "mean_range":
            # Official public implementation default: one mean/range per T-H-W patch.
            offset = data.mean().view(1, 1, 1)
            scale = (data.max() - data.min()).view(1, 1, 1)
        elif self.norm_type == "mean_range_hw":
            # Optional paper-literal mode: one mean/range per frame.
            offset = data.mean(dim=(-2, -1), keepdim=True)
            scale = data.amax(dim=(-2, -1), keepdim=True) - data.amin(dim=(-2, -1), keepdim=True)
        elif self.norm_type == "min_max":
            dmin, dmax = data.min(), data.max()
            offset = ((dmax + dmin) / 2).view(1, 1, 1)
            scale = ((dmax - dmin) / 2).view(1, 1, 1)
        else:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}")
        scale = torch.where(scale.abs() > eps, scale, torch.ones_like(scale))
        return (data - offset) / scale, offset, scale

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        c, t_idx, block_h, block_w = self._decode_index(idx)
        t0 = t_idx * self.t_stride
        if self.train:
            data = self._train_patch(c, t0)
        else:
            assert block_h is not None and block_w is not None
            data = self._validation_patch(c, t0, block_h, block_w)
        data, offset, scale = self._normalize(data)
        return {
            "input": data.unsqueeze(0),  # [1, T, H, W]
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


def load_caesar_d_diffusion(
    ckpt_path: str | Path,
    device: torch.device,
    diffusion_steps: int = 32,
    predicted_frames: int = 10,
):
    from CAESAR.models.video_diffusion_interpo import GaussianDiffusion, Unet3D

    checkpoint = safe_torch_load(ckpt_path, device)
    state = checkpoint["diffusion"] if isinstance(checkpoint, dict) and "diffusion" in checkpoint else checkpoint
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
        num_frames=predicted_frames,
        channels=64,
        timesteps=diffusion_steps,
        loss_type="l2",
    )
    diffusion.load_state_dict(state)
    return diffusion.to(device)


# ---------------------------------------------------------------------------
# VAE stage
# ---------------------------------------------------------------------------


def select_vae_input(full_x: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
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
    mean_bits = result["frame_bit"].mean()
    if args.model_type == "D" and args.stage == 1 and not args.d_stage1_all_frames:
        batch_size, _, t_all = full_x.shape[:3]
        n_keyframes = model_x.shape[2]
        bits_seq = result["frame_bit"].reshape(batch_size, n_keyframes).sum(dim=1)
        displayed_bpp = bits_seq.mean() / (t_all * full_x.shape[-2] * full_x.shape[-1])
    else:
        displayed_bpp = result["bpp"].mean()
    rate_term = mean_bits if args.rate_mode == "bits" else displayed_bpp
    return rate_term, displayed_bpp, mean_bits


def vae_distortion_metrics(
    result: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    full_x: torch.Tensor,
    model_x: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    non_blocking: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized_mse = F.mse_loss(result["output"], model_x)
    distortion_domain = getattr(args, "distortion_domain", "normalized")
    if "scale" not in batch and distortion_domain == "source":
        raise ValueError("source-domain distortion requires each dataset batch to provide scale")
    if "scale" not in batch:
        return normalized_mse, normalized_mse, normalized_mse

    scale = batch["scale"].to(device, non_blocking=non_blocking)
    if scale.ndim == full_x.ndim - 1:
        scale = scale.unsqueeze(1)
    if scale.ndim != full_x.ndim:
        raise ValueError(
            f"Cannot broadcast normalization scale {tuple(scale.shape)} "
            f"to input {tuple(full_x.shape)}"
        )
    if scale.shape[2] == full_x.shape[2] and model_x.shape[2] != full_x.shape[2]:
        indices = torch.arange(0, full_x.shape[2], args.interpo_rate, device=device)
        scale = scale.index_select(2, indices)
    source_mse = ((result["output"] - model_x) * scale).square().mean()
    distortion = source_mse if distortion_domain == "source" else normalized_mse
    return distortion, normalized_mse, source_mse


@torch.no_grad()
def eval_vae(
    model: torch.nn.Module,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    non_blocking: bool,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    sum_loss = sum_distortion = sum_mse = sum_source_mse = sum_bpp = sum_bits = 0.0
    n = 0
    for batch in val_loader:
        full_x = batch["input"].to(device, non_blocking=non_blocking)
        model_x = select_vae_input(full_x, args)
        result = model(model_x)
        distortion, mse, source_mse = vae_distortion_metrics(
            result, batch, full_x, model_x, args, device, non_blocking
        )
        rate_term, bpp, mean_bits = vae_rate_metrics(result, full_x, model_x, args)
        rd_loss = distortion + args.lambda_rate * rate_term
        bn = full_x.shape[0]
        n += bn
        sum_loss += rd_loss.item() * bn
        sum_distortion += distortion.item() * bn
        sum_mse += mse.item() * bn
        sum_source_mse += source_mse.item() * bn
        sum_bpp += bpp.item() * bn
        sum_bits += mean_bits.item() * bn
    if was_training:
        model.train()
    if n == 0:
        raise RuntimeError("Validation loader produced zero batches.")
    return {
        "loss": sum_loss / n,
        "distortion": sum_distortion / n,
        "mse": sum_mse / n,
        "source_mse": sum_source_mse / n,
        "bpp": sum_bpp / n,
        "mean_bits": sum_bits / n,
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
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable VAE parameters remain after applying --trainable_scope.")
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr)
    anchor_parameters = (
        {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        if args.anchor_weight > 0
        else {}
    )
    parameter_count = sum(parameter.numel() for parameter in trainable_parameters)
    milestone_steps = set(args.milestone_steps)
    non_blocking = device.type == "cuda"
    accum = args.gradient_accumulation_steps
    train_iter = infinite_batches(train_loader)

    initial_metrics = eval_vae(model, val_loader, args, device, non_blocking)
    best_val = initial_metrics["loss"]
    atomic_torch_save(model.state_dict(), args.output_ckpt)
    logger.log(
        {
            "val/loss": initial_metrics["loss"],
            "val/distortion": initial_metrics["distortion"],
            "val/mse": initial_metrics["mse"],
            "val/source_mse": initial_metrics["source_mse"],
            "val/bpp": initial_metrics["bpp"],
            "val/mean_bits": initial_metrics["mean_bits"],
            "val/update_step": 0,
            "val/epoch": 0.0,
            "val/is_initial": 1,
            "val/loss_change_from_initial_pct": 0.0,
            "val/distortion_change_from_initial_pct": 0.0,
            "val/mse_change_from_initial_pct": 0.0,
            "val/source_mse_change_from_initial_pct": 0.0,
            "val/bpp_change_from_initial_pct": 0.0,
        }
    )
    print(
        f"{label} initial validation: loss={initial_metrics['loss']:.6f}, "
        f"distortion={initial_metrics['distortion']:.6f}, "
        f"normalized_mse={initial_metrics['mse']:.6f}, "
        f"source_mse={initial_metrics['source_mse']:.6f}, "
        f"bpp={initial_metrics['bpp']:.6f}"
    )

    update_step = 0
    pbar = tqdm(total=args.iterations, desc=f"{label} fine-tune")
    start_time = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    while update_step < args.iterations:
        micro_loss = micro_distortion = micro_mse = micro_source_mse = micro_bpp = micro_bits = 0.0
        for _ in range(accum):
            batch = next(train_iter)
            full_x = batch["input"].to(device, non_blocking=non_blocking)
            model_x = select_vae_input(full_x, args)
            result = model(model_x)
            distortion, mse, source_mse = vae_distortion_metrics(
                result, batch, full_x, model_x, args, device, non_blocking
            )
            rate_term, bpp, mean_bits = vae_rate_metrics(result, full_x, model_x, args)
            anchor_loss = (
                sum(
                    (parameter - anchor_parameters[name]).square().sum()
                    for name, parameter in model.named_parameters()
                )
                / parameter_count
                if anchor_parameters
                else distortion.new_zeros(())
            )
            loss = distortion + args.lambda_rate * rate_term + args.anchor_weight * anchor_loss
            (loss / accum).backward()
            micro_loss += loss.item()
            micro_distortion += distortion.item()
            micro_mse += mse.item()
            micro_source_mse += source_mse.item()
            micro_bpp += bpp.item()
            micro_bits += mean_bits.item()

        current_lr = args.lr
        if args.warmup_updates > 0:
            current_lr *= min(1.0, (update_step + 1) / args.warmup_updates)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update_step += 1

        train_loss = micro_loss / accum
        train_distortion = micro_distortion / accum
        train_mse = micro_mse / accum
        train_source_mse = micro_source_mse / accum
        train_bpp = micro_bpp / accum
        train_bits = micro_bits / accum
        pbar.update(1)
        pbar.set_postfix(
            loss=f"{train_loss:.5f}",
            distortion=f"{train_distortion:.6f}",
            bpp=f"{train_bpp:.5f}",
        )

        if update_step % args.log_interval == 0:
            logger.log(
                {
                    "train/loss": train_loss,
                    "train/distortion": train_distortion,
                    "train/mse": train_mse,
                    "train/source_mse": train_source_mse,
                    "train/bpp": train_bpp,
                    "train/mean_bits": train_bits,
                    "train/anchor_loss": float(anchor_loss.detach()),
                    "train/update_step": update_step,
                    "train/epoch": update_step / len(train_loader),
                    "train/lr": current_lr,
                }
            )

        if update_step % args.val_interval == 0 or update_step == args.iterations:
            metrics = eval_vae(model, val_loader, args, device, non_blocking)
            logger.log(
                {
                    "val/loss": metrics["loss"],
                    "val/distortion": metrics["distortion"],
                    "val/mse": metrics["mse"],
                    "val/source_mse": metrics["source_mse"],
                    "val/bpp": metrics["bpp"],
                    "val/mean_bits": metrics["mean_bits"],
                    "val/update_step": update_step,
                    "val/epoch": update_step / len(train_loader),
                    "val/is_initial": 0,
                    "val/loss_change_from_initial_pct": percent_change(
                        metrics["loss"], initial_metrics["loss"]
                    ),
                    "val/distortion_change_from_initial_pct": percent_change(
                        metrics["distortion"], initial_metrics["distortion"]
                    ),
                    "val/mse_change_from_initial_pct": percent_change(
                        metrics["mse"], initial_metrics["mse"]
                    ),
                    "val/source_mse_change_from_initial_pct": percent_change(
                        metrics["source_mse"], initial_metrics["source_mse"]
                    ),
                    "val/bpp_change_from_initial_pct": percent_change(
                        metrics["bpp"], initial_metrics["bpp"]
                    ),
                }
            )
            if metrics["loss"] < best_val:
                best_val = metrics["loss"]
                atomic_torch_save(model.state_dict(), args.output_ckpt)
                pbar.write(
                    f"update {update_step}: best "
                    f"(val_loss={metrics['loss']:.6f}, "
                    f"distortion={metrics['distortion']:.6f}, "
                    f"normalized_mse={metrics['mse']:.6f}, "
                    f"source_mse={metrics['source_mse']:.6f}, "
                    f"bpp={metrics['bpp']:.6f})"
                )

        if update_step % args.save_interval == 0 or update_step in milestone_steps:
            atomic_torch_save(model.state_dict(), step_ckpt_path(args.output_ckpt, update_step))

    pbar.close()
    if not Path(args.output_ckpt).exists():
        raise RuntimeError("No best checkpoint saved.")
    model.load_state_dict(safe_torch_load(args.output_ckpt, device))
    print(f"{label} done: {update_step} updates, {time.time() - start_time:.0f}s, best_val={best_val:.6f}")
    return model


# ---------------------------------------------------------------------------
# Diffusion stage (CAESAR-D stage 2)
# ---------------------------------------------------------------------------


def cond_aligned_norm(latent: torch.Tensor, interpo_rate: int) -> torch.Tensor:
    """Normalize all target latents using statistics from zero-filled condition latents.

    The zero-filled conditional tensor mirrors the official CAESAR-D decoding
    input before sampling: keyframe positions contain VAE latents; positions to
    be generated contain zeros.
    """
    cond_idx = torch.arange(0, latent.shape[2], interpo_rate, device=latent.device)
    cond_latent = torch.zeros_like(latent)
    cond_latent.index_copy_(2, cond_idx, latent.index_select(2, cond_idx))
    x_min = cond_latent.amin(dim=(1, 2, 3, 4), keepdim=True)
    x_max = cond_latent.amax(dim=(1, 2, 3, 4), keepdim=True)
    scale = (x_max - x_min + 1e-8) / 2
    offset = x_min + scale
    return (latent - offset) / scale


@torch.no_grad()
def make_diffusion_latent(vae: torch.nn.Module, x: torch.Tensor, interpo_rate: int) -> torch.Tensor:
    latent = vae.inference_qlatent(x)
    return cond_aligned_norm(latent, interpo_rate)


def diffusion_objective_loss(
    diffusion: torch.nn.Module,
    latent: torch.Tensor,
    interpo_rate: int,
    objective: str,
    x0_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, noise, and reconstructed-latent losses for Stage2."""
    if objective == "noise":
        noise_loss = diffusion(latent, interpo_rate=interpo_rate)
        return noise_loss, noise_loss, noise_loss.new_zeros(())

    batch = latent.shape[0]
    timesteps = torch.randint(
        0, diffusion.num_timesteps, (batch,), device=latent.device
    ).long()
    noise = torch.randn_like(latent)
    noisy = diffusion.q_sample(x_start=latent, t=timesteps, noise=noise)
    condition_indices = torch.arange(
        0, latent.shape[2], interpo_rate, device=latent.device
    )
    predicted_mask = torch.ones(latent.shape[2], dtype=torch.bool, device=latent.device)
    predicted_mask[condition_indices] = False
    noisy[:, :, condition_indices] = latent[:, :, condition_indices]

    predicted_noise = diffusion.denoise_fn(noisy, timesteps)
    noise_loss = F.mse_loss(
        predicted_noise[:, :, predicted_mask], noise[:, :, predicted_mask]
    )
    predicted_x0 = diffusion.predict_start_from_noise(noisy, timesteps, predicted_noise)
    x0_loss = F.mse_loss(
        predicted_x0[:, :, predicted_mask], latent[:, :, predicted_mask]
    )
    if objective == "x0":
        total = x0_loss
    elif objective == "hybrid":
        total = noise_loss + x0_weight * x0_loss
    else:
        raise ValueError(f"Unsupported diffusion objective: {objective}")
    return total, noise_loss, x0_loss


@torch.no_grad()
def eval_diffusion(
    diffusion: torch.nn.Module,
    vae: torch.nn.Module,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    non_blocking: bool,
) -> float:
    was_training = diffusion.training
    diffusion.eval()
    total, count = 0.0, 0
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    # Diffusion loss samples timesteps and noise. Replaying the same validation
    # RNG stream makes checkpoint comparisons meaningful.
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(args.seed + 1_000_003)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + 1_000_003)
        for batch in val_loader:
            x = batch["input"].to(device, non_blocking=non_blocking)
            latent = make_diffusion_latent(vae, x, args.interpo_rate)
            loss, _, _ = diffusion_objective_loss(
                diffusion,
                latent,
                args.interpo_rate,
                args.diffusion_objective,
                args.diffusion_x0_weight,
            )
            bn = x.shape[0]
            total += loss.item() * bn
            count += bn
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
    anchor_parameters = (
        {name: parameter.detach().clone() for name, parameter in diffusion.named_parameters()}
        if args.anchor_weight > 0
        else {}
    )
    parameter_count = sum(parameter.numel() for parameter in diffusion.parameters())
    milestone_steps = set(args.milestone_steps)
    non_blocking = device.type == "cuda"
    accum = args.gradient_accumulation_steps
    train_iter = infinite_batches(train_loader)

    initial_val = eval_diffusion(
        diffusion, vae, val_loader, args, device, non_blocking
    )
    best_val = initial_val
    atomic_torch_save(
        {"vae": vae.state_dict(), "diffusion": diffusion.state_dict()},
        args.output_ckpt,
    )
    logger.log(
        {
            "val/diff_loss": initial_val,
            "val/update_step": 0,
            "val/epoch": 0.0,
            "val/is_initial": 1,
            "val/diff_loss_change_from_initial_pct": 0.0,
        }
    )
    print(f"Diffusion initial validation: loss={initial_val:.6f}")

    update_step = 0
    pbar = tqdm(total=args.iterations, desc="Diffusion fine-tune")
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    while update_step < args.iterations:
        micro_loss = 0.0
        micro_noise_loss = 0.0
        micro_x0_loss = 0.0
        for _ in range(accum):
            batch = next(train_iter)
            x = batch["input"].to(device, non_blocking=non_blocking)
            latent = make_diffusion_latent(vae, x, args.interpo_rate)
            diffusion_loss, noise_loss, x0_loss = diffusion_objective_loss(
                diffusion,
                latent,
                args.interpo_rate,
                args.diffusion_objective,
                args.diffusion_x0_weight,
            )
            anchor_loss = (
                sum(
                    (parameter - anchor_parameters[name]).square().sum()
                    for name, parameter in diffusion.named_parameters()
                )
                / parameter_count
                if anchor_parameters
                else diffusion_loss.new_zeros(())
            )
            loss = diffusion_loss + args.anchor_weight * anchor_loss
            (loss / accum).backward()
            micro_loss += loss.item()
            micro_noise_loss += noise_loss.item()
            micro_x0_loss += x0_loss.item()

        current_lr = args.lr
        if args.warmup_updates > 0:
            current_lr *= min(1.0, (update_step + 1) / args.warmup_updates)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update_step += 1

        train_loss = micro_loss / accum
        train_noise_loss = micro_noise_loss / accum
        train_x0_loss = micro_x0_loss / accum
        pbar.update(1)
        pbar.set_postfix(loss=f"{train_loss:.6f}")

        if update_step % args.log_interval == 0:
            logger.log(
                {
                    "train/diff_loss": train_loss,
                    "train/noise_loss": train_noise_loss,
                    "train/x0_loss": train_x0_loss,
                    "train/anchor_loss": float(anchor_loss.detach()),
                    "train/update_step": update_step,
                    "train/epoch": update_step / len(train_loader),
                    "train/lr": current_lr,
                }
            )

        if update_step % args.val_interval == 0 or update_step == args.iterations:
            val_loss = eval_diffusion(diffusion, vae, val_loader, args, device, non_blocking)
            logger.log(
                {
                    "val/diff_loss": val_loss,
                    "val/update_step": update_step,
                    "val/epoch": update_step / len(train_loader),
                    "val/is_initial": 0,
                    "val/diff_loss_change_from_initial_pct": percent_change(
                        val_loss, initial_val
                    ),
                }
            )
            if val_loss < best_val:
                best_val = val_loss
                atomic_torch_save(
                    {"vae": vae.state_dict(), "diffusion": diffusion.state_dict()},
                    args.output_ckpt,
                )
                pbar.write(f"update {update_step}: best (val={val_loss:.6f})")

        if update_step % args.save_interval == 0 or update_step in milestone_steps:
            atomic_torch_save(
                {"vae": vae.state_dict(), "diffusion": diffusion.state_dict()},
                step_ckpt_path(args.output_ckpt, update_step),
            )

    pbar.close()
    if not Path(args.output_ckpt).exists():
        raise RuntimeError("No best checkpoint saved.")
    best = safe_torch_load(args.output_ckpt, device)
    diffusion.load_state_dict(remove_module_prefix(best["diffusion"]))
    print(f"Diffusion done: {update_step} updates, {time.time() - start_time:.0f}s, best_val={best_val:.6f}")
    return diffusion


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune CAESAR on ERA5 mmap data")

    parser.add_argument("--model_type", default="V", choices=["V", "D"])
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
    parser.add_argument("--n_frame", type=int, default=None)
    parser.add_argument("--interpo_rate", type=int, default=3)
    parser.add_argument("--diffusion_steps", type=int, default=32)
    parser.add_argument(
        "--diffusion_objective",
        choices=["noise", "x0", "hybrid"],
        default="noise",
        help="Stage2 objective. 'noise' preserves the public CAESAR training path.",
    )
    parser.add_argument(
        "--diffusion_x0_weight",
        type=float,
        default=0.1,
        help="Reconstructed-latent loss weight when --diffusion_objective=hybrid.",
    )

    parser.add_argument("--data_dir", default="/workspace/Data/ERA5/finetune_processed")
    parser.add_argument(
        "--data_backend",
        choices=["mmap", "netcdf", "npy_shards"],
        default="mmap",
        help="Use a contiguous mmap, direct NetCDF streaming, or daily normalized mmap shards.",
    )
    parser.add_argument(
        "--cra5_stats_dir",
        default="/workspace/AIForCompression/models/CRA5/cra5/dataset",
        help="Directory containing CRA5 mean_std.json files for the NetCDF backend.",
    )
    parser.add_argument(
        "--train_timesteps",
        type=int,
        default=1920,
        help="Earliest chronological NetCDF frames used for training.",
    )
    parser.add_argument(
        "--val_timesteps",
        type=int,
        default=240,
        help="NetCDF frames immediately following training used for validation.",
    )
    parser.add_argument(
        "--netcdf_val_channel_stride",
        type=int,
        default=1,
        help="Validate every Nth physical channel for faster internal checkpoint screening.",
    )
    parser.add_argument(
        "--train_channel_start",
        type=int,
        default=0,
        help="First ERA5 physical channel used by NetCDF/npy-shard training.",
    )
    parser.add_argument(
        "--train_channel_end",
        type=int,
        default=None,
        help="Exclusive ERA5 channel bound; defaults to all 268 channels.",
    )
    parser.add_argument(
        "--netcdf_max_open_file_pairs",
        type=int,
        default=4,
        help="Per-worker LRU size for daily pressure/single NetCDF pairs.",
    )
    parser.add_argument("--train_size", type=int, default=256)
    parser.add_argument(
        "--temporal_stride",
        type=int,
        default=None,
        help="Temporal window stride. Default is n_frame (official non-overlap setting).",
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=1,
        help="Spacing between frames inside a sequence; use 24 for same-hour daily ERA5.",
    )
    parser.add_argument(
        "--norm_type",
        default="mean_range",
        choices=["mean_range", "mean_range_hw", "min_max"],
        help="mean_range matches the public CAESAR dataset default; mean_range_hw normalizes frame-wise.",
    )
    parser.add_argument(
        "--crop_multiplier",
        type=int,
        default=1,
        help="Number of independently sampled training crops per (channel, temporal-window) pair.",
    )

    parser.add_argument("--caesar_root", default="/workspace/AIForCompression/models/CAESAR")
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--vae_ckpt_path", default=None)
    parser.add_argument("--output_ckpt", default=None)

    parser.add_argument("--iterations", type=int, default=100000, help="Number of optimizer updates.")
    parser.add_argument("--batch_size", type=int, default=32, help="Micro-batch size per forward/backward pass.")
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=None,
        help="Validation batch size; defaults to --batch_size.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Effective batch size = batch_size * gradient_accumulation_steps.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_rate", type=float, default=1e-5)
    parser.add_argument("--rate_mode", choices=["bits", "bpp"], default="bits")
    parser.add_argument(
        "--distortion_domain",
        choices=["normalized", "source"],
        default="normalized",
        help=(
            "Compute VAE distortion after patch normalization (legacy behavior) "
            "or in the source CRA5 domain by restoring each patch scale."
        ),
    )
    parser.add_argument(
        "--anchor_weight",
        type=float,
        default=0.0,
        help="L2-SP weight on mean squared drift from the loaded pretrained parameters.",
    )
    parser.add_argument(
        "--warmup_updates",
        type=int,
        default=0,
        help="Linearly warm learning rate from lr/warmup_updates to --lr.",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--trainable_scope",
        choices=["all", "decoder", "sr"],
        default="all",
        help=(
            "VAE parameters to optimize. For CAESAR-V, 'decoder' trains "
            "entropy_model.dec and sr_model; for CAESAR-D stage 1, it trains "
            "dec. In both cases bitrate-producing modules remain frozen. 'sr' "
            "is available only for CAESAR-V and trains sr_model."
        ),
    )
    parser.add_argument("--log_interval", type=int, default=200, help="Interval in optimizer updates.")
    parser.add_argument("--val_interval", type=int, default=2000, help="Interval in optimizer updates.")
    parser.add_argument("--save_interval", type=int, default=10000, help="Interval in optimizer updates.")
    parser.add_argument(
        "--milestone_steps",
        type=int,
        nargs="*",
        default=[],
        help="Additional early optimizer updates at which checkpoints are saved.",
    )

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--d_stage1_all_frames",
        action="store_true",
        help="Train D-stage1 VAE on every frame instead of codec keyframes only.",
    )
    parser.add_argument("--wandb_project", default="caesar-finetune-era5")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=[])
    parser.add_argument("--require_wandb", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.model_type == "V" and args.stage != 1:
        raise ValueError("CAESAR-V only has stage 1.")
    if args.trainable_scope != "all" and args.model_type == "D" and args.stage != 1:
        raise ValueError("Restricted --trainable_scope is supported only for VAE stage 1.")
    if args.trainable_scope == "sr" and args.model_type != "V":
        raise ValueError("--trainable_scope sr is supported only for CAESAR-V.")
    if args.n_frame is None:
        args.n_frame = 8 if args.model_type == "V" else 16
    if args.model_type == "V" and args.n_frame % 4 != 0:
        raise ValueError("CAESAR-V n_frame must be divisible by 4.")
    if args.model_type == "D" and args.stage == 2:
        predicted_frames = count_predicted_frames(args.n_frame, args.interpo_rate)
        if predicted_frames <= 0:
            raise ValueError("CAESAR-D Stage 2 requires at least one predicted frame.")
    positive_fields = (
        "iterations",
        "batch_size",
        "gradient_accumulation_steps",
        "log_interval",
        "val_interval",
        "save_interval",
        "train_size",
    )
    for name in positive_fields:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive.")
    if args.val_batch_size is not None and args.val_batch_size <= 0:
        raise ValueError("--val_batch_size must be positive.")
    if args.anchor_weight < 0 or args.warmup_updates < 0:
        raise ValueError("--anchor_weight and --warmup_updates cannot be negative.")
    if args.diffusion_x0_weight < 0:
        raise ValueError("--diffusion_x0_weight cannot be negative.")
    if args.train_timesteps <= 0 or args.val_timesteps <= 0:
        raise ValueError("--train_timesteps and --val_timesteps must be positive.")
    if args.netcdf_val_channel_stride <= 0 or args.netcdf_max_open_file_pairs <= 0:
        raise ValueError("NetCDF stride/cache arguments must be positive.")
    if args.train_channel_start < 0:
        raise ValueError("--train_channel_start cannot be negative.")
    if (
        args.train_channel_end is not None
        and args.train_channel_end <= args.train_channel_start
    ):
        raise ValueError("--train_channel_end must be greater than --train_channel_start.")
    if args.data_backend == "mmap" and (
        args.train_channel_start != 0 or args.train_channel_end is not None
    ):
        raise ValueError("Channel-restricted training requires netcdf or npy_shards backend.")
    invalid_milestones = [step for step in args.milestone_steps if step <= 0 or step > args.iterations]
    if invalid_milestones:
        raise ValueError(f"--milestone_steps must be in [1, iterations], got {invalid_milestones}")
    if args.frame_step <= 0:
        raise ValueError("--frame_step must be positive.")
    if args.temporal_stride is not None and args.temporal_stride > args.n_frame:
        raise ValueError("--temporal_stride must be <= --n_frame.")


def default_paths(args: argparse.Namespace) -> None:
    base_ckpt = Path("/workspace/AIForCompression/checkpoints/caesar")
    if args.ckpt_path is None:
        args.ckpt_path = str(base_ckpt / ("caesar_v.pt" if args.model_type == "V" else "caesar_d.pt"))
    if args.output_ckpt is None:
        if args.model_type == "V":
            args.output_ckpt = str(base_ckpt / "caesar_v_tuning_era5.pt")
        elif args.stage == 1:
            args.output_ckpt = str(base_ckpt / "caesar_d_tuning_era5_vae.pt")
        else:
            args.output_ckpt = str(base_ckpt / "caesar_d_tuning_era5.pt")
    if args.model_type == "D" and args.stage == 2 and args.vae_ckpt_path is None:
        args.vae_ckpt_path = str(base_ckpt / "caesar_d_tuning_era5_vae.pt")


def configure_trainable_scope(
    model: torch.nn.Module, args: argparse.Namespace
) -> tuple[int, int]:
    if args.trainable_scope == "all":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    else:
        if args.trainable_scope == "sr":
            prefixes = ("sr_model.",)
        elif args.model_type == "D":
            prefixes = ("dec.",)
        else:
            prefixes = ("entropy_model.dec.", "sr_model.")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(prefixes))

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return trainable, total


def make_loaders(args: argparse.Namespace, use_pin_memory: bool) -> tuple[DataLoader, DataLoader]:
    train_batch_sampler = None
    if args.data_backend == "mmap":
        train_path = os.path.join(args.data_dir, "era5_train.npy")
        val_path = os.path.join(args.data_dir, "era5_val.npy")
        train_ds = ERA5MmapDataset(
            train_path,
            n_frame=args.n_frame,
            train=True,
            train_size=args.train_size,
            temporal_stride=args.temporal_stride,
            crop_multiplier=args.crop_multiplier,
            norm_type=args.norm_type,
        )
        val_ds = ERA5MmapDataset(
            val_path,
            n_frame=args.n_frame,
            train=False,
            train_size=args.train_size,
            temporal_stride=args.temporal_stride,
            norm_type=args.norm_type,
        )
    elif args.data_backend == "netcdf":
        from utils.era5_netcdf_dataset import (
            ERA5NetCDFDataset,
            N_CHANNELS,
            TemporalWindowBatchSampler,
            discover_netcdf_frames,
            load_cra5_channel_stats,
        )

        frames = discover_netcdf_frames(args.data_dir)
        required = args.train_timesteps + args.val_timesteps
        if len(frames) < required:
            raise ValueError(
                f"NetCDF backend requires {required} chronological frames, "
                f"but only found {len(frames)} in {args.data_dir}."
            )
        train_frames = frames[:args.train_timesteps]
        val_frames = frames[args.train_timesteps:required]
        means, stds = load_cra5_channel_stats(args.cra5_stats_dir)
        train_channels = resolve_channel_indices(
            args.train_channel_start,
            args.train_channel_end,
            N_CHANNELS,
        )
        channel_end = train_channels[-1] + 1
        val_channels = train_channels[:: args.netcdf_val_channel_stride]
        common_netcdf = {
            "means": means,
            "stds": stds,
            "n_frame": args.n_frame,
            "train_size": args.train_size,
            "temporal_stride": args.temporal_stride,
            "frame_step": args.frame_step,
            "norm_type": args.norm_type,
            "max_open_file_pairs": args.netcdf_max_open_file_pairs,
        }
        train_ds = ERA5NetCDFDataset(
            train_frames,
            train=True,
            crop_multiplier=args.crop_multiplier,
            channels=train_channels,
            **common_netcdf,
        )
        val_ds = ERA5NetCDFDataset(
            val_frames,
            train=False,
            channels=val_channels,
            **common_netcdf,
        )
        train_batch_sampler = TemporalWindowBatchSampler(
            train_ds, args.batch_size, args.seed
        )
        print(
            f"NetCDF chronological split: train=[0,{args.train_timesteps}) "
            f"val=[{args.train_timesteps},{required}) | "
            f"train channels=[{args.train_channel_start},{channel_end}) | "
            f"internal val channels={len(val_channels)}/{len(train_channels)}"
        )
    else:
        from utils.era5_netcdf_dataset import (
            ERA5NpyShardDataset,
            N_CHANNELS,
            TemporalWindowBatchSampler,
            discover_npy_shard_frames,
        )

        frames = discover_npy_shard_frames(args.data_dir)
        required = args.train_timesteps + args.val_timesteps
        if len(frames) < required:
            raise ValueError(
                f"npy_shards backend requires {required} chronological frames, "
                f"but only found {len(frames)} in {args.data_dir}."
            )
        train_frames = frames[:args.train_timesteps]
        val_frames = frames[args.train_timesteps:required]
        train_channels = resolve_channel_indices(
            args.train_channel_start,
            args.train_channel_end,
            N_CHANNELS,
        )
        channel_end = train_channels[-1] + 1
        val_channels = train_channels[:: args.netcdf_val_channel_stride]
        common_shards = {
            "n_frame": args.n_frame,
            "train_size": args.train_size,
            "temporal_stride": args.temporal_stride,
            "frame_step": args.frame_step,
            "norm_type": args.norm_type,
            "max_open_shards": args.netcdf_max_open_file_pairs,
        }
        train_ds = ERA5NpyShardDataset(
            train_frames,
            train=True,
            crop_multiplier=args.crop_multiplier,
            channels=train_channels,
            **common_shards,
        )
        val_ds = ERA5NpyShardDataset(
            val_frames,
            train=False,
            channels=val_channels,
            **common_shards,
        )
        train_batch_sampler = TemporalWindowBatchSampler(
            train_ds, args.batch_size, args.seed
        )
        print(
            f"Shard chronological split: train=[0,{args.train_timesteps}) "
            f"val=[{args.train_timesteps},{required}) | "
            f"train channels=[{args.train_channel_start},{channel_end}) | "
            f"internal val channels={len(val_channels)}/{len(train_channels)}"
        )

    print(f"Train: {len(train_ds)} items | Val: {len(val_ds)} items")
    print(
        f"Data shape: C={train_ds.C} T={train_ds.T_full} H={train_ds.H} W={train_ds.W} | "
        f"n_frame={args.n_frame} frame_step={getattr(train_ds, 'frame_step', 1)} "
        f"stride={train_ds.t_stride} tail_pad={train_ds.pad_t}"
    )
    print(
        f"Temporal windows: train={train_ds.t_windows} val={val_ds.t_windows} | "
        f"Val spatial blocks={val_ds.n_h}x{val_ds.n_w}={val_ds.spatial_blocks}"
    )

    common: dict[str, Any] = {
        "num_workers": args.num_workers,
        "pin_memory": use_pin_memory,
        "persistent_workers": args.num_workers > 0,
        "worker_init_fn": worker_init_fn if args.num_workers > 0 else None,
    }
    if args.num_workers > 0:
        common["prefetch_factor"] = args.prefetch_factor
        # Workers start lazily during the first validation pass, after the model
        # is already on CUDA. "spawn" prevents each worker from inheriting and
        # retaining a copy of the parent's CUDA context.
        common["multiprocessing_context"] = "spawn"

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    if train_batch_sampler is None:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            generator=generator,
            **common,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            generator=generator,
            **common,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch_size or args.batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )

    if len(train_loader) == 0:
        raise RuntimeError("No training batches; reduce --batch_size or provide more training samples.")
    if len(val_loader) == 0:
        raise RuntimeError("No validation batches.")
    return train_loader, val_loader


def main() -> None:
    args = parse_args()
    validate_args(args)
    default_paths(args)

    sys.path.insert(0, args.caesar_root)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    use_pin_memory = device.type == "cuda"

    effective_batch = args.batch_size * args.gradient_accumulation_steps
    print(
        f"CAESAR-{args.model_type} Stage {args.stage} | device={device} | "
        f"n_frame={args.n_frame} interpo_rate={args.interpo_rate} norm={args.norm_type}"
    )
    print(
        f"Updates={args.iterations} | micro_batch={args.batch_size} | "
        f"accum={args.gradient_accumulation_steps} | effective_batch={effective_batch}"
    )
    print(f"Data:       {args.data_dir}")
    print(f"Checkpoint: {args.ckpt_path}")
    if args.model_type == "D" and args.stage == 2:
        print(f"VAE ckpt:   {args.vae_ckpt_path}")
    print(f"Output:     {args.output_ckpt}")

    run_name = args.wandb_run_name or (
        f"CAESAR-{args.model_type}_S{args.stage}_era5_lr{args.lr}_"
        f"microb{args.batch_size}_effb{effective_batch}_{args.rate_mode}"
    )
    logger = MetricLogger(
        args.no_wandb,
        args.wandb_project,
        run_name,
        vars(args),
        group=args.wandb_group,
        tags=args.wandb_tags,
        required=args.require_wandb,
    )
    train_loader, val_loader = make_loaders(args, use_pin_memory)

    try:
        if args.model_type == "V":
            model = load_caesar_v(args.ckpt_path, device)
            trainable, total = configure_trainable_scope(model, args)
            print(
                f"CAESAR-V params: {total:,} total | {trainable:,} trainable "
                f"(scope={args.trainable_scope})"
            )
            finetune_vae(model, train_loader, val_loader, args, device, logger, "CAESAR-V")
        elif args.stage == 1:
            vae = load_caesar_d_vae(args.ckpt_path, device)
            trainable, total = configure_trainable_scope(vae, args)
            print(
                f"CAESAR-D VAE params: {total:,} total | {trainable:,} trainable "
                f"(scope={args.trainable_scope})"
            )
            finetune_vae(vae, train_loader, val_loader, args, device, logger, "CAESAR-D-VAE")
        else:
            vae = load_caesar_d_vae(args.vae_ckpt_path, device)
            diffusion = load_caesar_d_diffusion(
                args.ckpt_path,
                device,
                args.diffusion_steps,
                count_predicted_frames(args.n_frame, args.interpo_rate),
            )
            print(
                f"CAESAR-D VAE: {sum(p.numel() for p in vae.parameters()):,} | "
                f"Diffusion: {sum(p.numel() for p in diffusion.parameters()):,}"
            )
            finetune_diffusion(diffusion, vae, train_loader, val_loader, args, device, logger)
    finally:
        logger.finish()

    print("Done!")


if __name__ == "__main__":
    main()
