"""
Fine-tune CAESAR-V or CAESAR-D on a .npz dataset.
...
"""

import argparse
import math
import sys
import time
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import wandb

sys.path.insert(0, '/workspace/AIForCompression/models/CAESAR')


# ─── helpers ────────────────────────────────────────────────────────────

def remove_module_prefix(state_dict):
    new = OrderedDict()
    for k, v in state_dict.items():
        new[k.replace("module.", "")] = v
    return new


# ─── dataset ────────────────────────────────────────────────────────────

class SimpleTemporalDataset(Dataset):
    """Load .npz [V, S, T, H, W], yield [C=1, n_frame, 256, 256] patches."""

    def __init__(self, npz_path, n_frame, train=True, train_size=256):
        # Use mmap to avoid loading full 53GB into RAM at startup
        f = np.load(npz_path, mmap_mode='r')
        self._npz = f
        self._data = f['data']  # mmap-backed, lazy
        shape = self._data.shape

        self.V, self.S, self.T_full, self.H, self.W = shape
        self.n_frame = n_frame
        self.train = train
        self.train_size = train_size
        self.t_samples = max(1, (self.T_full - n_frame) // n_frame + 1)
        self.total = self.V * self.S * self.t_samples

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        v = idx // (self.S * self.t_samples)
        s = (idx // self.t_samples) % self.S
        t0 = (idx % self.t_samples) * self.n_frame

        # Read from mmap, convert to tensor (copies only this slice ~8*1024*1024*4=32MB)
        arr = np.array(self._data[v, s, t0:t0 + self.n_frame], dtype=np.float32)
        data = torch.from_numpy(arr)  # [T, H, W]

        # zero-mean unit-range normalization (paper Sec 4.2)
        offset = data.mean()
        scale = data.max() - data.min()
        if scale == 0:
            scale = torch.tensor(1.0)
        data = (data - offset) / scale

        # crop / pad to train_size
        cur = data.shape[-1]
        if self.train:
            if self.train_size < cur:
                h = torch.randint(0, cur - self.train_size + 1, (1,)).item()
                w = torch.randint(0, cur - self.train_size + 1, (1,)).item()
                data = data[..., h:h + self.train_size, w:w + self.train_size]
            elif self.train_size > cur:
                p = self.train_size - cur
                pl, pr = p // 2, p - p // 2
                data = F.pad(data[None], (pl, pr, pl, pr), mode='reflect')[0]
        else:
            if self.train_size < cur:
                h = (cur - self.train_size) // 2
                w = (cur - self.train_size) // 2
                data = data[..., h:h + self.train_size, w:w + self.train_size]

        return {"input": data.unsqueeze(0),          # [1, T, H, W]
                "offset": offset.view(1, 1, 1),
                "scale": scale.view(1, 1, 1)}


# ─── model loaders ──────────────────────────────────────────────────────

def load_caesar_v(ckpt_path, device):
    from CAESAR.models import compress_modules3d_mid_SR as cm
    model = cm.CompressorMix(dim=16, dim_mults=[1, 2, 3, 4], reverse_dim_mults=[4, 3, 2],
                             hyper_dims_mults=[4, 4, 4], channels=1, out_channels=1,
                             d3=True, sr_dim=16)
    sd = remove_module_prefix(torch.load(ckpt_path, map_location=device))
    model.load_state_dict(sd)
    return model.to(device)


def load_caesar_d_vae(ckpt_path, device):
    """Load CAESAR-D keyframe VAE only (from full or VAE-only checkpoint)."""
    from CAESAR.models import keyframe_compressor as kc
    ckpt = torch.load(ckpt_path, map_location=device)
    # Handle both full checkpoint {"vae": ..., "diffusion": ...} and vae-only state_dict
    if isinstance(ckpt, dict) and "vae" in ckpt:
        sd = remove_module_prefix(ckpt["vae"])
    else:
        sd = remove_module_prefix(ckpt)
    model = kc.ResnetCompressor(dim=16, dim_mults=[1, 2, 3, 4], reverse_dim_mults=[4, 3, 2, 1],
                                hyper_dims_mults=[4, 4, 4], channels=1, out_channels=1)
    model.load_state_dict(sd)
    return model.to(device)


def load_caesar_d_diffusion(ckpt_path, device):
    """Load CAESAR-D diffusion model."""
    from CAESAR.models.video_diffusion_interpo import Unet3D, GaussianDiffusion
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = remove_module_prefix(ckpt["diffusion"])

    unet = Unet3D(dim=64, out_dim=64, channels=64, dim_mults=(1, 2, 4, 8),
                  use_bert_text_cond=False)
    model = GaussianDiffusion(unet, image_size=16, num_frames=10, channels=64,
                              timesteps=32, loss_type='l2')
    model.load_state_dict(sd)
    return model.to(device)


# ─── training loops ─────────────────────────────────────────────────────

def finetune_vae(model, train_loader, val_loader, args, device, label="VAE"):
    """Rate-distortion fine-tuning of VAE (CAESAR-V or CAESAR-D keyframe).

    Paper Eq.18:  L = MSE(x, x̂) + λ * [ Ey[-log₂ p(y|μ,σ)] + Ez[-log₂ p(z)] ]
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    accum = args.gradient_accumulation_steps

    best_val = float('inf')
    step = 0
    pbar = tqdm(total=args.iterations, desc=f'{label} fine-tune')
    start_time = time.time()

    model.train()
    optimizer.zero_grad()

    while step < args.iterations:
        for batch in train_loader:
            if step >= args.iterations:
                break

            x = batch["input"].to(device)
            result = model(x)

            mse = F.mse_loss(result["output"], x)
            total_rate = result["frame_bit"].mean()
            loss = (mse + args.lambda_rate * total_rate) / accum

            loss.backward()

            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f'{loss.item():.4f}', mse=f'{mse.item():.6f}',
                             bpp=f'{total_rate.item() / (args.n_frame * 256 * 256):.4f}')

            if step % args.log_interval == 0:
                wandb.log({'train/loss': loss.item(), 'train/mse': mse.item(),
                           'train/step': step, 'train/lr': args.lr})

            if step % args.val_interval == 0 or step == args.iterations:
                model.eval()
                val_mse = val_rate = 0
                with torch.no_grad():
                    for batch_v in val_loader:
                        xv = batch_v["input"].to(device)
                        rv = model(xv)
                        val_mse += F.mse_loss(rv["output"], xv).item()
                        val_rate += rv["frame_bit"].mean().item()
                nv = len(val_loader)
                val_mse /= nv
                val_rate /= nv
                val_loss = val_mse + args.lambda_rate * val_rate

                wandb.log({'val/loss': val_loss, 'val/mse': val_mse,
                           'val/step': step, 'val/bpp': val_rate / (args.n_frame * 256 * 256)})

                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(model.state_dict(), args.output_ckpt)
                    pbar.write(f'  step {step}: best saved (val={val_loss:.4f})')

            if step % args.save_interval == 0:
                p = args.output_ckpt.replace('.pt', f'_step{step}.pt')
                torch.save(model.state_dict(), p)

    pbar.close()
    elapsed = time.time() - start_time
    print(f'{label} done: {step} iters in {elapsed/60:.1f} min, best_val={best_val:.4f}')
    return model


def finetune_diffusion(diffusion, vae, train_loader, val_loader, args, device):
    """Stage 2: Fine-tune latent diffusion model.

    Paper Eq.17: L_CDM = E[ || ε − ε_θ(y^N_t, t)_G ||²₂ ]
    i.e. noise-prediction MSE on non-keyframe latent positions only.
    """
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    diffusion.train()
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.lr)
    accum = args.gradient_accumulation_steps

    best_val = float('inf')
    step = 0
    pbar = tqdm(total=args.iterations, desc='Diffusion fine-tune')
    start_time = time.time()

    optimizer.zero_grad()

    while step < args.iterations:
        for batch in train_loader:
            if step >= args.iterations:
                break

            x = batch["input"].to(device)                     # [B, 1, T, H, W]

            # encode to latent via frozen VAE
            # VAE.encode expects [B*T, C, H, W]; we have [B, C=1, T, H, W]
            with torch.no_grad():
                B, C, T, H, W = x.shape
                x_flat = x.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
                latent_flat = vae.encode(x_flat)              # [B*T, latent_C, latent_H, latent_W]
                latent_C = latent_flat.shape[1]
                latent_H, latent_W = latent_flat.shape[2], latent_flat.shape[3]
                latent = latent_flat.view(B, T, latent_C, latent_H, latent_W)
                latent = latent.permute(0, 2, 1, 3, 4)       # [B, latent_C, T, latent_H, latent_W]

            # diffusion noise-prediction loss (only on non-keyframe positions)
            loss = diffusion(latent, interpo_rate=args.interpo_rate) / accum

            loss.backward()

            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f'{loss.item():.6f}')

            if step % args.log_interval == 0:
                wandb.log({'train/diff_loss': loss.item(), 'train/step': step})

            if step % args.val_interval == 0 or step == args.iterations:
                diffusion.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_v in val_loader:
                        xv = batch_v["input"].to(device)
                        Bv, Cv, Tv, Hv, Wv = xv.shape
                        xv_flat = xv.permute(0, 2, 1, 3, 4).reshape(-1, Cv, Hv, Wv)
                        lv = vae.encode(xv_flat)
                        lvC = lv.shape[1]
                        lv = lv.view(Bv, Tv, lvC, lv.shape[2], lv.shape[3])
                        lv = lv.permute(0, 2, 1, 3, 4)
                        val_loss += diffusion(lv, interpo_rate=args.interpo_rate).item()
                val_loss /= len(val_loader)
                diffusion.train()

                wandb.log({'val/diff_loss': val_loss, 'val/step': step})

                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(diffusion.state_dict(), args.output_ckpt)
                    pbar.write(f'  step {step}: best saved (val={val_loss:.6f})')

            if step % args.save_interval == 0:
                p = args.output_ckpt.replace('.pt', f'_step{step}.pt')
                torch.save(diffusion.state_dict(), p)

    pbar.close()
    elapsed = time.time() - start_time
    print(f'Diffusion done: {step} iters in {elapsed/60:.1f} min, best_val={best_val:.6f}')
    return diffusion


# ─── main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='V', choices=['V', 'D'])
    parser.add_argument('--stage', type=int, default=1,
                        help='For CAESAR-D: 1=keyframe VAE, 2=diffusion')
    parser.add_argument('--n_frame', type=int, default=None)
    parser.add_argument('--interpo_rate', type=int, default=3,
                        help='Keyframe interval (paper: 3)')
    parser.add_argument('--data_path', type=str,
                        default='/workspace/Data/lysozyme_processed/lysozyme_train_nf16.npz')
    parser.add_argument('--test_data_path', type=str,
                        default='/workspace/Data/lysozyme_processed/lysozyme_test_nf16.npz')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--vae_ckpt_path', type=str, default=None,
                        help='For D Stage 2: path to fine-tuned VAE checkpoint')
    parser.add_argument('--output_ckpt', type=str, default=None)
    parser.add_argument('--iterations', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Accumulate N batches before optimizer step (effective_batch = bs * N)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_rate', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--log_interval', type=int, default=200)
    parser.add_argument('--val_interval', type=int, default=2000)
    parser.add_argument('--save_interval', type=int, default=10000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--wandb_project', type=str, default='caesar-finetune')
    parser.add_argument('--train_samples', type=int, default=-1)
    parser.add_argument('--val_samples', type=int, default=-1)
    parser.add_argument('--no_wandb', action='store_true')
    args = parser.parse_args()

    # defaults per model type
    if args.n_frame is None:
        args.n_frame = 8 if args.model_type == 'V' else 16
    base_ckpt = '/workspace/AIForCompression/checkpoints/caesar'
    m = 'v' if args.model_type == 'V' else 'd'
    if args.ckpt_path is None:
        args.ckpt_path = f'{base_ckpt}/caesar_{m}.pt'
    if args.output_ckpt is None:
        tag = 'diff' if (args.model_type == 'D' and args.stage == 2) else 'vae'
        args.output_ckpt = f'{base_ckpt}/caesar_{m}_tuning_lysozyme_{tag}.pt'
    if args.vae_ckpt_path is None and args.model_type == 'D' and args.stage == 2:
        args.vae_ckpt_path = f'{base_ckpt}/caesar_{m}_tuning_lysozyme_vae.pt'

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"CAESAR-{args.model_type} Stage {args.stage} | device={device} "
          f"n_frame={args.n_frame} interpo_rate={args.interpo_rate}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Output:     {args.output_ckpt}")

    # wandb
    run_name = f"CAESAR-{args.model_type}_S{args.stage}_lysozyme_lr{args.lr}_b{args.batch_size}"
    if args.no_wandb:
        wandb.init(mode="disabled")
    else:
        try:
            wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        except Exception as e:
            print(f"wandb init failed ({e}), continuing without wandb")
            wandb.init(mode="disabled")

    # ── data ──
    train_ds = SimpleTemporalDataset(args.data_path, args.n_frame, train=True)
    val_ds = SimpleTemporalDataset(args.test_data_path, args.n_frame, train=False)
    if args.train_samples > 0:
        train_ds.S = min(train_ds.S, args.train_samples)
        train_ds.total = train_ds.V * train_ds.S * train_ds.t_samples
    if args.val_samples > 0:
        val_ds.S = min(val_ds.S, args.val_samples)
        val_ds.total = val_ds.V * val_ds.S * val_ds.t_samples

    print(f"Train: {len(train_ds)} items | Val: {len(val_ds)} items")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # ── model & training ──
    if args.model_type == 'V':
        model = load_caesar_v(args.ckpt_path, device)
        n = sum(p.numel() for p in model.parameters())
        print(f"CAESAR-V params: {n:,}")
        finetune_vae(model, train_loader, val_loader, args, device, label="CAESAR-V")

    elif args.model_type == 'D':
        if args.stage == 1:
            vae = load_caesar_d_vae(args.ckpt_path, device)
            n = sum(p.numel() for p in vae.parameters())
            print(f"CAESAR-D keyframe VAE params: {n:,}")
            vae = finetune_vae(vae, train_loader, val_loader, args, device,
                              label="CAESAR-D-VAE")
            # Save as vae-only for stage 2
            torch.save(vae.state_dict(), args.output_ckpt)

        elif args.stage == 2:
            vae = load_caesar_d_vae(args.vae_ckpt_path, device)
            diffusion = load_caesar_d_diffusion(args.ckpt_path, device)
            n_v = sum(p.numel() for p in vae.parameters())
            n_d = sum(p.numel() for p in diffusion.parameters())
            print(f"CAESAR-D VAE: {n_v:,}  Diffusion: {n_d:,}  Total: {n_v+n_d:,}")
            diffusion = finetune_diffusion(diffusion, vae, train_loader, val_loader, args, device)
            # Save full checkpoint
            torch.save({"vae": vae.state_dict(), "diffusion": diffusion.state_dict()},
                       args.output_ckpt.replace('_diff', ''))

    wandb.finish()
    print("Done!")


if __name__ == '__main__':
    main()
