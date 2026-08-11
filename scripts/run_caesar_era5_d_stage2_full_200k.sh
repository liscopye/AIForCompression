#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ERA5_SHARD_DIR:-/workspace/Data/ERA5/hourly_center512_shards_20240301_90d}"
VAE="$ROOT/checkpoints/caesar_era5_vd_lowrate_100k/d_s1_lr1em5_lam3em4_full100k_update100000.pt"
BASE="$ROOT/checkpoints/caesar/caesar_d.pt"
OUTPUT_DIR="${CAESAR_D_STAGE2_FULL_DIR:-$ROOT/checkpoints/caesar_era5_d_stage2_full_200k}"
LOG_DIR="${CAESAR_D_STAGE2_FULL_LOG_DIR:-$ROOT/logs/caesar_era5_d_stage2_full_200k}"
GPU="${CAESAR_D_STAGE2_FULL_GPU:-2}"
NAME="lam3em4_stage2_from_original_lr1em4_200k"

source /workspace/ai4cp/bin/activate
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
test -f "$VAE"
test -f "$BASE"
wandb login --verify >/dev/null

{
  date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
  printf 'data_dir=%s\n' "$DATA_DIR"
  printf 'objective=paper_scale_caesar_d_matching_diffusion\n'
  printf 'stage1_latent_source=%s\n' "$VAE"
  printf 'diffusion_source=%s\n' "$BASE"
  printf 'iterations=200000\n'
  printf 'lr=1e-4\n'
  printf 'diffusion_steps=32\n'
  printf 'micro_batch=32\n'
  printf 'gradient_accumulation_steps=2\n'
  printf 'effective_batch=64\n'
  sha256sum "$VAE" "$BASE"
} >"$OUTPUT_DIR/source_manifest.txt"

CUDA_VISIBLE_DEVICES="$GPU" python -u "$ROOT/scripts/finetune_caesar_era5.py" \
  --model_type D \
  --stage 2 \
  --data_backend npy_shards \
  --data_dir "$DATA_DIR" \
  --train_timesteps 1776 \
  --val_timesteps 384 \
  --n_frame 16 \
  --frame_step 24 \
  --temporal_stride 1 \
  --netcdf_val_channel_stride 16 \
  --netcdf_max_open_file_pairs 8 \
  --train_size 256 \
  --batch_size 32 \
  --gradient_accumulation_steps 2 \
  --val_batch_size 32 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --iterations 200000 \
  --diffusion_steps 32 \
  --lr 1e-4 \
  --warmup_updates 250 \
  --log_interval 100 \
  --val_interval 5000 \
  --save_interval 25000 \
  --milestone_steps 50 100 250 500 1000 2000 5000 10000 25000 50000 75000 100000 125000 150000 175000 200000 \
  --norm_type mean_range \
  --ckpt_path "$BASE" \
  --vae_ckpt_path "$VAE" \
  --output_ckpt "$OUTPUT_DIR/$NAME.pt" \
  --wandb_project caesar-era5-hourly-tuning \
  --wandb_group d-matching-stage2-full-200k \
  --wandb_run_name "$NAME" \
  --wandb_tags era5 caesar-d stage2 diffusion matching-latent paper-scale full-200k \
  --require_wandb \
  --device cuda:0 \
  >"$LOG_DIR/$NAME.log" 2>&1

touch "$OUTPUT_DIR/$NAME.done"
date -u '+finished_utc=%Y-%m-%dT%H:%M:%SZ' >>"$OUTPUT_DIR/source_manifest.txt"
