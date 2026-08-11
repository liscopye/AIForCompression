#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ERA5_SHARD_DIR:-/workspace/Data/ERA5/hourly_center512_shards_20240301_90d}"
VAE="$ROOT/checkpoints/caesar_era5_vd_lowrate_100k/d_s1_lr1em5_lam1em3_full100k_update100000.pt"
BASE="$ROOT/checkpoints/caesar/caesar_d.pt"
OUTPUT_DIR="${CAESAR_D_LAM1EM3_RECOVERY_DIR:-$ROOT/checkpoints/caesar_era5_d_lam1em3_recovery_pilot}"
LOG_DIR="${CAESAR_D_LAM1EM3_RECOVERY_LOG_DIR:-$ROOT/logs/caesar_era5_d_lam1em3_recovery_pilot}"

source /workspace/ai4cp/bin/activate
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
test -f "$VAE"
test -f "$BASE"
wandb login --verify >/dev/null

{
  date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
  printf 'objective=recover_quality_while_retaining_lam1em3_low_rate\n'
  printf 'data_dir=%s\n' "$DATA_DIR"
  printf 'stage1_decoder_updates=10000\nstage2_updates=5000\n'
  sha256sum "$VAE" "$BASE"
} >"$OUTPUT_DIR/source_manifest.txt"

data_args=(
  --data_backend npy_shards
  --data_dir "$DATA_DIR"
  --train_timesteps 1776
  --val_timesteps 384
  --n_frame 16
  --interpo_rate 3
  --frame_step 24
  --temporal_stride 1
  --netcdf_val_channel_stride 16
  --netcdf_max_open_file_pairs 8
  --train_size 256
  --batch_size 32
  --val_batch_size 32
  --num_workers 4
  --prefetch_factor 2
  --norm_type mean_range
  --wandb_project caesar-era5-hourly-tuning
  --require_wandb
  --device cuda:0
)

run_decoder() {
  local name="lam1em3_decoder_lr3em4_10k"
  CUDA_VISIBLE_DEVICES=5 python -u "$ROOT/scripts/finetune_caesar_era5.py" \
    "${data_args[@]}" \
    --model_type D \
    --stage 1 \
    --ckpt_path "$VAE" \
    --output_ckpt "$OUTPUT_DIR/$name.pt" \
    --iterations 10000 \
    --trainable_scope decoder \
    --rate_mode bpp \
    --distortion_domain normalized \
    --lambda_rate 1e-3 \
    --lr 3e-4 \
    --warmup_updates 250 \
    --log_interval 50 \
    --val_interval 2500 \
    --save_interval 10000 \
    --milestone_steps 500 1000 2500 5000 7500 10000 \
    --wandb_group d-lam1em3-decoder-recovery \
    --wandb_run_name "$name" \
    --wandb_tags era5 caesar-d stage1 decoder-only frozen-rate lam1em3 recovery \
    >"$LOG_DIR/$name.log" 2>&1
  touch "$OUTPUT_DIR/$name.done"
}

run_stage2() {
  local gpu="$1"
  local lr="$2"
  local tag="$3"
  local name="lam1em3_stage2_${tag}_5k"
  CUDA_VISIBLE_DEVICES="$gpu" python -u "$ROOT/scripts/finetune_caesar_era5.py" \
    "${data_args[@]}" \
    --model_type D \
    --stage 2 \
    --ckpt_path "$BASE" \
    --vae_ckpt_path "$VAE" \
    --output_ckpt "$OUTPUT_DIR/$name.pt" \
    --iterations 5000 \
    --gradient_accumulation_steps 2 \
    --diffusion_steps 32 \
    --lr "$lr" \
    --warmup_updates 250 \
    --log_interval 25 \
    --val_interval 500 \
    --save_interval 5000 \
    --milestone_steps 50 100 250 500 1000 2000 5000 \
    --wandb_group d-lam1em3-matching-stage2 \
    --wandb_run_name "$name" \
    --wandb_tags era5 caesar-d stage2 matching-latent lam1em3 pilot \
    >"$LOG_DIR/$name.log" 2>&1
  touch "$OUTPUT_DIR/$name.done"
}

pids=()
cleanup() {
  local pid
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

run_stage2 3 3e-5 lr3em5 & pids+=("$!")
run_stage2 4 1e-4 lr1em4 & pids+=("$!")
run_decoder & pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
pids=()
date -u '+finished_utc=%Y-%m-%dT%H:%M:%SZ' >>"$OUTPUT_DIR/source_manifest.txt"
trap - INT TERM EXIT
exit "$failed"
