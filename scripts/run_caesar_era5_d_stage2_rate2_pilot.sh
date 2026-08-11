#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ERA5_SHARD_DIR:-/workspace/Data/ERA5/hourly_center512_shards_20240301_90d}"
VAE="$ROOT/checkpoints/caesar_era5_vd_lowrate_100k/d_s1_lr1em5_lam3em4_full100k_update100000.pt"
BASE="$ROOT/checkpoints/caesar/caesar_d.pt"
OUTPUT_DIR="${CAESAR_D_STAGE2_RATE2_DIR:-$ROOT/checkpoints/caesar_era5_d_stage2_rate2_pilot}"
LOG_DIR="${CAESAR_D_STAGE2_RATE2_LOG_DIR:-$ROOT/logs/caesar_era5_d_stage2_rate2_pilot}"

source /workspace/ai4cp/bin/activate
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
test -f "$VAE"
test -f "$BASE"
wandb login --verify >/dev/null

{
  date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
  printf 'objective=matched_interpo_rate_2_stage2_screen\n'
  printf 'data_dir=%s\n' "$DATA_DIR"
  printf 'n_frame=16\ninterpo_rate=2\ncondition_frames=8\npredicted_frames=8\n'
  printf 'iterations=5000\ndiffusion_steps=32\neffective_batch=64\n'
  sha256sum "$VAE" "$BASE"
} >"$OUTPUT_DIR/source_manifest.txt"

common=(
  --model_type D
  --stage 2
  --data_backend npy_shards
  --data_dir "$DATA_DIR"
  --train_timesteps 1776
  --val_timesteps 384
  --n_frame 16
  --interpo_rate 2
  --frame_step 24
  --temporal_stride 1
  --netcdf_val_channel_stride 16
  --netcdf_max_open_file_pairs 8
  --train_size 256
  --batch_size 32
  --gradient_accumulation_steps 2
  --val_batch_size 32
  --num_workers 4
  --prefetch_factor 2
  --iterations 5000
  --diffusion_steps 32
  --warmup_updates 250
  --log_interval 25
  --val_interval 500
  --save_interval 5000
  --milestone_steps 50 100 250 500 1000 2000 5000
  --norm_type mean_range
  --ckpt_path "$BASE"
  --vae_ckpt_path "$VAE"
  --wandb_project caesar-era5-hourly-tuning
  --wandb_group d-stage2-interpo-rate2-pilot
  --wandb_tags era5 caesar-d stage2 interpo-rate-2 matching-latent pilot
  --require_wandb
  --device cuda:0
)

run_one() {
  local gpu="$1"
  local lr="$2"
  local tag="$3"
  local name="rate2_${tag}_5k"
  CUDA_VISIBLE_DEVICES="$gpu" python -u "$ROOT/scripts/finetune_caesar_era5.py" \
    "${common[@]}" \
    --lr "$lr" \
    --output_ckpt "$OUTPUT_DIR/$name.pt" \
    --wandb_run_name "$name" \
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

run_one 6 3e-5 lr3em5 & pids+=("$!")
run_one 7 1e-4 lr1em4 & pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
pids=()
date -u '+finished_utc=%Y-%m-%dT%H:%M:%SZ' >>"$OUTPUT_DIR/source_manifest.txt"
trap - INT TERM EXIT
exit "$failed"
