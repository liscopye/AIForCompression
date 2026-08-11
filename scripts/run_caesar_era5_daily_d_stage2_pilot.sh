#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ERA5_SHARD_DIR:-/workspace/Data/ERA5/hourly_center512_shards_20240301_90d}"
D1_DIR="${CAESAR_DAILY_D1_CKPT_DIR:-$ROOT/checkpoints/caesar_era5_daily_d_stage1_pilot}"
OUTPUT_DIR="${CAESAR_DAILY_D2_CKPT_DIR:-$ROOT/checkpoints/caesar_era5_daily_d_stage2_pilot}"
LOG_DIR="${CAESAR_DAILY_D2_LOG_DIR:-$ROOT/logs/caesar_era5_daily_d_stage2_pilot}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
wandb login --verify >/dev/null

common=(
  --model_type D
  --stage 2
  --data_backend npy_shards
  --data_dir "$DATA_DIR"
  --train_timesteps 1776
  --val_timesteps 384
  --n_frame 16
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
  --iterations 1000
  --warmup_updates 250
  --log_interval 25
  --val_interval 500
  --save_interval 1000
  --milestone_steps 250 500 1000
  --norm_type mean_range
  --ckpt_path "$ROOT/checkpoints/caesar/caesar_d.pt"
  --wandb_project caesar-era5-hourly-tuning
  --wandb_group daily-cadence-d-stage2-pilot
  --wandb_tags era5 daily-cadence frame-step-24 caesar-d stage2
  --require_wandb
  --device cuda:0
)

run_one() {
  local gpu="$1"
  local name="$2"
  local vae="$3"
  local lr="$4"
  echo "GPU $gpu: $name lr=$lr vae=$(basename "$vae")"
  CUDA_VISIBLE_DEVICES="$gpu" python -u "$ROOT/scripts/finetune_caesar_era5.py" \
    "${common[@]}" \
    --vae_ckpt_path "$vae" \
    --lr "$lr" \
    --output_ckpt "$OUTPUT_DIR/$name.pt" \
    --wandb_run_name "$name" \
    >"$LOG_DIR/$name.log" 2>&1
  touch "$OUTPUT_DIR/$name.done"
}

quality_vae="$D1_DIR/daily_d1_lr3e5_lam3e6.pt"
lowrate_vae="$D1_DIR/daily_d1_lr3e5_lam3e5.pt"
pids=()
run_one 2 daily_d2_quality_lr3e7 "$quality_vae" 3e-7 & pids+=("$!")
run_one 3 daily_d2_quality_lr1e6 "$quality_vae" 1e-6 & pids+=("$!")
run_one 4 daily_d2_quality_lr3e6 "$quality_vae" 3e-6 & pids+=("$!")
run_one 5 daily_d2_lowrate_lr3e7 "$lowrate_vae" 3e-7 & pids+=("$!")
run_one 6 daily_d2_lowrate_lr1e6 "$lowrate_vae" 1e-6 & pids+=("$!")
run_one 7 daily_d2_lowrate_lr3e6 "$lowrate_vae" 3e-6 & pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
exit "$failed"
