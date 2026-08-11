#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ERA5_SHARD_DIR:-/workspace/Data/ERA5/hourly_center512_shards_20240301_90d}"
OUTPUT_DIR="${CAESAR_DAILY_D1_CKPT_DIR:-$ROOT/checkpoints/caesar_era5_daily_d_stage1_pilot}"
LOG_DIR="${CAESAR_DAILY_D1_LOG_DIR:-$ROOT/logs/caesar_era5_daily_d_stage1_pilot}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
wandb login --verify >/dev/null

common=(
  --model_type D
  --stage 1
  --data_backend npy_shards
  --data_dir "$DATA_DIR"
  --train_timesteps 1776
  --val_timesteps 384
  --n_frame 16
  --frame_step 24
  --temporal_stride 1
  --netcdf_val_channel_stride 4
  --netcdf_max_open_file_pairs 8
  --train_size 256
  --batch_size 32
  --val_batch_size 32
  --num_workers 4
  --prefetch_factor 2
  --iterations 2000
  --rate_mode bpp
  --distortion_domain normalized
  --warmup_updates 500
  --log_interval 50
  --val_interval 1000
  --save_interval 2000
  --milestone_steps 500 1000 2000
  --norm_type mean_range
  --ckpt_path "$ROOT/checkpoints/caesar/caesar_d.pt"
  --wandb_project caesar-era5-hourly-tuning
  --wandb_group daily-cadence-d-stage1-normalized-pilot
  --wandb_tags era5 daily-cadence frame-step-24 normalized-distortion caesar-d stage1
  --require_wandb
  --device cuda:0
)

run_one() {
  local gpu="$1"
  local name="$2"
  local lr="$3"
  local lambda_rate="$4"
  echo "GPU $gpu: $name lr=$lr lambda=$lambda_rate"
  CUDA_VISIBLE_DEVICES="$gpu" python -u "$ROOT/scripts/finetune_caesar_era5.py" \
    "${common[@]}" \
    --lr "$lr" \
    --lambda_rate "$lambda_rate" \
    --output_ckpt "$OUTPUT_DIR/$name.pt" \
    --wandb_run_name "$name" \
    >"$LOG_DIR/$name.log" 2>&1
  touch "$OUTPUT_DIR/$name.done"
}

pids=()
run_one 2 daily_d1_lr1e5_lam3e6 1e-5 3e-6 & pids+=("$!")
run_one 3 daily_d1_lr1e5_lam1e5 1e-5 1e-5 & pids+=("$!")
run_one 4 daily_d1_lr1e5_lam3e5 1e-5 3e-5 & pids+=("$!")
run_one 5 daily_d1_lr3e5_lam3e6 3e-5 3e-6 & pids+=("$!")
run_one 6 daily_d1_lr3e5_lam1e5 3e-5 1e-5 & pids+=("$!")
run_one 7 daily_d1_lr3e5_lam3e5 3e-5 3e-5 & pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
exit "$failed"
