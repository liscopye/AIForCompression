#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ERA5_HOURLY_SHARD_DIR:-/workspace/Data/ERA5/hourly_center512_shards_20240301_90d}"
OUTPUT_DIR="${CAESAR_HOURLY_D_SMOKE_CKPT_DIR:-$ROOT/checkpoints/caesar_era5_hourly_day2_d_smoke}"
LOG_DIR="${CAESAR_HOURLY_D_SMOKE_LOG_DIR:-$ROOT/logs/caesar_era5_hourly_day2_d_smoke}"
SOURCE_CKPT="$ROOT/checkpoints/caesar/caesar_d.pt"

if [[ ! -f "$DATA_DIR/2024-03-02_hourly.npy" ]]; then
  echo "Missing completed day-2 shard: $DATA_DIR/2024-03-02_hourly.npy" >&2
  exit 2
fi

wandb login --verify >/dev/null
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

run_one() {
  local gpu="$1"
  local name="$2"
  local lr="$3"
  local lambda_rate="$4"
  local norm="$5"

  CUDA_VISIBLE_DEVICES="$gpu" python -u "$ROOT/scripts/finetune_caesar_era5.py" \
    --model_type D \
    --stage 1 \
    --data_backend npy_shards \
    --data_dir "$DATA_DIR" \
    --train_timesteps 32 \
    --val_timesteps 16 \
    --netcdf_val_channel_stride 8 \
    --train_size 256 \
    --batch_size 32 \
    --val_batch_size 32 \
    --num_workers 2 \
    --prefetch_factor 2 \
    --iterations 500 \
    --lr "$lr" \
    --lambda_rate "$lambda_rate" \
    --rate_mode bpp \
    --norm_type "$norm" \
    --warmup_updates 100 \
    --log_interval 10 \
    --val_interval 50 \
    --save_interval 500 \
    --milestone_steps 1 10 50 100 200 500 \
    --ckpt_path "$SOURCE_CKPT" \
    --output_ckpt "$OUTPUT_DIR/$name.pt" \
    --wandb_project caesar-era5-hourly-tuning \
    --wandb_group hourly-day2-realdata-d-stage1-smoke \
    --wandb_run_name "$name" \
    --wandb_tags era5 hourly center512 day2 smoke D stage1 \
    --require_wandb \
    --device cuda:0 \
    >"$LOG_DIR/$name.log" 2>&1
}

jobs=(
  "d_lr3e6_lam1e4_mr,3e-6,1e-4,mean_range"
  "d_lr1e5_lam1e4_mr,1e-5,1e-4,mean_range"
  "d_lr3e5_lam1e4_mr,3e-5,1e-4,mean_range"
  "d_lr1e4_lam1e4_mr,1e-4,1e-4,mean_range"
  "d_lr1e5_lam1e5_mr,1e-5,1e-5,mean_range"
  "d_lr1e5_lam3e4_mr,1e-5,3e-4,mean_range"
  "d_lr1e5_lam1e4_hw,1e-5,1e-4,mean_range_hw"
  "d_lr1e5_lam1e4_mm,1e-5,1e-4,min_max"
)

pids=()
names=()
for gpu in "${!jobs[@]}"; do
  IFS=, read -r name lr lambda_rate norm <<<"${jobs[$gpu]}"
  run_one "$gpu" "$name" "$lr" "$lambda_rate" "$norm" &
  pids+=("$!")
  names+=("$name")
  echo "GPU $gpu: started $name as PID $!"
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "finished ${names[$index]}"
  else
    echo "failed ${names[$index]}" >&2
    failed=1
  fi
done
exit "$failed"
