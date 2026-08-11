#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ERA5_HOURLY_SHARD_DIR:-/workspace/Data/ERA5/hourly_center512_shards_20240301_90d}"
OUTPUT_DIR="${CAESAR_HOURLY_CKPT_DIR:-$ROOT/checkpoints/caesar_era5_hourly_pilot}"
LOG_DIR="${CAESAR_HOURLY_LOG_DIR:-$ROOT/logs/caesar_era5_hourly_pilot}"
EXPECTED_DAYS="${ERA5_EXPECTED_DAYS:-90}"

actual_days=$(find "$DATA_DIR" -maxdepth 1 -name '*_hourly.npy' -type f 2>/dev/null | wc -l)
if [[ "$actual_days" -lt "$EXPECTED_DAYS" ]]; then
  echo "Refusing to start: found $actual_days/$EXPECTED_DAYS completed daily shards in $DATA_DIR" >&2
  exit 2
fi

wandb login --verify >/dev/null
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

common=(
  --stage 1
  --data_backend npy_shards
  --data_dir "$DATA_DIR"
  --train_timesteps 1920
  --val_timesteps 240
  --netcdf_val_channel_stride 8
  --netcdf_max_open_file_pairs 4
  --train_size 256
  --batch_size 32
  --val_batch_size 32
  --num_workers 4
  --prefetch_factor 2
  --iterations 10000
  --rate_mode bpp
  --warmup_updates 1000
  --log_interval 50
  --val_interval 1000
  --save_interval 10000
  --milestone_steps 100 500 1000 2000 5000 10000
  --wandb_project caesar-era5-hourly-tuning
  --wandb_group hourly-90d-stage1-10k-grid
  --wandb_tags era5 hourly center512 stage1 pilot
  --require_wandb
  --device cuda:0
)

run_config() {
  local gpu="$1"
  local model="$2"
  local name="$3"
  local lr="$4"
  local lambda_rate="$5"
  local norm="$6"
  local checkpoint
  if [[ "$model" == "V" ]]; then
    checkpoint="$ROOT/checkpoints/caesar/caesar_v.pt"
  else
    checkpoint="$ROOT/checkpoints/caesar/caesar_d.pt"
  fi

  CUDA_VISIBLE_DEVICES="$gpu" python -u "$ROOT/scripts/finetune_caesar_era5.py" \
    "${common[@]}" \
    --model_type "$model" \
    --lr "$lr" \
    --lambda_rate "$lambda_rate" \
    --norm_type "$norm" \
    --ckpt_path "$checkpoint" \
    --output_ckpt "$OUTPUT_DIR/$name.pt" \
    --wandb_run_name "$name" \
    >"$LOG_DIR/$name.log" 2>&1
}

pids=()
names=()
launch_queue() {
  local gpu="$1"
  shift
  (
    for encoded in "$@"; do
      IFS=, read -r model name lr lambda_rate norm <<<"$encoded"
      echo "GPU $gpu: starting $name"
      run_config "$gpu" "$model" "$name" "$lr" "$lambda_rate" "$norm"
      echo "GPU $gpu: finished $name"
    done
  ) &
  pids+=("$!")
  names+=("gpu${gpu}")
  echo "GPU $gpu queue PID $!"
}

launch_queue 0 \
  "V,v_lr3e6_lam1e4_mr,3e-6,1e-4,mean_range" \
  "V,v_lr1e5_lam3e4_mr,1e-5,3e-4,mean_range"
launch_queue 1 \
  "V,v_lr1e5_lam1e4_mr,1e-5,1e-4,mean_range" \
  "V,v_lr1e5_lam1e4_hw,1e-5,1e-4,mean_range_hw"
launch_queue 2 \
  "V,v_lr3e5_lam1e4_mr,3e-5,1e-4,mean_range" \
  "V,v_lr1e4_lam1e5_mr,1e-4,1e-5,mean_range"
launch_queue 3 \
  "V,v_lr1e4_lam1e4_mr,1e-4,1e-4,mean_range" \
  "V,v_lr1e4_lam1e4_hw,1e-4,1e-4,mean_range_hw"
launch_queue 4 \
  "D,d_lr3e6_lam1e4_mr,3e-6,1e-4,mean_range" \
  "D,d_lr1e5_lam3e4_mr,1e-5,3e-4,mean_range"
launch_queue 5 \
  "D,d_lr1e5_lam1e4_mr,1e-5,1e-4,mean_range" \
  "D,d_lr1e5_lam1e4_hw,1e-5,1e-4,mean_range_hw"
launch_queue 6 \
  "D,d_lr3e5_lam1e4_mr,3e-5,1e-4,mean_range" \
  "D,d_lr1e4_lam1e5_mr,1e-4,1e-5,mean_range"
launch_queue 7 \
  "D,d_lr1e4_lam1e4_mr,1e-4,1e-4,mean_range" \
  "D,d_lr1e4_lam1e4_hw,1e-4,1e-4,mean_range_hw"

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
