#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ERA5_HOURLY_SHARD_DIR:-/workspace/Data/ERA5/hourly_center512_shards_20240301_90d}"
OUTPUT_DIR="${CAESAR_QUALITY_CKPT_DIR:-$ROOT/checkpoints/caesar_era5_hourly_quality_sweep}"
LOG_DIR="${CAESAR_QUALITY_LOG_DIR:-$ROOT/logs/caesar_era5_hourly_quality_sweep}"
EXPECTED_DAYS="${ERA5_EXPECTED_DAYS:-90}"

actual_days=$(find "$DATA_DIR" -maxdepth 1 -name '*_hourly.npy' -type f 2>/dev/null | wc -l)
if [[ "$actual_days" -lt "$EXPECTED_DAYS" ]]; then
  echo "Refusing to start: found $actual_days/$EXPECTED_DAYS completed shards in $DATA_DIR" >&2
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
  --iterations 5000
  --rate_mode bpp
  --warmup_updates 1000
  --log_interval 50
  --val_interval 500
  --save_interval 5000
  --milestone_steps 500 1000 2000 5000
  --norm_type mean_range
  --wandb_project caesar-era5-hourly-tuning
  --wandb_group hourly-90d-stage1-quality-5k-grid
  --wandb_tags era5 hourly center512 stage1 quality-sweep
  --require_wandb
  --device cuda:0
)

run_config() {
  local gpu="$1"
  local model="$2"
  local name="$3"
  local lr="$4"
  local lambda_rate="$5"
  local checkpoint="$ROOT/checkpoints/caesar/caesar_${model,,}.pt"
  local done_file="$OUTPUT_DIR/$name.done"

  if [[ -f "$done_file" ]]; then
    echo "GPU $gpu: skipping completed $name"
    return
  fi

  CUDA_VISIBLE_DEVICES="$gpu" python -u "$ROOT/scripts/finetune_caesar_era5.py" \
    "${common[@]}" \
    --model_type "$model" \
    --lr "$lr" \
    --lambda_rate "$lambda_rate" \
    --ckpt_path "$checkpoint" \
    --output_ckpt "$OUTPUT_DIR/$name.pt" \
    --wandb_run_name "$name" \
    >"$LOG_DIR/$name.log" 2>&1
  touch "$done_file"
}

launch_queue() {
  local gpu="$1"
  shift
  (
    for encoded in "$@"; do
      IFS=, read -r model name lr lambda_rate <<<"$encoded"
      echo "GPU $gpu: starting $name"
      run_config "$gpu" "$model" "$name" "$lr" "$lambda_rate"
      echo "GPU $gpu: finished $name"
    done
  ) &
  pids+=("$!")
  names+=("gpu${gpu}")
}

pids=()
names=()
launch_queue 2 \
  "V,vq_lr1e5_lam0,1e-5,0" \
  "V,vq_lr1e5_lam1e6,1e-5,1e-6" \
  "D,dq_lr1e5_lam0,1e-5,0"
launch_queue 3 \
  "V,vq_lr1e5_lam3e6,1e-5,3e-6" \
  "V,vq_lr3e5_lam0,3e-5,0" \
  "D,dq_lr1e5_lam1e6,1e-5,1e-6"
launch_queue 4 \
  "V,vq_lr3e5_lam1e6,3e-5,1e-6" \
  "V,vq_lr3e5_lam3e6,3e-5,3e-6" \
  "D,dq_lr1e5_lam3e6,1e-5,3e-6"
launch_queue 5 \
  "V,vq_lr1e4_lam0,1e-4,0" \
  "V,vq_lr1e4_lam3e6,1e-4,3e-6" \
  "D,dq_lr3e5_lam0,3e-5,0"
launch_queue 6 \
  "D,dq_lr3e5_lam1e6,3e-5,1e-6" \
  "D,dq_lr3e5_lam3e6,3e-5,3e-6"
launch_queue 7 \
  "D,dq_lr1e4_lam0,1e-4,0" \
  "D,dq_lr1e4_lam3e6,1e-4,3e-6"

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
