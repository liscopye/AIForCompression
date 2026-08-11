#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ERA5_SHARD_DIR:-/workspace/Data/ERA5/hourly_center512_shards_20240301_90d}"
BASE_DIR="${CAESAR_DAILY_BASE_DIR:-$ROOT/checkpoints/caesar_era5_daily_cadence_pilot}"
OUTPUT_DIR="${CAESAR_DAILY_CONT_DIR:-$ROOT/checkpoints/caesar_era5_daily_v_continuation}"
LOG_DIR="${CAESAR_DAILY_CONT_LOG_DIR:-$ROOT/logs/caesar_era5_daily_v_continuation}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
wandb login --verify >/dev/null

common=(
  --model_type V
  --stage 1
  --data_backend npy_shards
  --data_dir "$DATA_DIR"
  --train_timesteps 1776
  --val_timesteps 384
  --n_frame 8
  --frame_step 24
  --temporal_stride 8
  --netcdf_val_channel_stride 4
  --netcdf_max_open_file_pairs 8
  --train_size 256
  --batch_size 32
  --val_batch_size 32
  --num_workers 4
  --prefetch_factor 2
  --iterations 8000
  --rate_mode bpp
  --distortion_domain normalized
  --warmup_updates 250
  --log_interval 50
  --val_interval 1000
  --save_interval 8000
  --milestone_steps 1000 3000 8000
  --norm_type mean_range
  --wandb_project caesar-era5-hourly-tuning
  --wandb_group daily-cadence-v-from2k-add8k
  --wandb_tags era5 daily-cadence frame-step-24 normalized-distortion continuation
  --require_wandb
  --device cuda:0
)

run_one() {
  local gpu="$1"
  local branch="$2"
  local base_name="$3"
  local lr="$4"
  local lambda_rate="$5"
  local name="${branch}_cont_lr${lr//-/m}_from2k_add8k"
  local base_ckpt="$BASE_DIR/$base_name.pt"
  local log="$LOG_DIR/$name.log"

  if [[ ! -f "$base_ckpt" ]]; then
    echo "Missing base checkpoint: $base_ckpt" >&2
    return 1
  fi

  echo "GPU $gpu: $name base=$base_name lr=$lr lambda=$lambda_rate"
  CUDA_VISIBLE_DEVICES="$gpu" python -u "$ROOT/scripts/finetune_caesar_era5.py" \
    "${common[@]}" \
    --ckpt_path "$base_ckpt" \
    --lr "$lr" \
    --lambda_rate "$lambda_rate" \
    --output_ckpt "$OUTPUT_DIR/$name.pt" \
    --wandb_run_name "$name" \
    >"$log" 2>&1
  touch "$OUTPUT_DIR/$name.done"
}

pids=()
run_one 2 quality daily_v_lr3e5_lam3e6 3e-6 3e-6 & pids+=("$!")
run_one 3 quality daily_v_lr3e5_lam3e6 1e-5 3e-6 & pids+=("$!")
run_one 4 quality daily_v_lr3e5_lam3e6 3e-5 3e-6 & pids+=("$!")
run_one 5 rd daily_v_lr3e5_lam3e5 3e-6 3e-5 & pids+=("$!")
run_one 6 rd daily_v_lr3e5_lam3e5 1e-5 3e-5 & pids+=("$!")
run_one 7 rd daily_v_lr3e5_lam3e5 3e-5 3e-5 & pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
exit "$failed"
