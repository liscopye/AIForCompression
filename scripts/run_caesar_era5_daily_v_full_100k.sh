#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ERA5_SHARD_DIR:-/workspace/Data/ERA5/hourly_center512_shards_20240301_90d}"
OUTPUT_DIR="${CAESAR_DAILY_FULL_DIR:-$ROOT/checkpoints/caesar_era5_daily_v_full_100k}"
LOG_DIR="${CAESAR_DAILY_FULL_LOG_DIR:-$ROOT/logs/caesar_era5_daily_v_full_100k}"
ORIGINAL="$ROOT/checkpoints/caesar/caesar_v.pt"
CONTINUED="$ROOT/checkpoints/caesar_era5_daily_v_continuation/rd_cont_lr3em5_from2k_add8k_update8000.pt"

source /workspace/ai4cp/bin/activate
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
test -f "$ORIGINAL"
test -f "$CONTINUED"
wandb login --verify >/dev/null

python - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
    raise SystemExit(
        f"Expected 8 visible CUDA GPUs, found {torch.cuda.device_count()}; refusing to start."
    )
PY

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
  --rate_mode bpp
  --distortion_domain normalized
  --warmup_updates 500
  --log_interval 100
  --val_interval 10000
  --save_interval 25000
  --norm_type mean_range
  --wandb_project caesar-era5-hourly-tuning
  --wandb_group daily-cadence-v-full-100k
  --wandb_tags era5 daily-cadence frame-step-24 normalized-distortion full-finetune paper-scale
  --require_wandb
  --device cuda:0
)

run_one() {
  local gpu="$1"
  local name="$2"
  local checkpoint="$3"
  local lr="$4"
  local lambda_rate="$5"
  local iterations="${6:-100000}"
  local log="$LOG_DIR/$name.log"
  local milestones=(10000 25000 50000 75000 100000)
  if [[ "$iterations" == "90000" ]]; then
    # These correspond to 25k, 50k, 75k, and 100k total updates.
    milestones=(15000 40000 65000 90000)
  fi

  echo "GPU $gpu: $name checkpoint=$checkpoint lr=$lr lambda=$lambda_rate iterations=$iterations"
  CUDA_VISIBLE_DEVICES="$gpu" python -u "$ROOT/scripts/finetune_caesar_era5.py" \
    "${common[@]}" \
    --ckpt_path "$checkpoint" \
    --iterations "$iterations" \
    --milestone_steps "${milestones[@]}" \
    --lr "$lr" \
    --lambda_rate "$lambda_rate" \
    --output_ckpt "$OUTPUT_DIR/$name.pt" \
    --wandb_run_name "$name" \
    >"$log" 2>&1
  touch "$OUTPUT_DIR/$name.done"
}

pids=()
# This is the ERA5 configuration that passed the independent June codec test.
run_one 3 rd_lr3em5_lam3em5 "$ORIGINAL" 3e-5 3e-5 & pids+=("$!")
# Continue the best measured 10k RD point with the same validated hyperparameters.
run_one 7 rd_from10k_add90k_lr3em5 "$CONTINUED" 3e-5 3e-5 90000 & pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
exit "$failed"
