#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ERA5_HOURLY_SHARD_DIR:-/workspace/Data/ERA5/hourly_center512_shards_20240301_90d}"
CKPT_DIR="${CAESAR_HOURLY_CKPT_DIR:-$ROOT/checkpoints/caesar_era5_hourly_pilot}"
SELECTED_DIR="${CAESAR_HOURLY_SELECTED_CKPT_DIR:-$ROOT/checkpoints/caesar_era5_hourly_selected}"
OUTPUT_DIR="$CKPT_DIR/packaged_d"
LOG_DIR="${CAESAR_HOURLY_STAGE2_LOG_DIR:-$ROOT/logs/caesar_era5_hourly_stage2_pilot}"
SOURCE_CKPT="$SELECTED_DIR/caesar_d.pt"
SELECTION_JSON="$SELECTED_DIR/selection.json"

if [[ ! -f "$SOURCE_CKPT" || ! -f "$SELECTION_JSON" ]]; then
  echo "Missing selected Stage-1 checkpoint: $SOURCE_CKPT" >&2
  exit 2
fi

readarray -t source_fields < <(python - "$SELECTION_JSON" <<'PY'
import json
import sys
selected = json.load(open(sys.argv[1]))["models"]["D"]["selected"]
print(selected["name"])
print(selected["norm_type"])
PY
)
SOURCE_NAME="${source_fields[0]}"
NORM_TYPE="${source_fields[1]}"
NORM_TAG="mr"
if [[ "$NORM_TYPE" == "mean_range_hw" ]]; then
  NORM_TAG="hw"
fi

wandb login --verify >/dev/null
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

run_one() {
  local gpu="$1"
  local name="$2"
  local lr="$3"

  CUDA_VISIBLE_DEVICES="$gpu" python -u "$ROOT/scripts/finetune_caesar_era5.py" \
    --model_type D \
    --stage 2 \
    --data_backend npy_shards \
    --data_dir "$DATA_DIR" \
    --train_timesteps 1920 \
    --val_timesteps 240 \
    --netcdf_val_channel_stride 32 \
    --train_size 256 \
    --batch_size 32 \
    --gradient_accumulation_steps 2 \
    --val_batch_size 32 \
    --num_workers 4 \
    --prefetch_factor 2 \
    --iterations 5000 \
    --lr "$lr" \
    --norm_type "$NORM_TYPE" \
    --warmup_updates 500 \
    --log_interval 25 \
    --val_interval 500 \
    --save_interval 5000 \
    --milestone_steps 50 100 250 500 1000 2000 5000 \
    --ckpt_path "$SOURCE_CKPT" \
    --vae_ckpt_path "$SOURCE_CKPT" \
    --output_ckpt "$OUTPUT_DIR/$name.pt" \
    --wandb_project caesar-era5-hourly-tuning \
    --wandb_group hourly-90d-stage2-5k-grid \
    --wandb_run_name "$name" \
    --wandb_tags era5 hourly center512 stage2 pilot \
    --require_wandb \
    --device cuda:0 \
    >"$LOG_DIR/$name.log" 2>&1
}

jobs=(
  "d_s2_${NORM_TAG}_lr3e7,3e-7"
  "d_s2_${NORM_TAG}_lr1e6,1e-6"
  "d_s2_${NORM_TAG}_lr3e6,3e-6"
  "d_s2_${NORM_TAG}_lr1e5,1e-5"
  "d_s2_${NORM_TAG}_lr1e4,1e-4"
)

pids=()
names=()
for gpu in "${!jobs[@]}"; do
  IFS=, read -r name lr <<<"${jobs[$gpu]}"
  run_one "$gpu" "$name" "$lr" &
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
