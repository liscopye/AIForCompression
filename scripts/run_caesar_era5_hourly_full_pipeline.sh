#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARD_DIR="${ERA5_HOURLY_SHARD_DIR:-/workspace/Data/ERA5/hourly_center512_shards_20240301_90d}"
CKPT_DIR="${CAESAR_HOURLY_CKPT_DIR:-$ROOT/checkpoints/caesar_era5_hourly_pilot}"
EVAL_DIR="${CAESAR_HOURLY_EVAL_DIR:-$ROOT/unified_results/caesar_era5_hourly_pilot_eval}"
SELECTION_DIR="${CAESAR_HOURLY_SELECTION_DIR:-$ROOT/unified_results/caesar_era5_hourly_selection}"
SELECTED_DIR="${CAESAR_HOURLY_SELECTED_CKPT_DIR:-$ROOT/checkpoints/caesar_era5_hourly_selected}"

actual_days=$(find "$SHARD_DIR" -maxdepth 1 -name '*_hourly.npy' -type f | wc -l)
if [[ "$actual_days" -lt 90 ]]; then
  echo "Refusing to start full pipeline: found $actual_days/90 shards" >&2
  exit 2
fi

python "$ROOT/utils/audit_era5_hourly_shards.py" \
  --shard-dir "$SHARD_DIR" \
  --start-date 2024-03-01 \
  --days 90 \
  --train-timesteps 1920 \
  --val-timesteps 240 \
  --objective-start-date 2024-06-01 \
  --objective-timesteps 16 \
  --objective-array /workspace/Data/ERA5/finetune_processed/era5_test.npy \
  --objective-raw-dir /workspace/Data/ERA5/test \
  --mean-std-dir "$ROOT/models/CRA5/cra5/dataset"

bash "$ROOT/scripts/archive/caesar_experiments/run_caesar_era5_hourly_pilot.sh"
bash "$ROOT/scripts/archive/caesar_experiments/run_caesar_era5_hourly_pilot_eval.sh"

python "$ROOT/scripts/select_caesar_era5_hourly_checkpoint.py" \
  --eval-dir "$EVAL_DIR" \
  --checkpoint-dir "$CKPT_DIR" \
  --original-dir "$ROOT/checkpoints/caesar" \
  --output-dir "$SELECTION_DIR" \
  --selected-checkpoint-dir "$SELECTED_DIR"

bash "$ROOT/scripts/archive/caesar_experiments/run_caesar_era5_hourly_stage2_pilot.sh"
bash "$ROOT/scripts/run_caesar_era5_hourly_stage2_eval.sh"

# Re-run selection after adding Stage-2 candidates, then perform independent
# seven-point objective-v1 evaluation and final BD-rate comparison.
bash "$ROOT/scripts/run_caesar_era5_hourly_objective.sh"
