#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROBE="${ERA5_HOURLY_PROBE:-/workspace/Data/ERA5/hourly_center512_validation_probe.npy}"
CKPT_DIR="${CAESAR_HOURLY_CKPT_DIR:-$ROOT/checkpoints/caesar_era5_hourly_pilot}"
SELECTED_DIR="${CAESAR_HOURLY_SELECTED_CKPT_DIR:-$ROOT/checkpoints/caesar_era5_hourly_selected}"
OUTPUT_DIR="${CAESAR_HOURLY_EVAL_DIR:-$ROOT/unified_results/caesar_era5_hourly_pilot_eval}"
LOG_DIR="${CAESAR_HOURLY_STAGE2_EVAL_LOG_DIR:-$ROOT/logs/caesar_era5_hourly_stage2_eval}"
STEPS=(50 100 250 500 1000 2000 5000)
EBS=(0.1 0.003 0.0001)
SELECTION_JSON="$SELECTED_DIR/selection.json"

if [[ ! -f "$PROBE" || ! -f "$SELECTION_JSON" ]]; then
  echo "Missing held-out probe: $PROBE" >&2
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
CONFIGS=(
  "d_s2_${NORM_TAG}_lr3e7"
  "d_s2_${NORM_TAG}_lr1e6"
  "d_s2_${NORM_TAG}_lr3e6"
  "d_s2_${NORM_TAG}_lr1e5"
  "d_s2_${NORM_TAG}_lr1e4"
)
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

run_eval() {
  local gpu="$1"
  local name="$2"
  local checkpoint="$3"
  CUDA_VISIBLE_DEVICES="$gpu" python "$ROOT/scripts/run_dataset_compression.py" \
    --dataset era5_npy \
    --data_root "$PROBE" \
    --output_dir "$OUTPUT_DIR/$name" \
    --models caesar_d \
    --max_samples 64 \
    --max_channels 30 \
    --resolution 240 240 \
    --caesar_ckpt_dir "$checkpoint" \
    --caesar_norm_type "$NORM_TYPE" \
    --caesar_eb "${EBS[@]}" \
    --caesar_num_windows 4 \
    --batch_size 64 \
    --no_lpips \
    >"$LOG_DIR/$name.log" 2>&1
}

pids=()
for gpu in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$gpu]}"
  (
    for step in "${STEPS[@]}"; do
      name="${config}_update${step}"
      run_eval "$gpu" "$name" "$CKPT_DIR/packaged_d/$name.pt"
    done
  ) &
  pids+=("$!")
  echo "GPU $gpu: stage-2 evaluation queue ${CONFIGS[$gpu]} (PID $!)"
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
exit "$failed"
