#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ERA5_OBJECTIVE_DATA:-/workspace/Data/ERA5/finetune_processed/era5_test.npy}"
OUTPUT_DIR="${CAESAR_100K_EB_DIR:-$ROOT/unified_results/caesar_era5_daily_v_100k_eb_compare}"
LOG_DIR="${CAESAR_100K_EB_LOG_DIR:-$ROOT/logs/caesar_era5_daily_v_100k_eb_compare}"
ORIGINAL="$ROOT/checkpoints/caesar/caesar_v.pt"
FINETUNED="$ROOT/checkpoints/caesar_era5_daily_v_full_100k/rd_lr3em5_lam3em5_update100000.pt"
ORIGINAL_NO_PCA="$ROOT/unified_results/caesar_era5_daily_cadence_real_codec/v_original/summary.json"
FINETUNED_NO_PCA="$ROOT/unified_results/caesar_era5_daily_v_full_100k_real_codec/scratch_total100k/summary.json"

source /workspace/ai4cp/bin/activate
mkdir -p "$OUTPUT_DIR/raw" "$LOG_DIR"
test -f "$DATA"
test -f "$ORIGINAL"
test -f "$FINETUNED"
test -f "$ORIGINAL_NO_PCA"
test -f "$FINETUNED_NO_PCA"

python - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
    raise SystemExit(f"Expected 8 visible GPUs, found {torch.cuda.device_count()}")
PY

eb_name() {
  local eb="$1"
  printf '%s' "${eb//./p}"
}

run_eval() {
  local gpu="$1"
  local variant="$2"
  local checkpoint="$3"
  local eb="$4"
  local name="${variant}_eb$(eb_name "$eb")"
  local result_dir="$OUTPUT_DIR/raw/$name"

  CUDA_VISIBLE_DEVICES="$gpu" python -u "$ROOT/scripts/run_dataset_compression.py" \
    --dataset era5_npy \
    --data_root "$DATA" \
    --output_dir "$result_dir" \
    --models caesar_v \
    --max_samples 16 \
    --max_channels 268 \
    --resolution 240 240 \
    --caesar_ckpt_dir "$checkpoint" \
    --caesar_norm_type mean_range \
    --caesar_eb "$eb" \
    --caesar_num_windows 2 \
    --batch_size 64 \
    --no_lpips \
    >"$LOG_DIR/$name.log" 2>&1

  python - "$result_dir/summary.json" <<'PY'
import json
import sys
rows = json.load(open(sys.argv[1], encoding="utf-8"))
if len([row for row in rows if "error" not in row]) != 1:
    raise SystemExit(f"Invalid result: {sys.argv[1]}")
PY
}

run_queue() {
  local gpu="$1"
  local variant="$2"
  local checkpoint="$3"
  shift 3
  for eb in "$@"; do
    run_eval "$gpu" "$variant" "$checkpoint" "$eb"
  done
}

pids=()
run_queue 2 original "$ORIGINAL" 0.1 0.003 0.0001 & pids+=("$!")
run_queue 3 original "$ORIGINAL" 0.03 0.001 & pids+=("$!")
run_queue 4 original "$ORIGINAL" 0.01 0.0003 & pids+=("$!")
run_queue 5 finetuned "$FINETUNED" 0.1 0.003 0.0001 & pids+=("$!")
run_queue 6 finetuned "$FINETUNED" 0.03 0.001 & pids+=("$!")
run_queue 7 finetuned "$FINETUNED" 0.01 0.0003 & pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" != 0 ]]; then
  exit "$failed"
fi

python "$ROOT/scripts/plot_caesar_era5_100k_eb_compare.py" \
  --result-root "$OUTPUT_DIR" \
  --original-no-pca "$ORIGINAL_NO_PCA" \
  --finetuned-no-pca "$FINETUNED_NO_PCA"
touch "$OUTPUT_DIR/complete"
