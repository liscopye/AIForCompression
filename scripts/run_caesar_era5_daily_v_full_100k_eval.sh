#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ERA5_OBJECTIVE_DATA:-/workspace/Data/ERA5/finetune_processed/era5_test.npy}"
CKPT_DIR="${CAESAR_DAILY_FULL_DIR:-$ROOT/checkpoints/caesar_era5_daily_v_full_100k}"
OUTPUT_DIR="${CAESAR_DAILY_FULL_EVAL_DIR:-$ROOT/unified_results/caesar_era5_daily_v_full_100k_real_codec}"
LOG_DIR="${CAESAR_DAILY_FULL_EVAL_LOG_DIR:-$ROOT/logs/caesar_era5_daily_v_full_100k_eval}"

source /workspace/ai4cp/bin/activate
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
test -f "$DATA"

python - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
    raise SystemExit(
        f"Expected 8 visible CUDA GPUs, found {torch.cuda.device_count()}; refusing to start."
    )
PY

run_eval() {
  local gpu="$1"
  local name="$2"
  local checkpoint="$3"
  local result_dir="$OUTPUT_DIR/$name"

  test -f "$checkpoint"
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
    --caesar_eb 0.001 \
    --caesar_num_windows 2 \
    --batch_size 64 \
    --caesar_no_pca \
    --no_lpips \
    >"$LOG_DIR/$name.log" 2>&1

  python - "$result_dir/summary.json" <<'PY'
import json
import sys

rows = json.load(open(sys.argv[1], encoding="utf-8"))
valid = [row for row in rows if "error" not in row]
if len(valid) != 1:
    raise SystemExit(f"Expected one valid result in {sys.argv[1]}, found {len(valid)}")
PY
  touch "$OUTPUT_DIR/$name.done"
}

scratch="$CKPT_DIR/rd_lr3em5_lam3em5"
continued="$CKPT_DIR/rd_from10k_add90k_lr3em5"

pids=()
(
  run_eval 2 scratch_total25k "${scratch}_update25000.pt"
  run_eval 2 scratch_total100k "${scratch}_update100000.pt"
) & pids+=("$!")
run_eval 3 scratch_total50k "${scratch}_update50000.pt" & pids+=("$!")
run_eval 4 scratch_total75k "${scratch}_update75000.pt" & pids+=("$!")
(
  run_eval 5 continued_total25k "${continued}_update15000.pt"
  run_eval 5 continued_total100k "${continued}_update90000.pt"
) & pids+=("$!")
run_eval 6 continued_total50k "${continued}_update40000.pt" & pids+=("$!")
run_eval 7 continued_total75k "${continued}_update65000.pt" & pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
exit "$failed"
