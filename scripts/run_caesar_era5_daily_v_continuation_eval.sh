#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ERA5_OBJECTIVE_DATA:-/workspace/Data/ERA5/finetune_processed/era5_test.npy}"
CKPT_DIR="${CAESAR_DAILY_CONT_DIR:-$ROOT/checkpoints/caesar_era5_daily_v_continuation}"
OUTPUT_DIR="${CAESAR_DAILY_CONT_EVAL_DIR:-$ROOT/unified_results/caesar_era5_daily_v_continuation_real_codec}"
LOG_DIR="${CAESAR_DAILY_CONT_EVAL_LOG_DIR:-$ROOT/logs/caesar_era5_daily_v_continuation_eval}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
test -f "$DATA"

if ! python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("CUDA is unavailable; refusing to run the codec evaluation on CPU.", file=sys.stderr)
    raise SystemExit(1)
if torch.cuda.device_count() < 8:
    print(
        f"Expected 8 visible GPUs, found {torch.cuda.device_count()}; refusing to start.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
then
  exit 2
fi

run_eval() {
  local gpu="$1"
  local name="$2"
  local checkpoint="$CKPT_DIR/${name}_update8000.pt"

  test -f "$checkpoint"
  CUDA_VISIBLE_DEVICES="$gpu" python -u "$ROOT/scripts/run_dataset_compression.py" \
    --dataset era5_npy \
    --data_root "$DATA" \
    --output_dir "$OUTPUT_DIR/$name" \
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
  touch "$OUTPUT_DIR/$name.done"
}

pids=()
run_eval 2 quality_cont_lr3em6_from2k_add8k & pids+=("$!")
run_eval 3 quality_cont_lr1em5_from2k_add8k & pids+=("$!")
run_eval 4 quality_cont_lr3em5_from2k_add8k & pids+=("$!")
run_eval 5 rd_cont_lr3em6_from2k_add8k & pids+=("$!")
run_eval 6 rd_cont_lr1em5_from2k_add8k & pids+=("$!")
run_eval 7 rd_cont_lr3em5_from2k_add8k & pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
exit "$failed"
