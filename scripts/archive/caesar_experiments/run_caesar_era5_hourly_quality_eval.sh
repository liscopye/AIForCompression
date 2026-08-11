#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROBE="${ERA5_HOURLY_PROBE:-/workspace/Data/ERA5/hourly_center512_validation_probe.npy}"
QUALITY_DIR="${CAESAR_QUALITY_CKPT_DIR:-$ROOT/checkpoints/caesar_era5_hourly_quality_sweep}"
PILOT_DIR="${CAESAR_HOURLY_CKPT_DIR:-$ROOT/checkpoints/caesar_era5_hourly_pilot}"
PACKAGED_DIR="$QUALITY_DIR/packaged_d"
OUTPUT_DIR="${CAESAR_QUALITY_EVAL_DIR:-$ROOT/unified_results/caesar_era5_hourly_quality_eval}"
LOG_DIR="${CAESAR_QUALITY_EVAL_LOG_DIR:-$ROOT/logs/caesar_era5_hourly_quality_eval}"
EBS=(0.1 0.003 0.0001)

mkdir -p "$PACKAGED_DIR" "$OUTPUT_DIR" "$LOG_DIR"
test -f "$PROBE"

python "$ROOT/scripts/package_caesar_d_stage1.py" \
  --vae "$QUALITY_DIR/dq_lr1e4_lam3e6.pt" \
  --base "$ROOT/checkpoints/caesar/caesar_d.pt" \
  --output "$PACKAGED_DIR/dq_lr1e4_lam3e6.pt"
python "$ROOT/scripts/package_caesar_d_stage1.py" \
  --vae "$QUALITY_DIR/dq_lr3e5_lam3e6.pt" \
  --base "$ROOT/checkpoints/caesar/caesar_d.pt" \
  --output "$PACKAGED_DIR/dq_lr3e5_lam3e6.pt"
python "$ROOT/scripts/package_caesar_d_stage1.py" \
  --vae "$PILOT_DIR/d_lr1e4_lam1e5_mr.pt" \
  --base "$ROOT/checkpoints/caesar/caesar_d.pt" \
  --output "$PACKAGED_DIR/d_lr1e4_lam1e5_mr.pt"

run_eval() {
  local gpu="$1"
  local model="$2"
  local name="$3"
  local checkpoint="$4"
  local windows=8
  if [[ "$model" == "caesar_d" ]]; then
    windows=4
  fi
  CUDA_VISIBLE_DEVICES="$gpu" python "$ROOT/scripts/run_dataset_compression.py" \
    --dataset era5_npy \
    --data_root "$PROBE" \
    --output_dir "$OUTPUT_DIR/$name" \
    --models "$model" \
    --max_samples 64 \
    --max_channels 30 \
    --resolution 240 240 \
    --caesar_ckpt_dir "$checkpoint" \
    --caesar_norm_type mean_range \
    --caesar_eb "${EBS[@]}" \
    --caesar_num_windows "$windows" \
    --batch_size 64 \
    --no_lpips \
    >"$LOG_DIR/$name.log" 2>&1
}

run_eval 2 caesar_v vq_lr1e4_lam3e6 "$QUALITY_DIR/vq_lr1e4_lam3e6.pt" &
pids=("$!")
run_eval 3 caesar_v vq_lr3e5_lam3e6 "$QUALITY_DIR/vq_lr3e5_lam3e6.pt" &
pids+=("$!")
run_eval 4 caesar_v v_lr1e4_lam1e5_mr "$PILOT_DIR/v_lr1e4_lam1e5_mr.pt" &
pids+=("$!")
run_eval 5 caesar_d dq_lr1e4_lam3e6 "$PACKAGED_DIR/dq_lr1e4_lam3e6.pt" &
pids+=("$!")
run_eval 6 caesar_d dq_lr3e5_lam3e6 "$PACKAGED_DIR/dq_lr3e5_lam3e6.pt" &
pids+=("$!")
run_eval 7 caesar_d d_lr1e4_lam1e5_mr "$PACKAGED_DIR/d_lr1e4_lam1e5_mr.pt" &
pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
exit "$failed"
