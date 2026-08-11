#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/AIForCompression
DATA=/workspace/Data/ERA5/finetune_processed_time_split_t45_v16/era5_val.npy
CKPT="$ROOT/checkpoints/caesar_era5_stability_20260723"
PACKAGED="$CKPT/packaged_d"
OUT="$ROOT/unified_results/caesar_era5_stability_eval_20260723"
LOG="$ROOT/logs/caesar_era5_stability_eval_20260723"
EBS=(0.001 0.005 0.05)
STEPS=(100 500 1000 2000)

mkdir -p "$PACKAGED" "$OUT" "$LOG"

d_configs=(
  d_mr_lam1e4_anchor0
  d_hw_lam1e4_anchor0
  d_hw_lam1e4_anchor1
  d_hw_lam3e4_anchor1
)
for name in "${d_configs[@]}"; do
  for step in "${STEPS[@]}"; do
    python "$ROOT/scripts/package_caesar_d_stage1.py" \
      --vae "$CKPT/${name}_update${step}.pt" \
      --base "$ROOT/checkpoints/caesar/caesar_d.pt" \
      --output "$PACKAGED/${name}_update${step}.pt" \
      >/dev/null
  done
done

pids=()
names=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

run_eval() {
  local gpu=$1
  local model=$2
  local name=$3
  local checkpoint=$4
  local max_samples=8
  if [[ "$model" == caesar_d ]]; then
    max_samples=16
  fi
  CUDA_VISIBLE_DEVICES="$gpu" python "$ROOT/scripts/run_dataset_compression.py" \
    --dataset era5_npy \
    --data_root "$DATA" \
    --output_dir "$OUT/$name" \
    --models "$model" \
    --max_samples "$max_samples" \
    --max_channels 3 \
    --resolution 240 240 \
    --caesar_ckpt_dir "$checkpoint" \
    --caesar_eb "${EBS[@]}" \
    --batch_size 32 \
    --no_lpips \
    >"$LOG/$name.log" 2>&1
}

launch_queue() {
  local gpu=$1
  local model=$2
  local config=$3
  local base_checkpoint=$4
  (
    if [[ "$gpu" == 0 || "$gpu" == 4 ]]; then
      run_eval "$gpu" "$model" "original_${model}" "$base_checkpoint"
    fi
    if [[ "$config" != original ]]; then
      for step in "${STEPS[@]}"; do
        local checkpoint="$CKPT/${config}_update${step}.pt"
        if [[ "$model" == caesar_d ]]; then
          checkpoint="$PACKAGED/${config}_update${step}.pt"
        fi
        run_eval "$gpu" "$model" "${config}_update${step}" "$checkpoint"
      done
    fi
  ) &
  pids+=("$!")
  names+=("$config")
  printf 'started gpu=%s pid=%s queue=%s\n' "$gpu" "$!" "$config"
}

launch_queue 0 caesar_v v_mr_lam1e4_anchor0 "$ROOT/checkpoints/caesar/caesar_v.pt"
launch_queue 1 caesar_v v_hw_lam1e4_anchor0 "$ROOT/checkpoints/caesar/caesar_v.pt"
launch_queue 2 caesar_v v_hw_lam1e4_anchor1 "$ROOT/checkpoints/caesar/caesar_v.pt"
launch_queue 3 caesar_v v_hw_lam3e4_anchor1 "$ROOT/checkpoints/caesar/caesar_v.pt"
launch_queue 4 caesar_d d_mr_lam1e4_anchor0 "$ROOT/checkpoints/caesar/caesar_d.pt"
launch_queue 5 caesar_d d_hw_lam1e4_anchor0 "$ROOT/checkpoints/caesar/caesar_d.pt"
launch_queue 6 caesar_d d_hw_lam1e4_anchor1 "$ROOT/checkpoints/caesar/caesar_d.pt"
launch_queue 7 caesar_d d_hw_lam3e4_anchor1 "$ROOT/checkpoints/caesar/caesar_d.pt"

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    printf 'finished queue=%s\n' "${names[$index]}"
  else
    printf 'failed queue=%s\n' "${names[$index]}" >&2
    failed=1
  fi
done
exit "$failed"
