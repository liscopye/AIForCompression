#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARD_DIR="${ERA5_HOURLY_SHARD_DIR:-/workspace/Data/ERA5/hourly_center512_shards_20240301_90d}"
PROBE="${ERA5_HOURLY_PROBE:-/workspace/Data/ERA5/hourly_center512_validation_probe.npy}"
CKPT_DIR="${CAESAR_HOURLY_CKPT_DIR:-$ROOT/checkpoints/caesar_era5_hourly_pilot}"
PACKAGED_DIR="$CKPT_DIR/packaged_d"
OUTPUT_DIR="${CAESAR_HOURLY_EVAL_DIR:-$ROOT/unified_results/caesar_era5_hourly_pilot_eval}"
LOG_DIR="${CAESAR_HOURLY_EVAL_LOG_DIR:-$ROOT/logs/caesar_era5_hourly_pilot_eval}"
STEPS=(100 500 1000 2000 5000 10000)
EBS=(0.1 0.003 0.0001)

mkdir -p "$PACKAGED_DIR" "$OUTPUT_DIR" "$LOG_DIR"
actual_days=$(find "$SHARD_DIR" -maxdepth 1 -name '*_hourly.npy' -type f 2>/dev/null | wc -l)
if [[ "$actual_days" -lt 90 ]]; then
  echo "Refusing to evaluate: found $actual_days/90 completed daily shards in $SHARD_DIR" >&2
  exit 2
fi

python "$ROOT/utils/build_era5_hourly_validation_probe.py" \
  --shard-dir "$SHARD_DIR" \
  --output "$PROBE" \
  --train-timesteps 1920 \
  --probe-timesteps 64 \
  --crop-size 240

v_configs=(
  v_lr3e6_lam1e4_mr
  v_lr1e5_lam1e4_mr
  v_lr3e5_lam1e4_mr
  v_lr1e4_lam1e4_mr
  v_lr1e4_lam1e5_mr
  v_lr1e5_lam3e4_mr
  v_lr1e5_lam1e4_hw
  v_lr1e4_lam1e4_hw
)
d_configs=(
  d_lr3e6_lam1e4_mr
  d_lr1e5_lam1e4_mr
  d_lr3e5_lam1e4_mr
  d_lr1e4_lam1e4_mr
  d_lr1e4_lam1e5_mr
  d_lr1e5_lam3e4_mr
  d_lr1e5_lam1e4_hw
  d_lr1e4_lam1e4_hw
)

for config in "${d_configs[@]}"; do
  for step in "${STEPS[@]}"; do
    python "$ROOT/scripts/package_caesar_d_stage1.py" \
      --vae "$CKPT_DIR/${config}_update${step}.pt" \
      --base "$ROOT/checkpoints/caesar/caesar_d.pt" \
      --output "$PACKAGED_DIR/${config}_update${step}.pt" \
      >/dev/null
  done
done

run_eval() {
  local gpu="$1"
  local model="$2"
  local name="$3"
  local checkpoint="$4"
  local norm_type="mean_range"
  if [[ "$name" == *_hw* ]]; then
    norm_type="mean_range_hw"
  fi
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
    --caesar_norm_type "$norm_type" \
    --caesar_eb "${EBS[@]}" \
    --caesar_num_windows "$windows" \
    --batch_size 64 \
    --no_lpips \
    >"$LOG_DIR/$name.log" 2>&1
}

pids=()
names=()
launch_queue() {
  local gpu="$1"
  local model="$2"
  shift 2
  (
    if [[ "$gpu" == 0 ]]; then
      run_eval "$gpu" caesar_v original_v "$ROOT/checkpoints/caesar/caesar_v.pt"
    elif [[ "$gpu" == 4 ]]; then
      run_eval "$gpu" caesar_d original_d "$ROOT/checkpoints/caesar/caesar_d.pt"
    fi
    for config in "$@"; do
      for step in "${STEPS[@]}"; do
        local checkpoint="$CKPT_DIR/${config}_update${step}.pt"
        if [[ "$model" == "caesar_d" ]]; then
          checkpoint="$PACKAGED_DIR/${config}_update${step}.pt"
        fi
        run_eval "$gpu" "$model" "${config}_update${step}" "$checkpoint"
      done
    done
  ) &
  pids+=("$!")
  names+=("gpu${gpu}")
  echo "GPU $gpu evaluation queue (PID $!)"
}

launch_queue 0 caesar_v "${v_configs[0]}" "${v_configs[5]}"
launch_queue 1 caesar_v "${v_configs[1]}" "${v_configs[6]}"
launch_queue 2 caesar_v "${v_configs[2]}" "${v_configs[4]}"
launch_queue 3 caesar_v "${v_configs[3]}" "${v_configs[7]}"
launch_queue 4 caesar_d "${d_configs[0]}" "${d_configs[5]}"
launch_queue 5 caesar_d "${d_configs[1]}" "${d_configs[6]}"
launch_queue 6 caesar_d "${d_configs[2]}" "${d_configs[4]}"
launch_queue 7 caesar_d "${d_configs[3]}" "${d_configs[7]}"

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
