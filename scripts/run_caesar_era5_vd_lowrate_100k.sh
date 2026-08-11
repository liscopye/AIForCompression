#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ERA5_SHARD_DIR:-/workspace/Data/ERA5/hourly_center512_shards_20240301_90d}"
OUTPUT_DIR="${CAESAR_VD_LOWRATE_100K_DIR:-$ROOT/checkpoints/caesar_era5_vd_lowrate_100k}"
LOG_DIR="${CAESAR_VD_LOWRATE_100K_LOG_DIR:-$ROOT/logs/caesar_era5_vd_lowrate_100k}"
V_SOURCE="$ROOT/checkpoints/caesar/caesar_v.pt"
D_SOURCE="$ROOT/checkpoints/caesar/caesar_d.pt"

source /workspace/ai4cp/bin/activate
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
test -f "$V_SOURCE"
test -f "$D_SOURCE"
wandb login --verify >/dev/null

actual_days=$(find "$DATA_DIR" -maxdepth 1 -name '*_hourly.npy' -type f 2>/dev/null | wc -l)
if [[ "$actual_days" -lt 90 ]]; then
  echo "Refusing to start: found $actual_days/90 completed ERA5 shards in $DATA_DIR" >&2
  exit 2
fi

{
  date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
  printf 'data_dir=%s\n' "$DATA_DIR"
  printf 'v_source=%s\n' "$V_SOURCE"
  printf 'd_source=%s\n' "$D_SOURCE"
  printf 'iterations=100000\n'
  printf 'optimizer_restart=from_original\n'
  sha256sum "$V_SOURCE" "$D_SOURCE"
} >"$OUTPUT_DIR/source_manifest.txt"

common=(
  --stage 1
  --data_backend npy_shards
  --data_dir "$DATA_DIR"
  --train_timesteps 1776
  --val_timesteps 384
  --frame_step 24
  --netcdf_val_channel_stride 4
  --netcdf_max_open_file_pairs 8
  --train_size 256
  --batch_size 32
  --val_batch_size 32
  --num_workers 4
  --prefetch_factor 2
  --iterations 100000
  --rate_mode bpp
  --distortion_domain normalized
  --warmup_updates 500
  --log_interval 100
  --val_interval 10000
  --save_interval 25000
  --milestone_steps 10000 25000 50000 75000 100000
  --norm_type mean_range
  --lr 1e-5
  --wandb_project caesar-era5-hourly-tuning
  --wandb_group vd-lowrate-from-original-100k
  --wandb_tags era5 daily-cadence low-rate from-original full-100k
  --require_wandb
  --device cuda:0
)

run_one() {
  local gpu="$1"
  local model="$2"
  local name="$3"
  local lambda_rate="$4"
  local checkpoint n_frame temporal_stride

  if [[ "$model" == "V" ]]; then
    checkpoint="$V_SOURCE"
    n_frame=8
    temporal_stride=8
  else
    checkpoint="$D_SOURCE"
    n_frame=16
    temporal_stride=1
  fi

  echo "GPU $gpu: $name model=$model source=$checkpoint lambda=$lambda_rate"
  CUDA_VISIBLE_DEVICES="$gpu" python -u "$ROOT/scripts/finetune_caesar_era5.py" \
    "${common[@]}" \
    --model_type "$model" \
    --n_frame "$n_frame" \
    --temporal_stride "$temporal_stride" \
    --ckpt_path "$checkpoint" \
    --lambda_rate "$lambda_rate" \
    --output_ckpt "$OUTPUT_DIR/$name.pt" \
    --wandb_run_name "$name" \
    >"$LOG_DIR/$name.log" 2>&1
  touch "$OUTPUT_DIR/$name.done"
}

pids=()
names=()

launch() {
  local gpu="$1"
  local model="$2"
  local name="$3"
  local lambda_rate="$4"
  run_one "$gpu" "$model" "$name" "$lambda_rate" &
  pids+=("$!")
  names+=("$name")
}

# Only retain the two Stage-1 paths used by the final selected V/D models.
launch 4 V v_lr1em5_lam1em3_full100k 1e-3
launch 6 D d_s1_lr1em5_lam3em4_full100k 3e-4

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "finished ${names[$index]}"
  else
    echo "failed ${names[$index]}" >&2
    failed=1
  fi
done

date -u '+finished_utc=%Y-%m-%dT%H:%M:%SZ' >>"$OUTPUT_DIR/source_manifest.txt"
exit "$failed"
