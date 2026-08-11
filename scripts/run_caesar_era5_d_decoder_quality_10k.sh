#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ERA5_SHARD_DIR:-/workspace/Data/ERA5/hourly_center512_shards_20240301_90d}"
SOURCE_DIR="$ROOT/checkpoints/caesar_era5_vd_lowrate_100k"
OUTPUT_DIR="${CAESAR_D_DECODER_10K_DIR:-$ROOT/checkpoints/caesar_era5_d_decoder_quality_10k}"
LOG_DIR="${CAESAR_D_DECODER_10K_LOG_DIR:-$ROOT/logs/caesar_era5_d_decoder_quality_10k}"

source /workspace/ai4cp/bin/activate
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
wandb login --verify >/dev/null

actual_days=$(find "$DATA_DIR" -maxdepth 1 -name '*_hourly.npy' -type f 2>/dev/null | wc -l)
if [[ "$actual_days" -lt 90 ]]; then
  echo "Refusing to start: found $actual_days/90 completed ERA5 shards in $DATA_DIR" >&2
  exit 2
fi

source_lam1em4="$SOURCE_DIR/d_s1_lr1em5_lam1em4_full100k_update100000.pt"
source_lam3em4="$SOURCE_DIR/d_s1_lr1em5_lam3em4_full100k_update100000.pt"
test -f "$source_lam1em4"
test -f "$source_lam3em4"

{
  date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
  printf 'data_dir=%s\n' "$DATA_DIR"
  printf 'objective=frozen_rate_caesar_d_stage1_decoder_quality_recovery\n'
  printf 'trainable_scope=decoder\n'
  printf 'iterations=10000\n'
  sha256sum "$source_lam1em4" "$source_lam3em4"
} >"$OUTPUT_DIR/source_manifest.txt"

common=(
  --model_type D
  --stage 1
  --data_backend npy_shards
  --data_dir "$DATA_DIR"
  --train_timesteps 1776
  --val_timesteps 384
  --n_frame 16
  --frame_step 24
  --temporal_stride 1
  --netcdf_val_channel_stride 4
  --netcdf_max_open_file_pairs 8
  --train_size 256
  --batch_size 32
  --val_batch_size 32
  --num_workers 4
  --prefetch_factor 2
  --iterations 10000
  --rate_mode bpp
  --distortion_domain normalized
  --lambda_rate 0
  --trainable_scope decoder
  --warmup_updates 250
  --log_interval 100
  --val_interval 1000
  --save_interval 10000
  --milestone_steps 500 1000 2500 5000 7500 10000
  --norm_type mean_range
  --wandb_project caesar-era5-hourly-tuning
  --wandb_group d-stage1-frozen-rate-decoder-quality-10k
  --wandb_tags era5 caesar-d stage1 decoder-only frozen-rate quality-recovery
  --require_wandb
  --device cuda:0
)

run_one() {
  local gpu="$1"
  local source_tag="$2"
  local source="$3"
  local lr="$4"
  local lr_tag="${lr//-/m}"
  local name="${source_tag}_decoder_lr${lr_tag}"

  echo "GPU $gpu: $name source=$(basename "$source") lr=$lr"
  CUDA_VISIBLE_DEVICES="$gpu" python -u "$ROOT/scripts/finetune_caesar_era5.py" \
    "${common[@]}" \
    --ckpt_path "$source" \
    --lr "$lr" \
    --output_ckpt "$OUTPUT_DIR/$name.pt" \
    --wandb_run_name "$name" \
    >"$LOG_DIR/$name.log" 2>&1
  touch "$OUTPUT_DIR/$name.done"
}

pids=()
names=()

launch() {
  local gpu="$1"
  local source_tag="$2"
  local source="$3"
  local lr="$4"
  run_one "$gpu" "$source_tag" "$source" "$lr" &
  pids+=("$!")
  names+=("${source_tag}_decoder_lr${lr//-/m}")
}

launch 2 lam1em4 "$source_lam1em4" 1e-4
launch 3 lam1em4 "$source_lam1em4" 3e-4
launch 4 lam1em4 "$source_lam1em4" 1e-3
launch 5 lam3em4 "$source_lam3em4" 1e-4
launch 6 lam3em4 "$source_lam3em4" 3e-4
launch 7 lam3em4 "$source_lam3em4" 1e-3

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
