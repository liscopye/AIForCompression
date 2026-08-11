#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ERA5_SHARD_DIR:-/workspace/Data/ERA5/hourly_center512_shards_20240301_90d}"
SOURCE="$ROOT/checkpoints/caesar_era5_vd_lowrate_100k/v_lr1em5_lam1em3_full100k_update100000.pt"
OUTPUT_DIR="${CAESAR_V_DECODER_QUALITY_DIR:-$ROOT/checkpoints/caesar_era5_v_decoder_quality_10k}"
LOG_DIR="${CAESAR_V_DECODER_QUALITY_LOG_DIR:-$ROOT/logs/caesar_era5_v_decoder_quality_10k}"

source /workspace/ai4cp/bin/activate
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
test -f "$SOURCE"
wandb login --verify >/dev/null

actual_days=$(find "$DATA_DIR" -maxdepth 1 -name '*_hourly.npy' -type f 2>/dev/null | wc -l)
if [[ "$actual_days" -lt 90 ]]; then
  echo "Refusing to start: found $actual_days/90 completed ERA5 shards in $DATA_DIR" >&2
  exit 2
fi

{
  date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
  printf 'data_dir=%s\n' "$DATA_DIR"
  printf 'source=%s\n' "$SOURCE"
  printf 'objective=frozen_encoder_decoder_quality_recovery\n'
  sha256sum "$SOURCE"
} >"$OUTPUT_DIR/source_manifest.txt"

common=(
  --model_type V
  --stage 1
  --data_backend npy_shards
  --data_dir "$DATA_DIR"
  --train_timesteps 1776
  --val_timesteps 384
  --n_frame 8
  --frame_step 24
  --temporal_stride 8
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
  --warmup_updates 250
  --log_interval 100
  --val_interval 2000
  --save_interval 10000
  --milestone_steps 500 2000 5000 10000
  --norm_type mean_range
  --ckpt_path "$SOURCE"
  --wandb_project caesar-era5-hourly-tuning
  --wandb_group v-lowrate-frozen-encoder-quality-10k
  --wandb_tags era5 low-rate decoder-only frozen-encoder quality-recovery
  --require_wandb
  --device cuda:0
)

run_one() {
  local gpu="$1"
  local scope="$2"
  local lr="$3"
  local name="${scope}_lr${lr//-/m}"

  echo "GPU $gpu: $name scope=$scope lr=$lr"
  CUDA_VISIBLE_DEVICES="$gpu" python -u "$ROOT/scripts/finetune_caesar_era5.py" \
    "${common[@]}" \
    --trainable_scope "$scope" \
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
  local scope="$2"
  local lr="$3"
  local name="${scope}_lr${lr//-/m}"
  run_one "$gpu" "$scope" "$lr" &
  pids+=("$!")
  names+=("$name")
}

launch 2 decoder 1e-6
launch 3 decoder 3e-6
launch 4 decoder 1e-5
launch 5 sr 1e-6
launch 6 sr 3e-6
launch 7 sr 1e-5

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
