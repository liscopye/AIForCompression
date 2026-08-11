#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ERA5_SHARD_DIR:-/workspace/Data/ERA5/hourly_center512_shards_20240301_90d}"
SOURCE="$ROOT/checkpoints/caesar_era5_vd_lowrate_100k/v_lr1em5_lam1em3_full100k_update100000.pt"
OUTPUT_DIR="${CAESAR_V_QUALITY_RECOVERY_DIR:-$ROOT/checkpoints/caesar_era5_v_lowrate_quality_recovery_10k}"
LOG_DIR="${CAESAR_V_QUALITY_RECOVERY_LOG_DIR:-$ROOT/logs/caesar_era5_v_lowrate_quality_recovery_10k}"

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
  printf 'objective=quality_recovery_at_low_rate\n'
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
  --warmup_updates 250
  --log_interval 100
  --val_interval 2000
  --save_interval 10000
  --milestone_steps 500 2000 5000 10000
  --norm_type mean_range
  --ckpt_path "$SOURCE"
  --lr 3e-6
  --wandb_project caesar-era5-hourly-tuning
  --wandb_group v-lowrate-100k-quality-recovery-10k
  --wandb_tags era5 low-rate quality-recovery source-domain continuation
  --require_wandb
  --device cuda:0
)

run_one() {
  local gpu="$1"
  local name="$2"
  local domain="$3"
  local lambda_rate="$4"

  echo "GPU $gpu: $name domain=$domain lambda=$lambda_rate"
  CUDA_VISIBLE_DEVICES="$gpu" python -u "$ROOT/scripts/finetune_caesar_era5.py" \
    "${common[@]}" \
    --distortion_domain "$domain" \
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
  local name="$2"
  local domain="$3"
  local lambda_rate="$4"
  run_one "$gpu" "$name" "$domain" "$lambda_rate" &
  pids+=("$!")
  names+=("$name")
}

launch 2 source_lam0_lr3em6 source 0
launch 3 source_lam1em3_lr3em6 source 1e-3
launch 4 source_lam3em3_lr3em6 source 3e-3
launch 5 source_lam1em2_lr3em6 source 1e-2
launch 6 source_lam3em2_lr3em6 source 3e-2
launch 7 normalized_lam1em4_lr3em6 normalized 1e-4

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
