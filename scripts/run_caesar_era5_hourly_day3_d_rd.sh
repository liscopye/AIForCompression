#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARD_DIR="${ERA5_HOURLY_SHARD_DIR:-/workspace/Data/ERA5/hourly_center512_shards_20240301_90d}"
PROBE="${ERA5_DAY3_D_PROBE:-/workspace/Data/ERA5/hourly_day3_d_validation_probe.npy}"
OUT="${ERA5_DAY3_D_RD_OUT:-$ROOT/unified_results/caesar_era5_hourly_day3_d_rd_diagnostic}"
LOG="${ERA5_DAY3_D_RD_LOG:-$ROOT/logs/caesar_era5_hourly_day3_d_rd_diagnostic}"
PACKAGED="$ROOT/checkpoints/caesar_era5_hourly_day2_d_smoke/packaged"

DAY3_SHARD="$SHARD_DIR/2024-03-03_hourly.npy"
if [[ ! -f "$DAY3_SHARD" ]]; then
  echo "Need $DAY3_SHARD for a D probe disjoint from its first-48-hour smoke split" >&2
  exit 2
fi

mkdir -p "$OUT" "$LOG"
python "$ROOT/utils/build_era5_hourly_validation_probe.py" \
  --shard-dir "$SHARD_DIR" \
  --output "$PROBE" \
  --train-timesteps 48 \
  --probe-timesteps 16 \
  --crop-size 240

run_one() {
  local gpu="$1"
  local name="$2"
  local checkpoint="$3"
  CUDA_VISIBLE_DEVICES="$gpu" python "$ROOT/scripts/run_dataset_compression.py" \
    --dataset era5_npy \
    --data_root "$PROBE" \
    --output_dir "$OUT/$name" \
    --models caesar_d \
    --max_samples 16 \
    --max_channels 30 \
    --resolution 240 240 \
    --caesar_ckpt_dir "$checkpoint" \
    --caesar_eb 0.1 0.003 0.0001 \
    --caesar_num_windows 1 \
    --batch_size 64 \
    --no_lpips \
    >"$LOG/$name.log" 2>&1
}

run_one 3 original_d "$ROOT/checkpoints/caesar/caesar_d.pt" &
p0=$!
run_one 4 d_update100 "$PACKAGED/d_lr1e4_lam1e4_mr_update100.pt" &
p1=$!
run_one 5 d_update500 "$PACKAGED/d_lr1e4_lam1e4_mr_update500.pt" &
p2=$!

failed=0
for pid in "$p0" "$p1" "$p2"; do
  wait "$pid" || failed=1
done
if [[ "$failed" -ne 0 ]]; then
  echo "One or more day-3 D RD jobs failed; inspect $LOG" >&2
  exit 1
fi

ROOT_PATH="$ROOT" OUT_PATH="$OUT" python - <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(os.environ["ROOT_PATH"])
out = Path(os.environ["OUT_PATH"])
sys.path.insert(0, str(root))
from scripts.select_caesar_era5_hourly_checkpoint import (
    load_curve,
    piecewise_log_bd_rate,
)

reference = load_curve(out / "original_d" / "summary.json")
result = {
    "role": "independent day-3 diagnostic; not objective-v1",
    "source_time_indices": [48, 64],
    "curves": {"original_d": reference},
    "bd_rate_percent_vs_original": {},
}
for name in ("d_update100", "d_update500"):
    curve = load_curve(out / name / "summary.json")
    result["curves"][name] = curve
    result["bd_rate_percent_vs_original"][name] = piecewise_log_bd_rate(
        reference, curve
    )
(out / "diagnostic.json").write_text(
    json.dumps(result, indent=2) + "\n", encoding="utf-8"
)
print(out / "diagnostic.json")
PY
