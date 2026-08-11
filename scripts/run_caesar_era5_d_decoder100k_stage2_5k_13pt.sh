#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_ROOT="${OBJECTIVE_INPUT_ROOT:-$ROOT/unified_results/objective_all_to_all_v1}"
CHECKPOINT="${CAESAR_D_13PT_CHECKPOINT:-$ROOT/checkpoints/caesar_era5_d_complete_candidates/lam3em4_decoder100k_stage2_5000.pt}"
VARIANT="${CAESAR_D_13PT_VARIANT:-d_lam3em4_decoder100k_stage2_5000}"
INTERPO_RATE="${CAESAR_D_13PT_INTERPO_RATE:-3}"
OUTPUT_ROOT="${CAESAR_D_DECODER100K_13PT_DIR:-$ROOT/unified_results/objective_era5_caesar_d_decoder100k_stage2_5k_rd}"
LOG_ROOT="${CAESAR_D_DECODER100K_13PT_LOG_DIR:-$ROOT/logs/objective_era5_caesar_d_decoder100k_stage2_5k_rd}"
EXPECTED_EBS=(0.3 0.1 0.05 0.03 0.025 0.02 0.015 0.01 0.003 0.001 0.0001 0.000003 0.000000001)

source /workspace/ai4cp/bin/activate
mkdir -p "$OUTPUT_ROOT/shards" "$LOG_ROOT"
test -f "$CHECKPOINT"

run_shard() {
  local gpu="$1"
  local shard_name="$2"
  shift 2
  local output="$OUTPUT_ROOT/shards/$shard_name"

  python -u "$ROOT/scripts/run_objective_benchmark.py" \
    --dataset era5_npy \
    --gpu "$gpu" \
    --output-root "$output" \
    --input-root "$INPUT_ROOT" \
    --models CAESAR-D \
    --caesar-checkpoint-root "$CHECKPOINT" \
    --caesar-variant "$VARIANT" \
    --caesar-norm-type mean_range \
    --caesar-interpo-rate "$INTERPO_RATE" \
    --caesar-eb "$@" \
    --warmups 0 \
    --repeats 1 \
    --no-lpips \
    >"$LOG_ROOT/$shard_name.log" 2>&1
}

{
  date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
  printf 'role=objective-v1 current-best complete 13-point CAESAR-D curve\n'
  printf 'variant=%s\n' "$VARIANT"
  printf 'interpo_rate=%s\n' "$INTERPO_RATE"
  printf 'ebs=%s\n' "${EXPECTED_EBS[*]}"
  sha256sum "$CHECKPOINT"
} >"$OUTPUT_ROOT/source_manifest.txt"

pids=()
cleanup() {
  local pid
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

run_shard 3 shard_a 0.3 0.025 0.003 0.000000001 & pids+=("$!")
run_shard 4 shard_b 0.1 0.02 0.001 & pids+=("$!")
run_shard 5 shard_c 0.05 0.015 0.0001 & pids+=("$!")
run_shard 6 shard_d 0.03 0.01 0.000003 & pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
pids=()
if [[ "$failed" != 0 ]]; then
  exit "$failed"
fi

python "$ROOT/scripts/merge_objective_shards.py" \
  "$OUTPUT_ROOT/era5_npy/summary.json" \
  "$OUTPUT_ROOT/shards/shard_a/era5_npy/summary.json" \
  "$OUTPUT_ROOT/shards/shard_b/era5_npy/summary.json" \
  "$OUTPUT_ROOT/shards/shard_c/era5_npy/summary.json" \
  "$OUTPUT_ROOT/shards/shard_d/era5_npy/summary.json"

python - "$OUTPUT_ROOT/era5_npy/summary.json" "$VARIANT" "${EXPECTED_EBS[@]}" <<'PY'
import json
import math
import sys

path, variant, *expected_text = sys.argv[1:]
rows = json.load(open(path, encoding="utf-8"))
expected = sorted(float(value) for value in expected_text)
actual = sorted(float(row["eb"]) for row in rows if row.get("checkpoint_variant") == variant)
if len(actual) != len(expected) or any(
    not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15)
    for left, right in zip(actual, expected)
):
    raise SystemExit(f"{path}: expected {expected}, found {actual}")
print(f"{path}: validated {len(actual)} EB points")
PY

touch "$OUTPUT_ROOT/complete"
date -u '+finished_utc=%Y-%m-%dT%H:%M:%SZ' >>"$OUTPUT_ROOT/source_manifest.txt"
trap - INT TERM EXIT
