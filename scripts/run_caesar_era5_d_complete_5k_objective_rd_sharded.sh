#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_ROOT="${OBJECTIVE_INPUT_ROOT:-$ROOT/unified_results/objective_all_to_all_v1}"
CANDIDATE="$ROOT/checkpoints/caesar_era5_d_complete_candidates/lam3em4_decoder_current_stage2_5k.pt"
ORIGINAL="$ROOT/checkpoints/caesar/caesar_d.pt"
CANDIDATE_OUT="${CAESAR_D_5K_OBJECTIVE_DIR:-$ROOT/unified_results/objective_era5_caesar_d_complete_5k_rd}"
ORIGINAL_OUT="${CAESAR_D_ORIGINAL_13PT_DIR:-$ROOT/unified_results/objective_era5_caesar_d_original_13pt_rd}"
LOG_DIR="${CAESAR_D_5K_OBJECTIVE_LOG_DIR:-$ROOT/logs/objective_era5_caesar_d_complete_5k_rd}"
CANDIDATE_VARIANT="d_lam3em4_decoder_current_stage2_5k"
EXPECTED_EBS=(0.3 0.1 0.05 0.03 0.025 0.02 0.015 0.01 0.003 0.001 0.0001 0.000003 0.000000001)

# EB=0.3 and 0.1 were completed before the original serial run was stopped.
SHARD_A=(0.05 0.025 0.015 0.003 0.0001 0.000000001)
SHARD_B=(0.03 0.02 0.01 0.001 0.000003)

source /workspace/ai4cp/bin/activate
mkdir -p "$CANDIDATE_OUT/shards" "$ORIGINAL_OUT/shards" "$LOG_DIR"
test -f "$CANDIDATE"
test -f "$ORIGINAL"
test -f "$CANDIDATE_OUT/era5_npy/summary.json"
test -f "$ORIGINAL_OUT/era5_npy/summary.json"

run_curve() {
  local gpu="$1"
  local checkpoint="$2"
  local variant="$3"
  local output="$4"
  local log="$5"
  shift 5

  python -u "$ROOT/scripts/run_objective_benchmark.py" \
    --dataset era5_npy \
    --gpu "$gpu" \
    --output-root "$output" \
    --input-root "$INPUT_ROOT" \
    --models CAESAR-D \
    --caesar-checkpoint-root "$checkpoint" \
    --caesar-variant "$variant" \
    --caesar-norm-type mean_range \
    --caesar-eb "$@" \
    --warmups 0 \
    --repeats 1 \
    --no-lpips \
    >"$log" 2>&1
}

pids=()
cleanup() {
  local pid
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

run_curve 3 "$CANDIDATE" "$CANDIDATE_VARIANT" \
  "$CANDIDATE_OUT/shards/remaining_a" "$LOG_DIR/candidate_remaining_a.log" \
  "${SHARD_A[@]}" &
pids+=("$!")
run_curve 4 "$CANDIDATE" "$CANDIDATE_VARIANT" \
  "$CANDIDATE_OUT/shards/remaining_b" "$LOG_DIR/candidate_remaining_b.log" \
  "${SHARD_B[@]}" &
pids+=("$!")
run_curve 6 "$ORIGINAL" original \
  "$ORIGINAL_OUT/shards/remaining_a" "$LOG_DIR/original_remaining_a.log" \
  "${SHARD_A[@]}" &
pids+=("$!")
run_curve 7 "$ORIGINAL" original \
  "$ORIGINAL_OUT/shards/remaining_b" "$LOG_DIR/original_remaining_b.log" \
  "${SHARD_B[@]}" &
pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
pids=()
if [[ "$failed" != 0 ]]; then
  exit "$failed"
fi

python "$ROOT/scripts/merge_objective_shards.py" \
  "$CANDIDATE_OUT/era5_npy/summary.json" \
  "$CANDIDATE_OUT/shards/remaining_a/era5_npy/summary.json" \
  "$CANDIDATE_OUT/shards/remaining_b/era5_npy/summary.json"
python "$ROOT/scripts/merge_objective_shards.py" \
  "$ORIGINAL_OUT/era5_npy/summary.json" \
  "$ORIGINAL_OUT/shards/remaining_a/era5_npy/summary.json" \
  "$ORIGINAL_OUT/shards/remaining_b/era5_npy/summary.json"

python - "$CANDIDATE_OUT/era5_npy/summary.json" "$CANDIDATE_VARIANT" "${EXPECTED_EBS[@]}" <<'PY'
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
    raise SystemExit(f"{path}: expected EB values {expected}, found {actual}")
print(f"{path}: validated {len(actual)} EB points")
PY

python - "$ORIGINAL_OUT/era5_npy/summary.json" original "${EXPECTED_EBS[@]}" <<'PY'
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
    raise SystemExit(f"{path}: expected EB values {expected}, found {actual}")
print(f"{path}: validated {len(actual)} EB points")
PY

touch "$CANDIDATE_OUT/complete" "$ORIGINAL_OUT/complete"
date -u '+sharded_finished_utc=%Y-%m-%dT%H:%M:%SZ' >>"$CANDIDATE_OUT/source_manifest.txt"
trap - INT TERM EXIT
