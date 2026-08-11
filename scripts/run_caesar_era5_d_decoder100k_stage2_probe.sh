#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_ROOT="${OBJECTIVE_INPUT_ROOT:-$ROOT/unified_results/objective_all_to_all_v1}"
DECODER="$ROOT/checkpoints/caesar_era5_d_decoder_quality_100k/lam3em4_from_lowrate_lr3em4.pt"
STAGE2_ROOT="$ROOT/checkpoints/caesar_era5_d_stage2_full_200k"
PACKAGED_ROOT="$ROOT/checkpoints/caesar_era5_d_complete_candidates"
OUTPUT_ROOT="${CAESAR_D_DECODER100K_PROBE_DIR:-$ROOT/unified_results/objective_era5_caesar_d_decoder100k_stage2_probes}"
LOG_ROOT="${CAESAR_D_DECODER100K_PROBE_LOG_DIR:-$ROOT/logs/objective_era5_caesar_d_decoder100k_stage2_probes}"
EBS=(0.3 0.03 0.015 0.003)
STEPS=(5000 10000)
GPUS=(3 4)

source /workspace/ai4cp/bin/activate
mkdir -p "$PACKAGED_ROOT" "$OUTPUT_ROOT" "$LOG_ROOT"
test -f "$DECODER"

run_probe() {
  local gpu="$1"
  local step="$2"
  local stage2="$STAGE2_ROOT/lam3em4_stage2_from_original_lr1em4_200k_update${step}.pt"
  local packaged="$PACKAGED_ROOT/lam3em4_decoder100k_stage2_${step}.pt"
  local variant="d_lam3em4_decoder100k_stage2_${step}"
  local output="$OUTPUT_ROOT/update${step}"
  local log="$LOG_ROOT/update${step}.log"
  test -f "$stage2"

  python "$ROOT/scripts/package_caesar_d_stage1.py" \
    --vae "$DECODER" \
    --base "$stage2" \
    --output "$packaged"

  {
    date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
    printf 'role=objective-v1 final-decoder Stage2 milestone probe\n'
    printf 'stage2_update=%s\n' "$step"
    printf 'ebs=%s\n' "${EBS[*]}"
    sha256sum "$DECODER" "$stage2" "$packaged"
  } >"$output.source_manifest.txt"

  python -u "$ROOT/scripts/run_objective_benchmark.py" \
    --dataset era5_npy \
    --gpu "$gpu" \
    --output-root "$output" \
    --input-root "$INPUT_ROOT" \
    --models CAESAR-D \
    --caesar-checkpoint-root "$packaged" \
    --caesar-variant "$variant" \
    --caesar-norm-type mean_range \
    --caesar-eb "${EBS[@]}" \
    --warmups 0 \
    --repeats 1 \
    --no-lpips \
    >"$log" 2>&1

  python - "$output/era5_npy/summary.json" "$variant" "${EBS[@]}" <<'PY'
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
PY

  touch "$output/complete"
  date -u '+finished_utc=%Y-%m-%dT%H:%M:%SZ' >>"$output.source_manifest.txt"
}

pids=()
cleanup() {
  local pid
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

for index in "${!STEPS[@]}"; do
  run_probe "${GPUS[$index]}" "${STEPS[$index]}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
pids=()
if [[ "$failed" != 0 ]]; then
  exit "$failed"
fi

touch "$OUTPUT_ROOT/complete"
trap - INT TERM EXIT
