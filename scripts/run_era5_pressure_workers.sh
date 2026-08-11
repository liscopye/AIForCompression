#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ERA5_HOURLY_DATA_DIR:-/workspace/Data/ERA5/hourly_center512_20240301_90d}"
LOG_DIR="${ERA5_HOURLY_LOG_DIR:-$ROOT/logs/era5_hourly_center512_20240301_90d}"
WORKERS="${ERA5_PRESSURE_WORKERS:-4}"
PRESSURE_DAYS_PER_REQUEST="${ERA5_PRESSURE_DAYS_PER_REQUEST:-9}"

mkdir -p "$DATA_DIR" "$LOG_DIR/pressure_workers"

# The ranges cover 2024-03-01 through 2024-05-29 exactly once.
starts=(2024-03-01 2024-03-13 2024-03-24 2024-04-04 2024-04-15 2024-04-27 2024-05-08 2024-05-19)
days=(12 11 11 11 12 11 11 11)

if (( WORKERS < 1 || WORKERS > ${#starts[@]} )); then
  echo "ERA5_PRESSURE_WORKERS must be between 1 and ${#starts[@]}" >&2
  exit 2
fi

run_lane() {
  local lane="$1"
  local index name
  for ((index = lane; index < ${#starts[@]}; index += WORKERS)); do
    name="range_$((index + 1))"
    python -u "$ROOT/utils/download_era5.py" \
      --start-date "${starts[$index]}" \
      --days "${days[$index]}" \
      --times hourly \
      --batch-times \
      --area 64 -64 -64 64 \
      --content pressure \
      --pressure-format grib \
      --pressure-days-per-request "$PRESSURE_DAYS_PER_REQUEST" \
      --data-dir "$DATA_DIR" \
      --skip-verify \
      >>"$LOG_DIR/pressure_workers/$name.log" 2>&1
  done
}

pids=()
for ((lane = 0; lane < WORKERS; lane++)); do
  run_lane "$lane" &
  pids+=("$!")
  echo "${pids[-1]}" >"$LOG_DIR/pressure_workers/lane_$((lane + 1)).pid"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
exit "$failed"
