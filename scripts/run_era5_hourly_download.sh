#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ERA5_HOURLY_DATA_DIR:-/workspace/Data/ERA5/hourly_center512_20240301_90d}"
SHARD_DIR="${ERA5_HOURLY_SHARD_DIR:-/workspace/Data/ERA5/hourly_center512_shards_20240301_90d}"
LOG_DIR="${ERA5_HOURLY_LOG_DIR:-$ROOT/logs/era5_hourly_center512_20240301_90d}"
START_DATE="${ERA5_START_DATE:-2024-03-01}"
DAYS="${ERA5_DAYS:-90}"

mkdir -p "$DATA_DIR" "$SHARD_DIR" "$LOG_DIR"

pid_file="$LOG_DIR/download.pid"
log_file="$LOG_DIR/download.log"
if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
  echo "download already running as PID $(cat "$pid_file")"
else
  # A 128-degree square on the 0.25-degree ERA5 grid yields 513x513 points.
  # One batched request per day avoids the CDS per-dataset queued-request limit.
  nohup setsid python -u "$ROOT/utils/download_era5.py" \
    --start-date "$START_DATE" \
    --days "$DAYS" \
    --times hourly \
    --batch-times \
    --area 64 -64 -64 64 \
    --content pressure \
    --pressure-format grib \
    --pressure-days-per-request "${ERA5_PRESSURE_DAYS_PER_REQUEST:-9}" \
    --data-dir "$DATA_DIR" \
    --skip-verify \
    >>"$log_file" 2>&1 </dev/null &
  echo "$!" >"$pid_file"
  echo "started download as PID $! -> $log_file"
fi

single_pid_file="$LOG_DIR/single.pid"
single_log_file="$LOG_DIR/single.log"
if [[ -f "$single_pid_file" ]] && kill -0 "$(cat "$single_pid_file")" 2>/dev/null; then
  echo "single-level download already running as PID $(cat "$single_pid_file")"
else
  nohup setsid python -u "$ROOT/utils/download_era5.py" \
    --start-date "$START_DATE" \
    --days "$DAYS" \
    --times hourly \
    --batch-times \
    --area 64 -64 -64 64 \
    --content single \
    --data-dir "$DATA_DIR" \
    --skip-verify \
    >>"$single_log_file" 2>&1 </dev/null &
  echo "$!" >"$single_pid_file"
  echo "started single-level download as PID $! -> $single_log_file"
fi

converter_pid_file="$LOG_DIR/converter.pid"
converter_log_file="$LOG_DIR/converter.log"
if [[ -f "$converter_pid_file" ]] && kill -0 "$(cat "$converter_pid_file")" 2>/dev/null; then
  echo "converter already running as PID $(cat "$converter_pid_file")"
else
  nohup setsid python -u "$ROOT/utils/prepare_era5_hourly_shards.py" \
    --input-dir "$DATA_DIR" \
    --output-dir "$SHARD_DIR" \
    --expected-days "$DAYS" \
    --poll-seconds 60 \
    --watch \
    >>"$converter_log_file" 2>&1 </dev/null &
  echo "$!" >"$converter_pid_file"
  echo "started converter as PID $! -> $converter_log_file"
fi
