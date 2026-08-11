#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ERA5_HOURLY_DATA_DIR:-/workspace/Data/ERA5/hourly_center512_20240301_90d}"
SHARD_DIR="${ERA5_HOURLY_SHARD_DIR:-/workspace/Data/ERA5/hourly_center512_shards_20240301_90d}"
LOG_DIR="${ERA5_HOURLY_LOG_DIR:-$ROOT/logs/era5_hourly_center512_20240301_90d}"
POLL_SECONDS="${ERA5_DOWNLOAD_WATCH_POLL_SECONDS:-600}"

mkdir -p "$DATA_DIR" "$SHARD_DIR" "$LOG_DIR"

alive() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null
}

start_downloader() {
  local name="$1"
  local start_date="$2"
  local days="$3"
  local content="$4"
  nohup setsid python -u "$ROOT/utils/download_era5.py" \
    --start-date "$start_date" \
    --days "$days" \
    --times hourly \
    --batch-times \
    --area 64 -64 -64 64 \
    --content "$content" \
    --pressure-format grib \
    --data-dir "$DATA_DIR" \
    --skip-verify \
    >>"$LOG_DIR/$name.log" 2>&1 </dev/null &
  echo "$!" >"$LOG_DIR/$name.pid"
  echo "$(date -u +%FT%TZ) restarted $name as PID $!"
}

start_pressure_workers() {
  nohup setsid bash "$ROOT/scripts/run_era5_pressure_workers.sh" \
    >>"$LOG_DIR/pressure_workers.log" 2>&1 </dev/null &
  echo "$!" >"$LOG_DIR/pressure_workers.pid"
  echo "$(date -u +%FT%TZ) restarted pressure workers as PID $!"
}

start_converter() {
  nohup setsid python -u "$ROOT/utils/prepare_era5_hourly_shards.py" \
    --input-dir "$DATA_DIR" \
    --output-dir "$SHARD_DIR" \
    --expected-days 90 \
    --expected-times 24 \
    --expected-height 513 \
    --expected-width 513 \
    --poll-seconds 60 \
    --watch \
    >>"$LOG_DIR/converter.log" 2>&1 </dev/null &
  echo "$!" >"$LOG_DIR/converter.pid"
  echo "$(date -u +%FT%TZ) restarted converter as PID $!"
}

while true; do
  pressure_count=$(
    find "$DATA_DIR" -maxdepth 1 -type f \
      \( -name '*_hourly_pressure.nc' -o -name '*_hourly_pressure.grib' \) \
      -printf '%f\n' \
      | sed -E 's/_pressure\.(nc|grib)$//' \
      | sort -u \
      | wc -l
  )
  single_count=$(find "$DATA_DIR" -maxdepth 1 -name '*_hourly_single.nc' -type f | wc -l)
  shard_count=$(find "$SHARD_DIR" -maxdepth 1 -name '*_hourly.npy' -type f | wc -l)
  echo "$(date -u +%FT%TZ) pressure=$pressure_count single=$single_count shards=$shard_count"

  if [[ "$pressure_count" -lt 90 ]]; then
    if ! alive "$LOG_DIR/pressure_workers.pid" \
      && ! alive "$LOG_DIR/download.pid" \
      && ! alive "$LOG_DIR/pressure_tail.pid"; then
      start_pressure_workers
    fi
  fi

  if [[ "$single_count" -lt 90 ]] && ! alive "$LOG_DIR/single.pid"; then
    start_downloader single 2024-03-01 90 single
  fi
  if [[ "$shard_count" -lt 90 ]] && ! alive "$LOG_DIR/converter.pid"; then
    start_converter
  fi

  if [[ "$pressure_count" -ge 90 && "$single_count" -ge 90 && "$shard_count" -ge 90 ]]; then
    echo "$(date -u +%FT%TZ) hourly ERA5 download and conversion complete"
    break
  fi
  sleep "$POLL_SECONDS"
done
