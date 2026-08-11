#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ERA5_HOURLY_DATA_DIR:-/workspace/Data/ERA5/hourly_center512_20240301_90d}"
LOG_DIR="${ERA5_HOURLY_LOG_DIR:-$ROOT/logs/era5_hourly_center512_20240301_90d}"
BOUNDARY_FILE="$DATA_DIR/2024-04-14_hourly_pressure.nc"
POLL_SECONDS="${ERA5_PRESSURE_GUARD_POLL_SECONDS:-60}"

while [[ ! -f "$BOUNDARY_FILE" ]]; do
  sleep "$POLL_SECONDS"
done

head_pid=$(cat "$LOG_DIR/download.pid")
tail_pid=$(cat "$LOG_DIR/pressure_tail.pid")
if ! kill -0 "$tail_pid" 2>/dev/null; then
  echo "Boundary reached, but tail worker $tail_pid is not alive; leaving head worker running."
  exit 1
fi

if kill -0 "$head_pid" 2>/dev/null; then
  kill -- "-$head_pid" 2>/dev/null || kill "$head_pid"
  echo "Boundary reached; stopped head worker $head_pid after completing 2024-04-14."
else
  echo "Boundary reached; head worker $head_pid had already stopped."
fi
