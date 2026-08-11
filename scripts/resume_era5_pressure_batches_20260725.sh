#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="/workspace/Data/ERA5/hourly_center512_20240301_90d"
LOG_DIR="$ROOT/logs/era5_hourly_center512_20240301_90d"
RESUME_LOG_DIR="$LOG_DIR/pressure_resume_20260725"
mkdir -p "$RESUME_LOG_DIR"

request_ids=(
  37c6a303-a126-44a2-b2e5-cae98cc41a35
  56237a3a-fd33-41b2-aa3c-509a77c0b1bc
  ae886205-00b1-413e-a6c6-dbdfd37fd45e
  b0e83529-55d3-43dc-93b8-89dd24235f51
)
date_groups=(
  "2024-04-16 2024-04-17 2024-04-18 2024-04-19 2024-04-20 2024-04-21 2024-04-22 2024-04-23"
  "2024-05-01 2024-05-02 2024-05-03 2024-05-04 2024-05-05 2024-05-06 2024-05-07"
  "2024-05-17 2024-05-18"
  "2024-05-19 2024-05-20"
)

pids=()
for index in "${!request_ids[@]}"; do
  read -r -a dates <<<"${date_groups[$index]}"
  python -u "$ROOT/utils/resume_era5_pressure_batch.py" \
    --request-id "${request_ids[$index]}" \
    --dates "${dates[@]}" \
    --data-dir "$DATA_DIR" \
    --max-retries 500 \
    --retry-wait 2 \
    >"$RESUME_LOG_DIR/batch_$((index + 1)).log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

nohup setsid bash "$ROOT/scripts/run_era5_pressure_workers.sh" \
  >>"$LOG_DIR/pressure_workers.log" 2>&1 </dev/null &
echo "$!" >"$LOG_DIR/pressure_workers.pid"

if ! pgrep -f "bash $ROOT/scripts/watch_era5_hourly_download.sh" >/dev/null; then
  nohup setsid bash "$ROOT/scripts/watch_era5_hourly_download.sh" \
    >>"$LOG_DIR/watchdog.log" 2>&1 </dev/null &
  echo "$!" >"$LOG_DIR/watchdog.pid"
fi

exit "$failed"
