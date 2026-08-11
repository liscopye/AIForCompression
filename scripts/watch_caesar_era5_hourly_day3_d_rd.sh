#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARD_DIR="${ERA5_HOURLY_SHARD_DIR:-/workspace/Data/ERA5/hourly_center512_shards_20240301_90d}"
TARGET="$SHARD_DIR/2024-03-03_hourly.npy"
POLL_SECONDS="${ERA5_DAY3_POLL_SECONDS:-600}"

while [[ ! -f "$TARGET" ]]; do
  printf '%s waiting for %s\n' "$(date -u +%FT%TZ)" "$TARGET"
  sleep "$POLL_SECONDS"
done

printf '%s found %s; starting independent D RD diagnostic\n' \
  "$(date -u +%FT%TZ)" "$TARGET"
exec bash "$ROOT/scripts/run_caesar_era5_hourly_day3_d_rd.sh"
