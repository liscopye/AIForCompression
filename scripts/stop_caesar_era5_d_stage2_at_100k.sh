#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_PID="${CAESAR_D_STAGE2_TRAIN_PID:-3310608}"
CHECKPOINT="${CAESAR_D_STAGE2_100K_CHECKPOINT:-$ROOT/checkpoints/caesar_era5_d_stage2_full_200k/lam3em4_stage2_from_original_lr1em4_200k_update100000.pt}"
POLL_SECONDS="${CAESAR_D_STAGE2_STOP_POLL_SECONDS:-60}"
LOG="${CAESAR_D_STAGE2_STOP_LOG:-$ROOT/logs/caesar_era5_d_stage2_full_200k/stop_at_100k.log}"

mkdir -p "$(dirname "$LOG")"

while [[ ! -s "$CHECKPOINT" ]]; do
  sleep "$POLL_SECONDS"
done

size_before="$(stat -c '%s' "$CHECKPOINT")"
sleep 30
size_after="$(stat -c '%s' "$CHECKPOINT")"
if [[ "$size_before" != "$size_after" ]]; then
  printf 'checkpoint_not_stable size_before=%s size_after=%s\n' \
    "$size_before" "$size_after" >>"$LOG"
  exit 1
fi

if [[ ! -r "/proc/$TRAIN_PID/cmdline" ]]; then
  printf 'training_already_stopped checkpoint=%s\n' "$CHECKPOINT" >>"$LOG"
  exit 0
fi

cmdline="$(tr '\0' ' ' <"/proc/$TRAIN_PID/cmdline")"
if [[ "$cmdline" != *"finetune_caesar_era5.py"* || "$cmdline" != *"lam3em4_stage2_from_original_lr1em4_200k.pt"* ]]; then
  printf 'pid_validation_failed pid=%s cmdline=%q\n' "$TRAIN_PID" "$cmdline" >>"$LOG"
  exit 1
fi

date -u '+sigint_sent_utc=%Y-%m-%dT%H:%M:%SZ' >>"$LOG"
printf 'pid=%s checkpoint=%s bytes=%s\n' \
  "$TRAIN_PID" "$CHECKPOINT" "$size_after" >>"$LOG"
kill -INT "$TRAIN_PID"

for _ in $(seq 1 120); do
  if [[ ! -d "/proc/$TRAIN_PID" ]]; then
    date -u '+training_stopped_utc=%Y-%m-%dT%H:%M:%SZ' >>"$LOG"
    exit 0
  fi
  sleep 1
done

printf 'training_still_running_after_sigint pid=%s\n' "$TRAIN_PID" >>"$LOG"
exit 1
