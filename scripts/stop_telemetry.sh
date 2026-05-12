#!/usr/bin/env bash
# stop_telemetry.sh — Clean shutdown of telemetry_sampler.sh on all nodes.
#
# Usage:
#   bash scripts/stop_telemetry.sh

set -uo pipefail

HOSTS="${TELEMETRY_HOSTS:-vasp-01 vasp-02 vasp-03}"
declare -A HOST_IP=(
  [vasp-01]=10.0.0.20
  [vasp-02]=10.0.0.21
  [vasp-03]=10.0.0.22
)

for h in $HOSTS; do
  ip="${HOST_IP[$h]:-$h}"
  echo "=== $h ($ip) ==="
  pids="$(ssh -o ConnectTimeout=5 root@"$ip" 'pgrep -f telemetry_sampler.sh' 2>/dev/null || echo '')"
  if [[ -z "$pids" ]]; then
    echo "  no sampler running"
    continue
  fi
  echo "  killing pids: $(echo "$pids" | tr '\n' ' ')"
  ssh -o ConnectTimeout=5 root@"$ip" \
    "pkill -f telemetry_sampler.sh 2>/dev/null; sleep 0.5; pkill -9 -f telemetry_sampler.sh 2>/dev/null; true"
  sleep 1
  remaining="$(ssh -o ConnectTimeout=5 root@"$ip" 'pgrep -f telemetry_sampler.sh' 2>/dev/null || echo '')"
  if [[ -n "$remaining" ]]; then
    echo "  WARN: pids still alive: $remaining"
  else
    echo "  stopped"
  fi
done
