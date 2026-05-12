#!/usr/bin/env bash
# Quick status of Exp 1 (identifier shootout). Read-only.
set -uo pipefail
echo "=== Exp 1 status @ $(date '+%Y-%m-%d %H:%M:%S') ==="
for s in 1 2 3; do
  case $s in 1) h=10.0.0.20;; 2) h=10.0.0.21;; 3) h=10.0.0.22;; esac
  echo "--- shard $s ($h) ---"
  ssh -o ConnectTimeout=5 root@$h "
    pgrep -f 'python.*parameter_sweep' >/dev/null && echo 'STATE: running' || echo 'STATE: stopped'
    echo 'iters_done:' \$(ls -d /cluster/shared/cf-libs-bench/results/exp001/shard${s}/iter-* 2>/dev/null | wc -l)
    echo 'manifest_rows:' \$(wc -l </cluster/shared/cf-libs-bench/results/exp001/shard${s}/manifest.jsonl 2>/dev/null || echo 0)
    echo 'log_lines:' \$(wc -l </tmp/exp001-shard${s}.log 2>/dev/null || echo 0)
  " 2>&1 | sed 's/^/  /'
done
