#!/usr/bin/env bash
# bench-pause.sh — pause the auto-fired post-merge benchmark gate.
#
# Every PR-push to dev triggers
# `beefcake-swarm/scripts/post-merge-benchmark.sh classify+run` via the
# orchestrator's pre-PR benchmark gate (python/benchmark_gate.py). That
# script drains llama.cpp on vasp-03 and runs a 30-90 min benchmark.
# During parameter sweeps and other intentional 24h experiments this
# competes for GPU and ruins the sweep's wall-time budget.
#
# This helper creates `/tmp/cf-libs-bench-paused` on the orchestrator
# host (default: localhost; ai-proxy when run on the cluster). The
# patched post-merge-benchmark.sh short-circuits its `do_classify` (echo
# "skip") and `do_run` (log + return 0) paths when that file exists or
# when `BENCH_PAUSED=1` is in the environment.
#
# Usage:
#   ./scripts/bench-pause.sh                 # pause locally
#   ./scripts/bench-pause.sh --remote        # pause on default orchestrator host
#   ./scripts/bench-pause.sh --remote <host> # pause on <host> over ssh
#   ./scripts/bench-pause.sh --reason "..."  # annotate the pause flag
#
# T1.4 — CF-LIBS-improved-5t6n.
#
# See: docs/bench-pause.md

set -euo pipefail

FLAG="/tmp/cf-libs-bench-paused"
DEFAULT_REMOTE_HOST="brian@100.105.113.58"   # ai-proxy
REMOTE_HOST=""
REASON=""

usage() {
  cat <<EOF
Usage: $0 [--remote [host]] [--reason TEXT]

Pause the post-merge benchmark gate by creating $FLAG.

Options:
  --remote [host]   Apply on the orchestrator host (default: $DEFAULT_REMOTE_HOST).
                    Without --remote, the flag is created on the local host.
  --reason TEXT     Free-form reason written into the flag file.
  -h, --help        Show this message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote)
      if [[ $# -ge 2 && "$2" != --* ]]; then
        REMOTE_HOST="$2"
        shift 2
      else
        REMOTE_HOST="$DEFAULT_REMOTE_HOST"
        shift
      fi
      ;;
    --reason)
      REASON="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

body=""
body+="paused_at=$(date -Iseconds)"$'\n'
body+="paused_by=${USER:-unknown}@$(hostname)"$'\n'
[[ -n "$REASON" ]] && body+="reason=$REASON"$'\n'

if [[ -n "$REMOTE_HOST" ]]; then
  printf '%s' "$body" | ssh "$REMOTE_HOST" "cat > '$FLAG'"
  echo "[bench-pause] created $FLAG on $REMOTE_HOST"
  ssh "$REMOTE_HOST" "ls -la '$FLAG'"
else
  printf '%s' "$body" > "$FLAG"
  echo "[bench-pause] created $FLAG locally"
  ls -la "$FLAG"
fi
