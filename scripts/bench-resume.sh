#!/usr/bin/env bash
# bench-resume.sh — resume the auto-fired post-merge benchmark gate.
#
# Inverse of bench-pause.sh. Removes the `/tmp/cf-libs-bench-paused`
# flag so subsequent post-merge-benchmark.sh dispatches run normally.
#
# Usage:
#   ./scripts/bench-resume.sh                 # resume locally
#   ./scripts/bench-resume.sh --remote        # resume on default orchestrator
#   ./scripts/bench-resume.sh --remote <host> # resume on <host> over ssh
#
# T1.4 — CF-LIBS-improved-5t6n.
#
# See: docs/bench-pause.md

set -euo pipefail

FLAG="/tmp/cf-libs-bench-paused"
DEFAULT_REMOTE_HOST="brian@100.105.113.58"   # ai-proxy
REMOTE_HOST=""

usage() {
  cat <<EOF
Usage: $0 [--remote [host]]

Resume the post-merge benchmark gate by deleting $FLAG.

Options:
  --remote [host]   Apply on the orchestrator host (default: $DEFAULT_REMOTE_HOST).
                    Without --remote, the flag is removed on the local host.
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

if [[ -n "$REMOTE_HOST" ]]; then
  if ssh "$REMOTE_HOST" "test -f '$FLAG'"; then
    ssh "$REMOTE_HOST" "rm -f '$FLAG'"
    echo "[bench-resume] removed $FLAG on $REMOTE_HOST"
  else
    echo "[bench-resume] no flag at $FLAG on $REMOTE_HOST (already resumed)"
  fi
else
  if [[ -f "$FLAG" ]]; then
    rm -f "$FLAG"
    echo "[bench-resume] removed $FLAG locally"
  else
    echo "[bench-resume] no flag at $FLAG locally (already resumed)"
  fi
fi
