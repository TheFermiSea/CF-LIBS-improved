#!/usr/bin/env bash
# start_telemetry.sh — Install + launch telemetry_sampler.sh on all vasp nodes.
#
# - rsyncs the sampler to /usr/local/sbin/telemetry_sampler.sh on each node
# - kills any prior sampler (clean shutdown)
# - relaunches under setsid+nohup so it survives ssh logout
# - verifies a sample lands in the NFS output dir within 15s
#
# Usage:
#   bash scripts/start_telemetry.sh [interval_sec]
#
# Args:
#   interval_sec   forwarded to sampler (default: 5)
#
# Env:
#   TELEMETRY_HOSTS  override host list (default: "vasp-01 vasp-02 vasp-03")

set -uo pipefail

INTERVAL="${1:-5}"
HOSTS="${TELEMETRY_HOSTS:-vasp-01 vasp-02 vasp-03}"
ROOT="${TELEMETRY_ROOT:-/cluster/shared/cf-libs-bench/telemetry}"

declare -A HOST_IP=(
  [vasp-01]=10.0.0.20
  [vasp-02]=10.0.0.21
  [vasp-03]=10.0.0.22
)

SCRIPT_SRC="$(cd "$(dirname "$0")" && pwd)/telemetry_sampler.sh"
if [[ ! -f "$SCRIPT_SRC" ]]; then
  echo "FATAL: $SCRIPT_SRC not found" >&2
  exit 1
fi

# Strict mode for deploy phase; per-host failures are reported but don't abort
# the entire roll-out.
overall_rc=0

for h in $HOSTS; do
  ip="${HOST_IP[$h]:-$h}"
  echo "=== $h ($ip) ==="

  # Step 1: rsync sampler. (scp is fine too; rsync is idempotent.)
  if ! scp -q -o ConnectTimeout=5 "$SCRIPT_SRC" "root@${ip}:/usr/local/sbin/telemetry_sampler.sh"; then
    echo "  ERR: scp failed"; overall_rc=1; continue
  fi
  ssh -o ConnectTimeout=5 root@"$ip" "chmod +x /usr/local/sbin/telemetry_sampler.sh"

  # Step 2: kill any prior sampler. pgrep on the script name is reliable.
  ssh -o ConnectTimeout=5 root@"$ip" \
    "pkill -f telemetry_sampler.sh 2>/dev/null; sleep 0.5; pkill -9 -f telemetry_sampler.sh 2>/dev/null; true"

  # Step 3: launch under setsid+nohup. setsid detaches from controlling tty so
  # the process keeps running after ssh disconnects.
  ssh -o ConnectTimeout=5 root@"$ip" \
    "setsid nohup /usr/local/sbin/telemetry_sampler.sh $INTERVAL >/tmp/telemetry-sampler.log 2>&1 </dev/null &"

  # Step 4: tiny wait + verify pid + verify NFS write.
  sleep 2
  pid="$(ssh -o ConnectTimeout=5 root@"$ip" 'pgrep -f telemetry_sampler.sh | head -1' 2>/dev/null || echo '')"
  if [[ -z "$pid" ]]; then
    echo "  ERR: no sampler pid found post-launch"; overall_rc=1; continue
  fi
  echo "  started pid=$pid"
done

# Step 5: verify samples land in NFS.
# We verify by SSHing into each node (since the deploy host — typically
# ai-proxy — does NOT mount /cluster/shared). This avoids false negatives.
echo
echo "=== verifying NFS writes (waiting up to 20s) ==="
ok=1
for i in 1 2 3 4 5 6; do
  sleep 3
  missing=0
  for h in $HOSTS; do
    ip="${HOST_IP[$h]:-$h}"
    today="$(date -u +%Y-%m-%d)"
    hour="$(date -u +%H)"
    f="$ROOT/$h/$today/$hour.jsonl"
    lines="$(ssh -o ConnectTimeout=5 root@"$ip" "wc -l <\"$f\" 2>/dev/null" 2>/dev/null || echo 0)"
    if [[ "${lines:-0}" -lt 1 ]]; then
      missing=1
    fi
  done
  if [[ $missing -eq 0 ]]; then
    echo "  all hosts have NFS samples after $((i*3))s"
    ok=0
    break
  fi
done
if [[ $ok -ne 0 ]]; then
  echo "  WARN: not all hosts wrote a sample within 20s"
  for h in $HOSTS; do
    ip="${HOST_IP[$h]:-$h}"
    today="$(date -u +%Y-%m-%d)"
    hour="$(date -u +%H)"
    f="$ROOT/$h/$today/$hour.jsonl"
    lines="$(ssh -o ConnectTimeout=5 root@"$ip" "wc -l <\"$f\" 2>/dev/null" 2>/dev/null || echo 0)"
    if [[ "${lines:-0}" -ge 1 ]]; then
      echo "  OK   $h: $lines lines in $f"
    else
      echo "  MISS $h: $f"
    fi
  done
  overall_rc=1
fi

echo
echo "Telemetry sampler deployed. Output: $ROOT/<host>/<UTC-date>/<UTC-hour>.jsonl"
exit "$overall_rc"
