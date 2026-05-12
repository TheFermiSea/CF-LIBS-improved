#!/usr/bin/env bash
# telemetry_sampler.sh — Lightweight per-node telemetry sampler.
#
# Runs as a background daemon (setsid+nohup), wakes every 5s, captures a
# JSON line of GPU + CPU + memory + top-process + JAX-cache + network state,
# and appends to /cluster/shared/cf-libs-bench/telemetry/<host>/<DATE>/<HH>.jsonl
# (rotates hourly, append-only). Idempotent: re-launching just adds to the
# current file.
#
# Designed for <1% CPU per sample. No jq dependency (pure bash + awk).
#
# Usage:
#   bash scripts/telemetry_sampler.sh [interval_sec]
#
# Args:
#   interval_sec   sample interval in seconds (default: 5)
#
# Env:
#   TELEMETRY_ROOT      override output root (default: /cluster/shared/cf-libs-bench/telemetry)
#   TELEMETRY_IFACE     primary network interface (default: enp6s18)
#   TELEMETRY_JAX_CACHE jax cache dir (default: /cluster/shared/jax-cache)
#   TELEMETRY_LOGFILE   sampler stderr/stdout log (default: /tmp/telemetry-sampler.log)

set -uo pipefail

INTERVAL="${1:-5}"
HOST="$(hostname -s)"
ROOT="${TELEMETRY_ROOT:-/cluster/shared/cf-libs-bench/telemetry}"
IFACE="${TELEMETRY_IFACE:-enp6s18}"
JAX_CACHE="${TELEMETRY_JAX_CACHE:-/cluster/shared/jax-cache}"

# JSON string escape (newlines, quotes, backslashes). Stdin → stdout.
json_escape() {
  awk 'BEGIN{ORS=""} {
    gsub(/\\/, "\\\\", $0);
    gsub(/"/, "\\\"", $0);
    gsub(/\t/, " ", $0);
    gsub(/\r/, "", $0);
    print
  }'
}

# Numeric sanitizer: trims spaces, falls back to "null" if NaN/empty.
num() {
  local v
  v="$(printf '%s' "$1" | tr -d '[:space:]')"
  if [[ -z "$v" || "$v" == "N/A" || "$v" == "[N/A]" ]]; then
    printf 'null'
  else
    printf '%s' "$v"
  fi
}

sample_gpu() {
  # First GPU only (single V100S per node). On failure → all-nulls.
  local row
  row="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null | head -1)"
  if [[ -z "$row" ]]; then
    printf 'null,null,null,null,null'
    return
  fi
  local util mem_used mem_total power temp
  IFS=',' read -r util mem_used mem_total power temp <<<"$row"
  printf '%s,%s,%s,%s,%s' \
    "$(num "$util")" "$(num "$mem_used")" "$(num "$mem_total")" \
    "$(num "$power")" "$(num "$temp")"
}

sample_loadavg() {
  # Returns "1m,5m,15m" — three numbers from /proc/loadavg.
  awk '{printf "%s,%s,%s", $1,$2,$3}' /proc/loadavg 2>/dev/null \
    || printf 'null,null,null'
}

sample_mem() {
  # Returns "used_mb,total_mb,free_mb" from free -m.
  free -m 2>/dev/null | awk '/^Mem:/ {printf "%s,%s,%s",$3,$2,$4}' \
    || printf 'null,null,null'
}

# Top N processes by sort field ($3=cpu, $4=mem). Outputs JSON array of objects.
# Args: $1 = sort field (3 or 4), $2 = N
sample_top_procs() {
  local field="$1" n="$2"
  # ps -eo pid,user,%cpu,%mem,comm,args : sort by field desc, take top N.
  # Build JSON array manually.
  ps -eo pid,%cpu,%mem,comm --no-headers 2>/dev/null \
    | sort -k"$field" -rn 2>/dev/null \
    | head -"$n" \
    | awk -v n="$n" '
        BEGIN { printf "[" }
        NR>1 { printf "," }
        {
          pid = $1; cpu = $2; mem = $3;
          # comm = remainder; rejoin in case of spaces
          cmd = ""
          for (i = 4; i <= NF; i++) cmd = cmd (i==4?"":" ") $i
          # escape backslashes and quotes
          gsub(/\\/, "\\\\", cmd)
          gsub(/"/,  "\\\"", cmd)
          # cap command length
          if (length(cmd) > 80) cmd = substr(cmd, 1, 80)
          printf "{\"pid\":%s,\"cpu\":%s,\"mem\":%s,\"cmd\":\"%s\"}", pid, cpu, mem, cmd
        }
        END { printf "]" }
      '
}

sample_jax_cache() {
  # Returns "size_mb,file_count" — falls back to "null,null" if dir missing.
  if [[ ! -d "$JAX_CACHE" ]]; then
    printf 'null,null'
    return
  fi
  # Use du -sm + find. Both are fast on NFS for shallow trees.
  local size files
  size="$(du -sm "$JAX_CACHE" 2>/dev/null | awk '{print $1}')"
  files="$(find "$JAX_CACHE" -maxdepth 3 -type f 2>/dev/null | wc -l)"
  printf '%s,%s' "$(num "$size")" "$(num "$files")"
}

# RX/TX bytes for $IFACE. Returns "rx_bytes,tx_bytes" or "null,null".
sample_net() {
  local rx tx
  rx="$(cat /sys/class/net/"$IFACE"/statistics/rx_bytes 2>/dev/null)"
  tx="$(cat /sys/class/net/"$IFACE"/statistics/tx_bytes 2>/dev/null)"
  printf '%s,%s' "$(num "$rx")" "$(num "$tx")"
}

emit_sample() {
  local ts gpu loadavg mem net jax top_cpu top_mem
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  gpu="$(sample_gpu)"
  loadavg="$(sample_loadavg)"
  mem="$(sample_mem)"
  net="$(sample_net)"
  jax="$(sample_jax_cache)"
  top_cpu="$(sample_top_procs 2 3)"   # field 2 = %cpu in ps output above
  top_mem="$(sample_top_procs 3 3)"   # field 3 = %mem

  IFS=',' read -r g_util g_mu g_mt g_p g_t <<<"$gpu"
  IFS=',' read -r la1 la5 la15 <<<"$loadavg"
  IFS=',' read -r m_used m_tot m_free <<<"$mem"
  IFS=',' read -r rx tx <<<"$net"
  IFS=',' read -r jc_size jc_files <<<"$jax"

  # Compose JSON line. No trailing newline inside; printf adds one.
  printf '{"ts":"%s","host":"%s","gpu_util":%s,"gpu_mem_mb":%s,"gpu_mem_total_mb":%s,"gpu_power_w":%s,"gpu_temp_c":%s,"loadavg":[%s,%s,%s],"mem_used_mb":%s,"mem_total_mb":%s,"mem_free_mb":%s,"net_rx_bytes":%s,"net_tx_bytes":%s,"top_cpu":%s,"top_mem":%s,"jax_cache_size_mb":%s,"jax_cache_files":%s}\n' \
    "$ts" "$HOST" \
    "$g_util" "$g_mu" "$g_mt" "$g_p" "$g_t" \
    "$la1" "$la5" "$la15" \
    "$m_used" "$m_tot" "$m_free" \
    "$rx" "$tx" \
    "$top_cpu" "$top_mem" \
    "$jc_size" "$jc_files"
}

current_outfile() {
  local d h
  d="$(date -u +%Y-%m-%d)"
  h="$(date -u +%H)"
  printf '%s/%s/%s/%s.jsonl' "$ROOT" "$HOST" "$d" "$h"
}

main() {
  mkdir -p "$ROOT/$HOST" 2>/dev/null || true
  echo "[telemetry_sampler] pid=$$ host=$HOST interval=${INTERVAL}s out_root=$ROOT" >&2

  local outfile prev_outfile=""
  while :; do
    outfile="$(current_outfile)"
    if [[ "$outfile" != "$prev_outfile" ]]; then
      mkdir -p "$(dirname "$outfile")" 2>/dev/null || true
      prev_outfile="$outfile"
    fi
    emit_sample >>"$outfile" 2>/dev/null || true
    sleep "$INTERVAL"
  done
}

main "$@"
