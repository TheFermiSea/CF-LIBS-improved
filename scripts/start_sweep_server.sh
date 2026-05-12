#!/usr/bin/env bash
# Launcher wrapper for cflibs.parameter_sweep_server.
#
# Usage (typical):
#
#   scripts/start_sweep_server.sh                    # default 127.0.0.1:8501
#   SWEEP_PORT=8602 scripts/start_sweep_server.sh    # custom port
#   SWEEP_HOST=0.0.0.0 SWEEP_PORT=8501 ./start_sweep_server.sh
#
# The script forwards every env var prefixed `SWEEP_` to the daemon.
# Relevant variables (see ServerConfig):
#
#   SWEEP_HOST                (default 127.0.0.1)
#   SWEEP_PORT                (default 8501)
#   SWEEP_DB_PATH             (default ASD_da/libs_production.db)
#   SWEEP_DATA_DIR            (default data)
#   SWEEP_BASIS_DIR           (default output/basis_libraries)
#   SWEEP_SYNTHETIC_CORPUS    (optional)
#   SWEEP_QUEUE_MAX           (default 4)
#   SWEEP_DRAIN_SEC           (default 60)
#   SWEEP_QUICK               (0|1, default 0)
#   SWEEP_JAX_CACHE_DIR       (optional NFS path for persistent JIT cache)
#   SWEEP_LOG_LEVEL           (default INFO)

set -euo pipefail

# Locate the repository root regardless of where the script is invoked.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

# Default JAX to CPU unless the operator explicitly chose otherwise.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

# Honour an opt-in persistent JIT cache. Caller can pre-create the NFS
# directory; we lazily mkdir inside the daemon if it does not exist.
if [[ -n "${SWEEP_JAX_CACHE_DIR:-}" ]]; then
  export JAX_COMPILATION_CACHE_DIR="$SWEEP_JAX_CACHE_DIR"
fi

PYBIN="${PYTHON:-python3}"

exec "$PYBIN" -m cflibs.parameter_sweep_server.server "$@"
