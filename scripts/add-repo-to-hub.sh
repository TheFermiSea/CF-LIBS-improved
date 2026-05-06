#!/usr/bin/env bash
#
# add-repo-to-hub.sh — Onboard the current beads workspace to the Dolt
# federation hub. Run from the repo root after `bd init` has succeeded.
#
# What this does:
#   1. Reads the local Dolt database name from .beads/metadata.json.
#   2. Adds the hub as a Dolt remote (remotesapi gRPC on its own port).
#   3. Pushes the local Dolt DB to the hub (creates it remotely on first push).
#   4. Adds the hub as a federation peer using the SQL port.
#   5. Verifies the peer is reachable.
#
# Required environment:
#   BEADS_HUB_HOST       Hub IP / hostname (e.g. Tailscale IP of the LXC)
#   BEADS_HUB_PASSWORD   Password from /etc/default/beads-dolt on the hub
#
# Optional environment:
#   BEADS_HUB_USER             SQL user on the hub (default: beads)
#   BEADS_HUB_SQL_PORT         SQL port (default: 3306) — used for federation peer
#   BEADS_HUB_REMOTESAPI_PORT  gRPC remotesapi port (default: 50051) — used for push/pull
#   BEADS_HUB_SCHEME           remotesapi scheme: http or https (default: http; use
#                              https only if the hub serves TLS)
#   BEADS_DB                   Override the local database name (default:
#                              auto-detected from .beads/metadata.json)
#   PEER_NAME                  Federation peer / Dolt remote name (default: hub)

set -euo pipefail

HUB_HOST="${BEADS_HUB_HOST:-}"
HUB_USER="${BEADS_HUB_USER:-beads}"
HUB_PASSWORD="${BEADS_HUB_PASSWORD:-}"
HUB_SQL_PORT="${BEADS_HUB_SQL_PORT:-3306}"
HUB_REMOTESAPI_PORT="${BEADS_HUB_REMOTESAPI_PORT:-50051}"
HUB_SCHEME="${BEADS_HUB_SCHEME:-http}"
PEER_NAME="${PEER_NAME:-hub}"
REPO_DB="${BEADS_DB:-}"

if [[ -z "$HUB_HOST" || -z "$HUB_PASSWORD" ]]; then
    cat >&2 <<EOF
Missing required environment.

Usage:
  BEADS_HUB_HOST=<ip-or-hostname> BEADS_HUB_PASSWORD=<password> \\
      $0

Optional:
  BEADS_HUB_USER             (default: beads)
  BEADS_HUB_SQL_PORT         (default: 3306)
  BEADS_HUB_REMOTESAPI_PORT  (default: 50051)
  BEADS_HUB_SCHEME           (default: http; use https for TLS-enabled hubs)
  BEADS_DB                   (default: auto from .beads/metadata.json)
  PEER_NAME                  (default: hub)
EOF
    exit 2
fi

if [[ ! -f .beads/metadata.json ]]; then
    echo "No .beads/metadata.json found. Run from a beads workspace root." >&2
    exit 1
fi

if [[ -z "$REPO_DB" ]]; then
    if ! command -v jq &>/dev/null; then
        echo "Need jq (or set BEADS_DB explicitly)." >&2
        exit 1
    fi
    REPO_DB="$(jq -r '.dolt_database // empty' .beads/metadata.json)"
fi

if [[ -z "$REPO_DB" ]]; then
    echo "Could not determine database name. Set BEADS_DB=<name>." >&2
    exit 1
fi

REMOTESAPI_URL="${HUB_SCHEME}://${HUB_HOST}:${HUB_REMOTESAPI_PORT}/${REPO_DB}"
SQL_URL="${HUB_HOST}:${HUB_SQL_PORT}/${REPO_DB}"

echo "Onboarding $REPO_DB"
echo "  remotesapi: $REMOTESAPI_URL"
echo "  SQL peer:   $SQL_URL"

# 1. Dolt remote -------------------------------------------------------------
# bd dolt remote add registers the gRPC remotesapi URL for push/pull.
if bd dolt remote list 2>/dev/null | awk '{print $1}' | grep -qx "$PEER_NAME"; then
    echo "  Dolt remote '$PEER_NAME' already configured — skipping add."
else
    bd dolt remote add "$PEER_NAME" "$REMOTESAPI_URL"
fi

# 2. Pre-create the empty database on the hub (idempotent) ------------------
# Dolt's gRPC remotesapi requires the database to exist before push. We use
# a quick mariadb-client call against the SQL port to issue CREATE DATABASE.
if ! command -v mariadb &>/dev/null && ! command -v mysql &>/dev/null; then
    echo "Need mariadb-client (or mysql) on PATH to pre-create the remote DB." >&2
    echo "Install: apt-get install mariadb-client  /  brew install mariadb" >&2
    exit 1
fi
SQL_CLIENT="$(command -v mariadb || command -v mysql)"
echo "  Ensuring database '$REPO_DB' exists on hub..."
"$SQL_CLIENT" -h "$HUB_HOST" -P "$HUB_SQL_PORT" -u "$HUB_USER" -p"$HUB_PASSWORD" \
    -e "CREATE DATABASE IF NOT EXISTS \`$REPO_DB\`" 2>&1 \
    | grep -v 'Using a password' || true

# 3. Initial push -----------------------------------------------------------
# First push needs --force because the hub's freshly-created database has
# its own initial commit with no common ancestor to the local Dolt history.
# Subsequent pushes don't need --force.
echo "  Pushing local Dolt history to hub..."
push_args=(--remote "$PEER_NAME")
# Heuristic: if the hub has only the database's auto-init commit, force-push.
# We always pass --force here; bd dolt push is a no-op when local matches
# remote, and force-push is what's expected on first onboarding anyway.
if [[ "${BEADS_HUB_FORCE_FIRST_PUSH:-1}" == "1" ]]; then
    push_args+=(--force)
fi
DOLT_REMOTE_USER="$HUB_USER" DOLT_REMOTE_PASSWORD="$HUB_PASSWORD" \
    bd dolt push "${push_args[@]}"

# 4. Federation peer ---------------------------------------------------------
# Federation uses the SQL port (handles bidirectional issue sync, conflict
# resolution, etc.) — separate from the gRPC remotesapi used by push/pull.
if bd federation list-peers 2>/dev/null | awk '{print $1}' | grep -qx "$PEER_NAME"; then
    echo "  Federation peer '$PEER_NAME' already configured — skipping add."
else
    bd federation add-peer "$PEER_NAME" "$SQL_URL" \
        --user "$HUB_USER" --password "$HUB_PASSWORD"
fi

# 5. Push the federation_peers update so subsequent syncs don't conflict ----
echo "  Pushing federation peer registration to hub..."
DOLT_REMOTE_USER="$HUB_USER" DOLT_REMOTE_PASSWORD="$HUB_PASSWORD" \
    bd dolt push --remote "$PEER_NAME" 2>&1 | tail -3 || true

# 6. Verify ------------------------------------------------------------------
echo
echo "Federation status:"
bd federation status

cat <<NEXT

Done. The repo's beads database is now mirrored to the hub.

Routine sync:
  bd federation sync                # pulls + pushes for every peer

Auto-sync via the installed git hooks already covers push-on-commit and
pull-on-merge. For a stand-alone systemd timer (cluster, server), see
docs/federation.md §"Continuous sync".
NEXT
