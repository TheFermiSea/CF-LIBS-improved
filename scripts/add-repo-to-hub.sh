#!/usr/bin/env bash
#
# add-repo-to-hub.sh — Onboard the current beads workspace to the Dolt
# federation hub. Run from the repo root after `bd init` has succeeded.
#
# What this does:
#   1. Reads the local Dolt database name from .beads/metadata.json.
#   2. Adds the hub as a Dolt remote.
#   3. Pushes the local Dolt DB to the hub (creates it remotely on first push).
#   4. Adds the hub as a federation peer with stored credentials.
#   5. Verifies the peer is reachable.
#
# Required environment:
#   BEADS_HUB_HOST       Hub IP / hostname (e.g. Tailscale IP of the LXC)
#   BEADS_HUB_PASSWORD   Password from /etc/default/beads-dolt on the hub
#
# Optional environment:
#   BEADS_HUB_USER       SQL user on the hub (default: beads)
#   BEADS_HUB_SQL_PORT   SQL port (default: 3306)
#   BEADS_DB             Override the local database name (default:
#                        auto-detected from .beads/metadata.json)
#   PEER_NAME            Federation peer name (default: hub)

set -euo pipefail

HUB_HOST="${BEADS_HUB_HOST:-}"
HUB_USER="${BEADS_HUB_USER:-beads}"
HUB_PASSWORD="${BEADS_HUB_PASSWORD:-}"
HUB_PORT="${BEADS_HUB_SQL_PORT:-3306}"
PEER_NAME="${PEER_NAME:-hub}"
REPO_DB="${BEADS_DB:-}"

if [[ -z "$HUB_HOST" || -z "$HUB_PASSWORD" ]]; then
    cat >&2 <<EOF
Missing required environment.

Usage:
  BEADS_HUB_HOST=<ip-or-hostname> BEADS_HUB_PASSWORD=<password> \\
      $0

Optional:
  BEADS_HUB_USER (default: beads)
  BEADS_HUB_SQL_PORT (default: 3306)
  BEADS_DB (default: auto from .beads/metadata.json)
  PEER_NAME (default: hub)
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

REMOTE_URL="$HUB_HOST:$HUB_PORT/$REPO_DB"

echo "Onboarding $REPO_DB to $REMOTE_URL ..."

# 1. Dolt remote -------------------------------------------------------------
if bd dolt remote list 2>/dev/null | awk '{print $1}' | grep -qx "$PEER_NAME"; then
    echo "  Dolt remote '$PEER_NAME' already configured — skipping add."
else
    bd dolt remote add "$PEER_NAME" "$REMOTE_URL"
fi

# 2. Initial push (creates the database on the hub) --------------------------
echo "  Pushing local Dolt history to hub..."
DOLT_REMOTE_USER="$HUB_USER" DOLT_REMOTE_PASSWORD="$HUB_PASSWORD" \
    bd dolt push --remote "$PEER_NAME"

# 3. Federation peer ---------------------------------------------------------
if bd federation list-peers 2>/dev/null | awk '{print $1}' | grep -qx "$PEER_NAME"; then
    echo "  Federation peer '$PEER_NAME' already configured — skipping add."
else
    bd federation add-peer "$PEER_NAME" "$REMOTE_URL" \
        --user "$HUB_USER" --password "$HUB_PASSWORD"
fi

# 4. Verify ------------------------------------------------------------------
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
