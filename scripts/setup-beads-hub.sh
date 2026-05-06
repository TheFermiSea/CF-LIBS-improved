#!/usr/bin/env bash
#
# setup-beads-hub.sh — Provision a Dolt SQL server inside an LXC for beads
# federation. Run inside the container as root (or via sudo).
#
# What this does:
#   1. Creates an unprivileged 'dolt' system user.
#   2. Installs a pinned Dolt binary at /usr/local/bin/dolt.
#   3. Generates a strong password and writes /etc/default/beads-hub.
#   4. Installs and enables the beads-hub.service systemd unit.
#   5. Starts the server listening on 0.0.0.0:3306 (SQL) and 0.0.0.0:50051
#      (remotesapi gRPC). Restrict access at the network layer (Tailscale
#      ACLs / firewall) — these ports are NOT auth-hardened against the
#      open internet.
#
# Idempotent: re-running upgrades the binary if DOLT_VERSION changed and
# leaves credentials untouched.
#
# Environment overrides:
#   DOLT_VERSION  Dolt release to install (default below; bump as needed)
#   DOLT_USER     SQL username for clients (default: beads)
#   DATA_DIR      Dolt data directory (default: /var/lib/dolt)

set -euo pipefail

DOLT_VERSION="${DOLT_VERSION:-1.87.0}"
DOLT_USER="${DOLT_USER:-beads}"
DATA_DIR="${DATA_DIR:-/var/lib/dolt}"
SERVICE_FILE="/etc/systemd/system/beads-hub.service"
ENV_FILE="/etc/default/beads-hub"
BIN_PATH="/usr/local/bin/dolt"

if [[ "$EUID" -ne 0 ]]; then
    echo "Run as root (use sudo)." >&2
    exit 1
fi

# 1. System user --------------------------------------------------------------
if ! id -u dolt &>/dev/null; then
    useradd --system --no-create-home --shell /usr/sbin/nologin \
        --home-dir "$DATA_DIR" dolt
fi

# 2. Install / upgrade Dolt binary -------------------------------------------
ARCH="$(uname -m)"
case "$ARCH" in
    x86_64)  DOLT_ARCH="amd64" ;;
    aarch64) DOLT_ARCH="arm64" ;;
    *) echo "Unsupported arch: $ARCH" >&2; exit 1 ;;
esac

needs_install=1
if [[ -x "$BIN_PATH" ]]; then
    current="$("$BIN_PATH" version 2>/dev/null | awk '/dolt version/ {print $3; exit}')"
    if [[ "$current" == "$DOLT_VERSION" ]]; then
        needs_install=0
    fi
fi

if [[ "$needs_install" -eq 1 ]]; then
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    url="https://github.com/dolthub/dolt/releases/download/v${DOLT_VERSION}/dolt-linux-${DOLT_ARCH}.tar.gz"
    echo "Downloading $url"
    curl -fsSL "$url" -o "$tmp/dolt.tar.gz"
    tar -xzf "$tmp/dolt.tar.gz" -C "$tmp"
    install -m 0755 "$tmp/dolt-linux-${DOLT_ARCH}/bin/dolt" "$BIN_PATH"
fi

echo "Dolt: $("$BIN_PATH" version | head -1)"

# 3. Data dir ----------------------------------------------------------------
install -d -o dolt -g dolt -m 0750 "$DATA_DIR"

# 4. Credentials -------------------------------------------------------------
if [[ ! -f "$ENV_FILE" ]]; then
    password="$(openssl rand -base64 48 | tr -d '/+=\n' | head -c 40)"
    install -m 0640 -o root -g dolt /dev/null "$ENV_FILE"
    cat >"$ENV_FILE" <<EOF
# Generated $(date -Iseconds) by setup-beads-hub.sh
BEADS_DOLT_USER=$DOLT_USER
BEADS_DOLT_PASSWORD=$password
EOF
    cat <<BANNER

============================================================
Generated credentials for the beads federation hub.
Saved to $ENV_FILE (root:dolt 0640).

  user:     $DOLT_USER
  password: $password

Save this password in your password manager — every client
needs it to push/pull/federate. View later with:
  sudo cat $ENV_FILE
============================================================
BANNER
fi

# 5. systemd unit ------------------------------------------------------------
cat >"$SERVICE_FILE" <<'UNIT'
[Unit]
Description=Dolt SQL Server (beads federation hub)
Documentation=https://docs.dolthub.com/sql-reference/server/server-configuration
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=dolt
Group=dolt
WorkingDirectory=/var/lib/dolt
EnvironmentFile=/etc/default/beads-hub
ExecStart=/usr/local/bin/dolt sql-server \
    --host 0.0.0.0 \
    --port 3306 \
    --remotesapi-port 50051 \
    --user ${BEADS_DOLT_USER} \
    --password ${BEADS_DOLT_PASSWORD} \
    --data-dir /var/lib/dolt
Restart=on-failure
RestartSec=5s
LimitNOFILE=65536
TimeoutStopSec=30s

# Hardening
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/dolt
PrivateTmp=true
NoNewPrivileges=true
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
RestrictSUIDSGID=true
LockPersonality=true

[Install]
WantedBy=multi-user.target
UNIT

# 6. Enable + start ----------------------------------------------------------
systemctl daemon-reload
systemctl enable --now beads-hub.service

# 7. Status ------------------------------------------------------------------
sleep 2
systemctl --no-pager --lines=0 status beads-hub.service || true
echo
echo "Listening sockets:"
ss -tlnp 2>/dev/null | grep -E ':(3306|50051)\b' || \
    echo "  (none yet — give the server a moment, then re-run: ss -tlnp | grep dolt)"

cat <<NEXT

Done. Next:
  1. Confirm Tailscale (or whatever VPN you're using) is up:
       tailscale status
       tailscale ip -4
  2. Restrict 3306 + 50051 to the Tailnet / management VLAN at the firewall.
  3. On each client, follow the runbook in docs/federation.md.
NEXT
