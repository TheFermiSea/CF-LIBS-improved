# Beads federation hub

This repository tracks issues with [beads (`bd`)](https://github.com/steveyegge/beads),
which stores its data in a Dolt SQL database. Federation lets multiple environments
(laptop, cluster, server, cloud) share the same issue data by syncing through a
central Dolt SQL server.

This document describes the hub design we use here and the runbook for bringing
it up and onboarding repos.

## Architecture

```
                        ┌──────────────────────────┐
                        │   beads-lxc              │
                        │   ──────────────         │
                        │   dolt sql-server        │
                        │   :3306  (SQL)           │
                        │   :50051 (remotesapi)    │
                        │                          │
                        │   /var/lib/dolt/         │
                        │     CF_LIBS_improved/    │
                        │     beefcake_swarm/      │
                        │     <one dir per repo>/  │
                        └──────────────┬───────────┘
                                       │ Tailscale (WireGuard)
                ┌──────────────────────┼──────────────────────┐
                │                      │                      │
        ┌───────┴─────┐        ┌───────┴───────┐      ┌───────┴───────┐
        │   laptop    │        │   cluster     │      │   cloud / CI  │
        │   bd 1.0.x  │        │   bd 1.0.x    │      │   bd 1.0.x    │
        └─────────────┘        └───────────────┘      └───────────────┘
```

- **One LXC, many repos.** Each beads workspace gets its own Dolt database
  (`/var/lib/dolt/<dbname>`). Database names match `.beads/metadata.json`'s
  `dolt_database` field — usually a sanitised form of the repo name.
- **One auth identity.** A single SQL user (`beads` by default) has privileges
  on every database. Network access is gated by Tailscale ACLs / firewall, so
  the SQL password is defence-in-depth, not the only line of defence.
- **Two ports.** SQL on `3306`, gRPC remotesapi on `50051`. Both must be
  reachable from every client. Restrict both to your Tailnet / management VLAN.
- **No public exposure.** Dolt's auth surface has not been hardened against the
  open internet. Treat the hub as an internal service.

## Requirements

- Proxmox (or any LXC host).
- Tailscale (or another VPN) joining the LXC and every client.
- An LXC template that can run a single Go binary — Debian 12 / Ubuntu 24.04
  minimal is fine. ~1 GB RAM, 10 GB disk to start.

## Hub bring-up

### 1. Provision the LXC

In Proxmox:

- Hostname: `beads-lxc`
- Distro: Debian 12 minimal
- Resources: 2 vCPU, 1–2 GB RAM, 10 GB disk
- Network: static IP on your management VLAN
- Features: `nesting=1` is fine; no special device passthrough needed

Then on the host:

```bash
pct enter <vmid>            # or ssh root@<ip>
apt update && apt install -y curl ca-certificates openssl tar
curl -fsSL https://tailscale.com/install.sh | sh
tailscale up --hostname=beads-lxc
tailscale ip -4             # note this — clients will use it
```

### 2. Run the setup script

Copy `scripts/setup-beads-lxc.sh` from this repo into the LXC and execute it:

```bash
scp scripts/setup-beads-lxc.sh root@beads-lxc:/root/
ssh root@beads-lxc "bash /root/setup-beads-lxc.sh"
```

The script will:

1. Create an unprivileged `dolt` system user.
2. Install the pinned Dolt binary (`DOLT_VERSION` env var to override).
3. Generate a random 40-character password and write `/etc/default/beads-dolt`.
4. Install `beads-dolt.service` and enable+start it.

Save the printed password — every client needs it. To retrieve later:

```bash
ssh root@beads-lxc "cat /etc/default/beads-dolt"
```

### 3. Restrict the network surface

Use whatever tag your Tailnet ACL already defines for backend infrastructure
(e.g. `tag:server`). The LXC advertises that tag at `tailscale up` time:

```bash
tailscale up --hostname=beads-lxc --advertise-tags=tag:server --ssh
```

Tailscale ACL fragment locking ports `3306` and `50051` to clients tagged
`tag:beads-client` (or `autogroup:member`, etc.):

```hujson
"acls": [
  { "action": "accept",
    "src":    ["autogroup:member"],
    "dst":    ["tag:server:3306", "tag:server:50051"] }
]
```

If `tag:server` is not yet defined in your tailnet, add it to `tagOwners`
in the admin console first, otherwise `tailscale up` returns
`requested tags [tag:server] are invalid or not permitted`.

### 4. Smoke test

From any client on the Tailnet:

```bash
nc -zv <hub-tailscale-ip> 3306    # SQL port
nc -zv <hub-tailscale-ip> 50051   # remotesapi
```

## Client onboarding (per repo)

Run from the repo root after `bd init`:

```bash
export BEADS_HUB_HOST=<hub-tailscale-ip>
export BEADS_HUB_PASSWORD=<password-from-setup>

scripts/add-repo-to-hub.sh
```

The script:

1. Reads the local Dolt DB name from `.beads/metadata.json`.
2. Adds the hub as a Dolt remote (`bd dolt remote add hub …`).
3. Pushes local Dolt history to the hub — first push for the repo creates the
   database server-side.
4. Adds the hub as a federation peer with stored credentials.
5. Prints `bd federation status` so you can confirm the peer is reachable.

The local git pre-push and post-merge hooks (already installed by `bd hooks
install`) will keep things in sync as you commit and push code.

### Onboarding additional repos

Repeat the same script in each repo's root. The script is idempotent — it
detects existing Dolt remotes and federation peers and skips them.

## Continuous sync

The git hooks cover the common case (push-on-commit, pull-on-merge). For
unattended environments where you want sync without git activity, install a
systemd timer:

```ini
# /etc/systemd/system/beads-sync.service
[Unit]
Description=Periodic beads federation sync
After=network-online.target

[Service]
Type=oneshot
User=<your-user>
WorkingDirectory=/path/to/repo
ExecStart=/usr/local/bin/bd federation sync
```

```ini
# /etc/systemd/system/beads-sync.timer
[Unit]
Description=Run beads federation sync every 5 minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=5min
Unit=beads-sync.service

[Install]
WantedBy=timers.target
```

```bash
sudo systemctl enable --now beads-sync.timer
```

Repeat per repo (one timer + service per beads workspace), or wrap a
multi-repo loop in the `ExecStart` script.

## Operations

### Daily

Nothing. The hub is self-running.

### When `bd doctor` flags peer unreachable

```bash
ssh root@beads-lxc "systemctl status beads-dolt.service"
ssh root@beads-lxc "journalctl -u beads-dolt.service -n 50"
tailscale status | grep beads-lxc        # is the LXC online?
```

### Upgrading Dolt

Dolt is generally backward-compatible across minor versions, but pin the
version on the hub for predictability:

```bash
ssh root@beads-lxc "DOLT_VERSION=<new-version> bash /root/setup-beads-lxc.sh"
systemctl restart beads-dolt.service       # script does this for you
```

### Upgrading bd

Stay on the latest tagged release. On macOS:

```bash
brew upgrade beads
```

Linux clients: re-download the matching release from
[beads releases](https://github.com/steveyegge/beads/releases). Don't run
`HEAD` builds in production — they're a moving target.

### Backup

The Dolt data dir (`/var/lib/dolt`) is the source of truth on the hub. Two
sufficient backup strategies:

- **Snapshot the LXC** in Proxmox (cheap, fast, full-state).
- **Off-site mirror** to DoltHub or a second Dolt server:

  ```bash
  # On the hub:
  cd /var/lib/dolt/<dbname>
  dolt remote add backup dolthub.com/<account>/<repo>-beads
  dolt push backup main
  ```

### Rotating the password

```bash
ssh root@beads-lxc
sudo nano /etc/default/beads-dolt        # change BEADS_DOLT_PASSWORD
sudo systemctl restart beads-dolt.service

# Then on every client:
bd federation remove-peer hub
bd dolt remote remove hub
BEADS_HUB_PASSWORD=<new-password> scripts/add-repo-to-hub.sh
```

## Troubleshooting

### `Dolt server unreachable at 127.0.0.1:0`

Local Dolt server fell over, not a federation problem. See the
project's `CLAUDE.md` § "Native Beads Repair".

### `failed to get peer credentials: not found: federation peer <name>`

You have a Dolt remote with that name but no federation peer entry. Add the
peer:

```bash
bd federation add-peer hub <hub-host>:3306/<dbname> \
    --user beads --password <password>
```

### `database "<name>" not found on Dolt server`

The hub doesn't have a database with that name. Check the local
`.beads/metadata.json` `dolt_database` value, then `bd dolt push --remote hub`
to create it remotely.

### Conflicts during sync

```bash
bd federation sync --strategy theirs    # accept hub's version
bd federation sync --strategy ours      # keep local
```

For non-trivial conflicts, sync without a strategy and resolve manually
(beads will tell you which tables conflicted).

## Why a dedicated LXC

Considered alternatives:

| Option              | Why we didn't pick it                                        |
|---------------------|--------------------------------------------------------------|
| Co-host on ai-proxy | Couples unrelated services; Dolt OOM/crash hits inference.   |
| DoltHub             | Public repos free, private $$; less control, external dep.   |
| Local-only          | No multi-environment story.                                  |
| Self-host on laptop | Only available when laptop is awake and on the right VPN.    |

A dedicated LXC is small (1 GB RAM is plenty), has its own lifecycle, and
becomes the natural home for any future beads workspace.
