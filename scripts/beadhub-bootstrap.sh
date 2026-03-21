#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

quiet=0
force=0

usage() {
    cat <<'EOF'
Usage: ./scripts/beadhub-bootstrap.sh [--quiet] [--force]

Bootstraps BeadHub for the current worktree without writing to ~/.config/aw.

This script:
1. Calls the local BeadHub /v1/init endpoint directly.
2. Writes a per-worktree .beadhub file.
3. Writes per-worktree AW files at .aw/config.yaml and .aw/context.
4. Mirrors the AW config at .beadhub-cache/aw-config.yaml for debugging.

Environment overrides:
  BEADHUB_URL
  BEADHUB_PROJECT
  BEADHUB_ALIAS
  BEADHUB_HUMAN
  BEADHUB_ROLE
  BEADHUB_REPO_ORIGIN
EOF
}

log() {
    if [[ "$quiet" -eq 0 ]]; then
        printf '%s\n' "$*"
    fi
}

err() {
    printf 'beadhub-bootstrap: %s\n' "$*" >&2
}

derive_alias() {
    local workspace_path=$1
    local server_url=$2
    local project_slug=$3

    if [[ "$workspace_path" =~ /\.codex/worktrees/([^/]+)/ ]]; then
        printf 'codex-%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi

    python3 - "$server_url" "$project_slug" <<'PY'
import json
import sys
import urllib.error
import urllib.request

server_url, project_slug = sys.argv[1:3]
request = urllib.request.Request(
    f"{server_url.rstrip('/')}/v1/agents/suggest-alias-prefix",
    data=json.dumps({"project_slug": project_slug}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(request) as response:
        payload = json.loads(response.read().decode())
except urllib.error.HTTPError as exc:
    body = exc.read().decode()
    raise SystemExit(f"suggest-alias-prefix failed: HTTP {exc.code}: {body}") from exc

name_prefix = payload.get("name_prefix", "").strip()
if not name_prefix:
    raise SystemExit("suggest-alias-prefix returned no name_prefix")
print(name_prefix)
PY
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quiet)
            quiet=1
            ;;
        --force)
            force=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            err "unknown argument: $1"
            usage >&2
            exit 1
            ;;
    esac
    shift
done

repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || {
    err "must run inside a git checkout"
    exit 1
}

cd "$repo_root"

if [[ ! -d .beads ]]; then
    log "No .beads directory found; skipping BeadHub bootstrap."
    exit 0
fi

beadhub_file="$repo_root/.beadhub"
cache_dir="$repo_root/.beadhub-cache"
aw_dir="$repo_root/.aw"
aw_config="$aw_dir/config.yaml"
aw_context="$aw_dir/context"
cache_aw_config="$cache_dir/aw-config.yaml"

server_url=${BEADHUB_URL:-http://localhost:8000}
project_slug=${BEADHUB_PROJECT:-cf-libs}
role=${BEADHUB_ROLE:-developer}
human_name=${BEADHUB_HUMAN:-${USER:-agent}}
repo_origin=${BEADHUB_REPO_ORIGIN:-$(git remote get-url origin 2>/dev/null || true)}
workspace_path=$(pwd -P)
hostname_value=$(hostname 2>/dev/null || uname -n)
is_codex_worktree=0
if [[ "$workspace_path" =~ /\.codex/worktrees/([^/]+)/ ]]; then
    is_codex_worktree=1
fi

existing_alias=""
if [[ -f "$beadhub_file" && ! ( "$force" -eq 1 && "$is_codex_worktree" -eq 1 ) ]]; then
    alias_parse_stderr=$(mktemp)
    if ! existing_alias=$(awk -F': ' '/^alias: / {gsub(/^"|"$/, "", $2); print $2; exit}' "$beadhub_file" 2>"$alias_parse_stderr"); then
        err "failed to parse alias from $beadhub_file: $(cat "$alias_parse_stderr")"
        existing_alias=""
    elif [[ -z "$existing_alias" ]]; then
        log "No existing alias found in $beadhub_file; deriving a new alias."
    fi
    rm -f "$alias_parse_stderr"
fi

alias_value=${BEADHUB_ALIAS:-$existing_alias}
if [[ -z "$alias_value" ]]; then
    alias_value=$(derive_alias "$workspace_path" "$server_url" "$project_slug")
fi

if [[ "$force" -eq 0 && -f "$beadhub_file" && -f "$aw_config" && -f "$aw_context" && -f "$cache_aw_config" ]]; then
    log "BeadHub already bootstrapped for alias ${alias_value}."
    exit 0
fi

if [[ -z "$repo_origin" ]]; then
    err "could not determine git origin URL"
    exit 1
fi

payload=$(python3 - "$project_slug" "$alias_value" "$human_name" "$role" "$repo_origin" "$hostname_value" "$workspace_path" <<'PY'
import json
import sys

project_slug, alias, human_name, role, repo_origin, hostname_value, workspace_path = sys.argv[1:8]
print(json.dumps(
    {
        "project_slug": project_slug,
        "alias": alias,
        "human_name": human_name,
        "role": role,
        "repo_origin": repo_origin,
        "hostname": hostname_value,
        "workspace_path": workspace_path,
    }
))
PY
)

response=$(curl -fsS \
    -H "Content-Type: application/json" \
    -d "$payload" \
    "${server_url%/}/v1/init")

mkdir -p "$cache_dir" "$aw_dir"

RESPONSE_JSON="$response" \
SERVER_URL="$server_url" \
BEADHUB_FILE="$beadhub_file" \
AW_CONFIG_FILE="$aw_config" \
AW_CONTEXT_FILE="$aw_context" \
CACHE_AW_CONFIG_FILE="$cache_aw_config" \
HUMAN_NAME="$human_name" \
ROLE="$role" \
REPO_ORIGIN="$repo_origin" \
python3 - <<'PY'
import json
import os
from pathlib import Path
from urllib.parse import urlparse

response = json.loads(os.environ["RESPONSE_JSON"])
server_url = os.environ["SERVER_URL"].rstrip("/")
beadhub_file = Path(os.environ["BEADHUB_FILE"])
aw_config_file = Path(os.environ["AW_CONFIG_FILE"])
aw_context_file = Path(os.environ["AW_CONTEXT_FILE"])
cache_aw_config_file = Path(os.environ["CACHE_AW_CONFIG_FILE"])
human_name = os.environ["HUMAN_NAME"]
role = os.environ["ROLE"]
repo_origin = os.environ["REPO_ORIGIN"]

server_name = urlparse(server_url).netloc or server_url.replace("http://", "").replace("https://", "")
server_key = server_name.replace(":", "-")
account_name = f"acct-{server_key}__{response['project_slug']}__{response['alias']}"

def yaml_key(value):
    return json.dumps(str(value))


def yaml_value(value):
    if value is None:
        return '""'
    return json.dumps(str(value))

beadhub_lines = [
    "# Generated by: scripts/beadhub-bootstrap.sh",
    "# DO NOT COMMIT - add to .gitignore",
    "",
    f"workspace_id: {yaml_value(response.get('workspace_id') or response['agent_id'])}",
    f"beadhub_url: {yaml_value(server_url)}",
    f"project_slug: {yaml_value(response['project_slug'])}",
    f"repo_id: {yaml_value(response.get('repo_id'))}",
    f"repo_origin: {yaml_value(repo_origin)}",
    f"canonical_origin: {yaml_value(response.get('canonical_origin'))}",
    f"alias: {yaml_value(response['alias'])}",
    f"human_name: {yaml_value(human_name)}",
    f"role: {yaml_value(role)}",
    "",
]

aw_lines = [
    "servers:",
    f"  {yaml_key(server_key)}:",
    f"    url: {yaml_value(server_url)}",
    "accounts:",
    f"  {yaml_key(account_name)}:",
    f"    server: {yaml_value(server_key)}",
    f"    api_key: {yaml_value(response['api_key'])}",
    f"    default_project: {yaml_value(response['project_slug'])}",
    f"    agent_id: {yaml_value(response['agent_id'])}",
    f"    agent_alias: {yaml_value(response['alias'])}",
    f"    did: {yaml_value(response.get('did'))}",
    f"    custody: {yaml_value(response.get('custody'))}",
    f"    lifetime: {yaml_value(response.get('lifetime', 'ephemeral'))}",
    f"default_account: {yaml_value(account_name)}",
    "",
]

aw_context_lines = [
    f"default_account: {yaml_value(account_name)}",
    "server_accounts:",
    f"  {yaml_key(server_key)}: {yaml_value(account_name)}",
    "",
]

beadhub_file.write_text("\n".join(beadhub_lines))
aw_config_file.write_text("\n".join(aw_lines))
cache_aw_config_file.write_text("\n".join(aw_lines))
aw_context_file.write_text("\n".join(aw_context_lines))
PY

log "Bootstrapped BeadHub for alias ${alias_value}."
log "Use bash ./scripts/bdh for BeadHub commands in this repo."
