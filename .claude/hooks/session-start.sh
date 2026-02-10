#!/bin/bash
#
# SessionStart: Initialize beads context and show task status
#
# Based on beads startup-hooks best practices:
# - bd version change detection
# - Worktree cleanup for merged branches
# - Task status overview (in_progress → ready → blocked → stale)
#

# ============================================================
# Prerequisites
# ============================================================

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
BEADS_DIR="$PROJECT_DIR/.beads"

if [[ ! -d "$BEADS_DIR" ]]; then
  echo "No .beads directory found. Run 'bd init' to initialize."
  exit 0
fi

if ! command -v bd &>/dev/null; then
  echo "beads CLI (bd) not found. Install from: https://github.com/steveyegge/beads"
  exit 0
fi

# ============================================================
# bd Version Check (from beads startup-hooks pattern)
# ============================================================

METADATA_FILE="$BEADS_DIR/metadata.json"
CURRENT_VERSION=$(bd --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)

if [[ -n "$CURRENT_VERSION" ]] && command -v jq &>/dev/null; then
  # Initialize metadata if missing
  if [[ ! -f "$METADATA_FILE" ]]; then
    echo '{"database":"beads.db"}' > "$METADATA_FILE"
  fi

  LAST_VERSION=$(jq -r '.last_bd_version // "unknown"' "$METADATA_FILE" 2>/dev/null)

  if [[ "$CURRENT_VERSION" != "$LAST_VERSION" && "$LAST_VERSION" != "unknown" ]]; then
    echo "bd upgraded: $LAST_VERSION -> $CURRENT_VERSION"
    bd info --whats-new 2>/dev/null || true
    echo ""
  fi

  # Update stored version
  TEMP_FILE=$(mktemp)
  if jq --arg v "$CURRENT_VERSION" '.last_bd_version = $v' "$METADATA_FILE" > "$TEMP_FILE" 2>/dev/null; then
    mv "$TEMP_FILE" "$METADATA_FILE"
  else
    rm -f "$TEMP_FILE"
  fi
fi

# ============================================================
# Dirty Main Check
# ============================================================

REPO_ROOT=$(git -C "$PROJECT_DIR" rev-parse --show-toplevel 2>/dev/null)
if [[ -n "$REPO_ROOT" ]]; then
  DIRTY=$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null | head -5)
  if [[ -n "$DIRTY" ]]; then
    echo "WARNING: Main directory has uncommitted changes."
    echo "   Agents should only work in .worktrees/"
    echo ""
  fi
fi

# ============================================================
# Worktree Cleanup: Detect merged branches
# ============================================================

WORKTREES_DIR="$PROJECT_DIR/.worktrees"
if [[ -d "$WORKTREES_DIR" && -n "$REPO_ROOT" ]]; then
  while IFS= read -r worktree; do
    [[ -z "$worktree" ]] && continue
    BEAD_ID=$(basename "$worktree" | sed 's/bd-//')
    BRANCH=$(basename "$worktree")

    if git -C "$REPO_ROOT" branch --merged main 2>/dev/null | grep -q "$BRANCH"; then
      echo "bd-$BEAD_ID was merged - consider cleaning up"
      echo "   Run: git worktree remove \"$worktree\" && bd close \"$BEAD_ID\""
      echo ""
    fi
  done < <(git -C "$REPO_ROOT" worktree list --porcelain 2>/dev/null | grep "^worktree.*\.worktrees/bd-" | awk '{print $2}')
fi

# ============================================================
# Sync Status Check
# ============================================================

SYNC_STATUS=$(BEADS_NO_DAEMON=1 bd sync --status 2>&1 | grep -i "conflict\|behind\|ahead" | head -1)
if [[ -n "$SYNC_STATUS" ]]; then
  echo "bd sync: $SYNC_STATUS"
  echo ""
fi

# ============================================================
# Task Status
# ============================================================

echo ""
echo "## Task Status"
echo ""

# In-progress first (resume these)
IN_PROGRESS=$(BEADS_NO_DAEMON=1 bd list --status=in_progress 2>/dev/null | head -5)
if [[ -n "$IN_PROGRESS" ]]; then
  echo "### In Progress (resume these):"
  echo "$IN_PROGRESS"
  echo ""
fi

# Ready (unblocked)
READY=$(BEADS_NO_DAEMON=1 bd ready 2>/dev/null | head -5)
if [[ -n "$READY" ]]; then
  echo "### Ready (no blockers):"
  echo "$READY"
  echo ""
fi

# Blocked
BLOCKED=$(BEADS_NO_DAEMON=1 bd blocked 2>/dev/null | head -3)
if [[ -n "$BLOCKED" ]]; then
  echo "### Blocked:"
  echo "$BLOCKED"
  echo ""
fi

# Stale (no activity in 3 days)
STALE=$(BEADS_NO_DAEMON=1 bd stale --days 3 2>/dev/null | head -3)
if [[ -n "$STALE" ]]; then
  echo "### Stale (no activity in 3 days):"
  echo "$STALE"
  echo ""
fi

if [[ -z "$IN_PROGRESS" && -z "$READY" && -z "$BLOCKED" && -z "$STALE" ]]; then
  echo "No active beads. Create one with: bd create \"Task title\" -d \"Description\""
fi
