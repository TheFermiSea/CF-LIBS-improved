#!/bin/bash
#
# PreToolUse:Bash - Validate all epic children are complete before closing
#
# Prevents closing an epic when children are still open.
# Reads hook input from stdin (standard PreToolUse format).
#

INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // empty')

# Only check Bash tool
[[ "$TOOL_NAME" != "Bash" ]] && exit 0

COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# Only check bd close commands
echo "$COMMAND" | grep -qE 'bd\s+close' || exit 0

# Extract the ID being closed (first ID after "bd close")
CLOSE_ID=$(echo "$COMMAND" | sed -E 's/.*bd\s+close\s+([A-Za-z0-9._-]+).*/\1/')
[[ -z "$CLOSE_ID" ]] && exit 0

# Check if this is an epic by looking for children
CHILDREN=$(BEADS_NO_DAEMON=1 bd show "$CLOSE_ID" --json 2>/dev/null | jq -r '.[0].children // empty' 2>/dev/null || echo "")

if [[ -z "$CHILDREN" || "$CHILDREN" == "null" ]]; then
  # Not an epic or no children — allow close
  exit 0
fi

# This is an epic — check if all children are complete
INCOMPLETE=$(BEADS_NO_DAEMON=1 bd list --json 2>/dev/null | jq -r --arg epic "$CLOSE_ID" '
  [.[] | select(.parent == $epic and .status != "done" and .status != "closed")] | length
' 2>/dev/null || echo "0")

if [[ "$INCOMPLETE" != "0" && -n "$INCOMPLETE" ]]; then
  INCOMPLETE_LIST=$(BEADS_NO_DAEMON=1 bd list --json 2>/dev/null | jq -r --arg epic "$CLOSE_ID" '
    [.[] | select(.parent == $epic and .status != "done" and .status != "closed")] | .[] | "\(.id) (\(.status))"
  ' 2>/dev/null | tr '\n' ', ' | sed 's/,$//')

  cat << EOF
{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"Cannot close epic '$CLOSE_ID' — has $INCOMPLETE incomplete children: $INCOMPLETE_LIST\n\nClose or complete all children first, then close the epic."}}
EOF
  exit 0
fi

exit 0
