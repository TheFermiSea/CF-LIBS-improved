#!/bin/bash
#
# PreToolUse: Block Edit/Write on main/master branch
#
# Protects main branch from direct edits. Work must happen on
# feature branches or in .worktrees/bd-{BEAD_ID}/ directories.
#

INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // empty')

# Only check Edit and Write tools
[[ "$TOOL_NAME" != "Edit" ]] && [[ "$TOOL_NAME" != "Write" ]] && exit 0

# Get the file path being edited
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# Allow edits to hook/settings files (meta-configuration)
[[ "$FILE_PATH" == *"/.claude/"* ]] && exit 0

# Allow if editing within a worktree directory
[[ "$FILE_PATH" == *"/.worktrees/"* ]] && exit 0

# Check current branch
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null)

# Allow if not on main or master
[[ "$CURRENT_BRANCH" != "main" ]] && [[ "$CURRENT_BRANCH" != "master" ]] && exit 0

# Block: on main/master and not in a worktree
cat << EOF
{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"Cannot edit files on '$CURRENT_BRANCH'. Create a worktree or feature branch first:\n\n  git worktree add .worktrees/bd-{BEAD_ID} -b bd-{BEAD_ID} main\n\nOr with beads:\n\n  bd create \"Task title\" -d \"Description\"\n  git worktree add .worktrees/bd-{BEAD_ID} -b bd-{BEAD_ID} main"}}
EOF
exit 0
