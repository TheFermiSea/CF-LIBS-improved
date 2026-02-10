#!/bin/bash
#
# UserPromptSubmit: Nudge clarification for very short/ambiguous requests
#
# Only triggers on extremely short prompts (<30 chars) that are likely
# too vague to act on. Does NOT block — just adds a reminder.
#

INPUT=$(cat)
PROMPT=$(echo "$INPUT" | jq -r '.prompt // empty')
LENGTH=${#PROMPT}

# Only nudge for very short prompts that are likely ambiguous
if [[ $LENGTH -lt 30 ]]; then
  # Skip if it looks like a slash command, "yes"/"no", or continuation
  if [[ "$PROMPT" =~ ^/ ]] || [[ "$PROMPT" =~ ^(yes|no|y|n|ok|continue|proceed|done|stop)$ ]]; then
    exit 0
  fi

  cat << 'EOF'
<system-reminder>
Short request detected. Consider clarifying scope before proceeding:
- What specific outcome is expected?
- Which files or components are involved?
Use AskUserQuestion if the intent is unclear.
</system-reminder>
EOF
fi

exit 0
