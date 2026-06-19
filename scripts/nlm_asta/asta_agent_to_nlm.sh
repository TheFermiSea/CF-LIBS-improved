#!/usr/bin/env bash
# asta_agent_to_nlm.sh — take the A2A result JSONs from an asta AGENT run
# (`asta analyze-data` / `asta generate-theories`), export them to Markdown with
# `asta artifacts`, load them into a NotebookLM notebook, and run a studio-only
# brief pipeline. This is the non-literature half of the asta<->nlm bridge.
#
#   asta_agent_to_nlm.sh <result_json_dir> [pipeline]
#     result_json_dir : dir of A2A result JSON files (asta agent output)
#     pipeline        : studio-only nlm pipeline (default: asta-findings-brief)
#
# Prereqs: `asta auth login`, `nlm login`. Performs LIVE actions.
set -euo pipefail

INDIR="${1:?usage: asta_agent_to_nlm.sh <result_json_dir> [pipeline=asta-findings-brief]}"
PIPELINE="${2:-asta-findings-brief}"
[ -d "$INDIR" ] || { echo "not a directory: $INDIR" >&2; exit 1; }

OUTDIR="$(mktemp -d)"
echo ">> asta artifacts: $INDIR -> $OUTDIR (markdown)" >&2
asta artifacts --input "$INDIR" --output "$OUTDIR" --format md

shopt -s nullglob
mds=("$OUTDIR"/*.md)
[ "${#mds[@]}" -ge 1 ] || { echo "asta artifacts produced no .md files" >&2; exit 1; }

nb="$(nlm notebook create "asta-findings: $(basename "$INDIR")" 2>/dev/null \
  | grep -oE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' | head -1)"
[ -n "$nb" ] || { echo "could not parse notebook id" >&2; exit 1; }
echo ">> notebook: $nb" >&2

for md in "${mds[@]}"; do
  echo ">> add source: $(basename "$md")" >&2
  nlm source add "$nb" --file "$md" --title "$(basename "$md")"
done

echo ">> nlm pipeline run $PIPELINE (notebook=$nb)" >&2
exec nlm pipeline run "$PIPELINE" -n "$nb"
