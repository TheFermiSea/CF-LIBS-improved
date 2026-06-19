#!/usr/bin/env bash
# asta_pdf_to_nlm.sh — OCR/extract one or more PDFs with asta, load the text into
# a NotebookLM notebook, and run a studio-only synthesis pipeline across them.
#
#   asta_pdf_to_nlm.sh <pipeline> <pdf> [pdf ...]
#     pipeline : studio-only nlm pipeline (default: asta-corpus-synthesis)
#
# Uses `asta pdf-extraction remote` (Asta OCR API). Prereqs: `asta auth login`,
# `nlm login`. Performs LIVE actions. Pipeline source_add can't take files, so we
# extract -> text -> `nlm source add --text`, then run a studio-only pipeline.
set -euo pipefail

PIPELINE="${1:?usage: asta_pdf_to_nlm.sh <pipeline> <pdf> [pdf ...]}"; shift
[ "$#" -ge 1 ] || { echo "give at least one PDF" >&2; exit 1; }

nb="$(nlm notebook create "asta-pdfs: $(date +%Y%m%d-%H%M%S)" 2>/dev/null \
  | grep -oE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' | head -1)"
[ -n "$nb" ] || { echo "could not parse notebook id" >&2; exit 1; }
echo ">> notebook: $nb" >&2

for pdf in "$@"; do
  [ -f "$pdf" ] || { echo "skip (missing): $pdf" >&2; continue; }
  echo ">> extracting: $pdf" >&2
  txt="$(asta pdf-extraction remote "$pdf")"   # text to stdout
  [ -n "$txt" ] || { echo "   no text extracted from $pdf" >&2; continue; }
  nlm source add "$nb" --text "$txt" --title "$(basename "$pdf")"
done

echo ">> nlm pipeline run $PIPELINE (notebook=$nb)" >&2
exec nlm pipeline run "$PIPELINE" -n "$nb"
