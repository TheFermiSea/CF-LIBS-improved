#!/usr/bin/env bash
# asta_paper_to_nlm.sh — search Semantic Scholar via asta, pick the top paper's
# URL, create a NotebookLM notebook, and run an nlm pipeline on that URL.
#
#   asta_paper_to_nlm.sh "<search query>" [pipeline] [limit]
#     pipeline : nlm pipeline to run (default: asta-paper-podcast)
#     limit    : how many search hits to consider; the 1st with a usable URL wins (default 5)
#
# Prereqs: `asta auth login` (Semantic Scholar) and `nlm login` (NotebookLM).
# This performs LIVE actions (creates a notebook, generates artifacts).
set -euo pipefail

QUERY="${1:?usage: asta_paper_to_nlm.sh \"<query>\" [pipeline=asta-paper-podcast] [limit=5]}"
PIPELINE="${2:-asta-paper-podcast}"
LIMIT="${3:-5}"

echo ">> asta papers search: $QUERY" >&2
json="$(asta papers search "$QUERY" --format json --limit "$LIMIT" \
  --fields title,year,url,externalIds,openAccessPdf)"

url="$(printf '%s' "$json" | python3 - <<'PY'
import json, sys
data = json.load(sys.stdin)
papers = data if isinstance(data, list) else (data.get("data") or data.get("papers") or [])
for p in papers:
    oa = (p.get("openAccessPdf") or {}).get("url")
    ext = p.get("externalIds") or {}
    url = oa or p.get("url")
    if not url and ext.get("ArXiv"):
        url = f"https://arxiv.org/abs/{ext['ArXiv']}"
    if not url and ext.get("DOI"):
        url = f"https://doi.org/{ext['DOI']}"
    if url:
        print(url)
        print(f">> picked: {p.get('title')} ({p.get('year')})", file=sys.stderr)
        break
else:
    sys.exit("no paper with a usable URL in the search results")
PY
)"

echo ">> creating notebook" >&2
nb="$(nlm notebook create "asta: ${QUERY}" 2>/dev/null \
  | grep -oE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' | head -1)"
[ -n "$nb" ] || { echo "could not parse notebook id from 'nlm notebook create' — create one manually and pass -n" >&2; exit 1; }

echo ">> nlm pipeline run $PIPELINE  (notebook=$nb, url=$url)" >&2
exec nlm pipeline run "$PIPELINE" -n "$nb" -u "$url"
