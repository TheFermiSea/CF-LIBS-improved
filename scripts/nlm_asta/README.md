# asta × NotebookLM pipelines

A few **`nlm pipeline`** definitions designed to be fed by **asta** tools — not just
literature search, but asta's agent tools too (DataVoyager, Theorizer) and its
PDF/document utilities. The `nlm` pipeline engine runs a fixed sequence of
NotebookLM steps; asta produces the *input* (a URL, extracted text, or an
exported Markdown artifact) that the pipeline ingests and turns into an audio
overview / report / flashcards / quiz.

## The integration seam (important)

`nlm pipeline` steps are NotebookLM-native. Their `source_add` step accepts only
**`type: url`** (bound to the runtime `$INPUT_URL`) or **`type: text`** — *not*
files. So there are two ways asta hands off to a pipeline:

| asta output | handoff | pipeline shape |
|---|---|---|
| a paper **URL** (`asta papers …`) | `nlm pipeline run … -u <url>` → the `source_add(url)` step | URL-driven (`asta-paper-*`) |
| **text** (`asta pdf-extraction`, `asta documents fetch`) | pre-load with `nlm source add --text`, then run a **studio-only** pipeline | `asta-corpus-synthesis` |
| **files** — agent results via `asta artifacts --format md` (`asta analyze-data`, `asta generate-theories`) | pre-load with `nlm source add --file`, then run a **studio-only** pipeline | `asta-findings-brief` |

Studio-only pipelines have **no `source_add` step**, so they operate on whatever
the notebook already holds — that's how file/multi-source inputs get processed.

## The pipelines

| pipeline | input | produces | pair with |
|---|---|---|---|
| `asta-paper-podcast` | paper URL (`$INPUT_URL`) | summary query → **audio deep-dive** + **briefing report** | `asta papers search/get` |
| `asta-paper-study-pack` | paper URL (`$INPUT_URL`) | **briefing report** + **flashcards** + **quiz** | `asta papers search/citations` |
| `asta-corpus-synthesis` | *pre-loaded* sources | cross-source synthesis → **report** + **audio** | `asta pdf-extraction`, `asta documents fetch`, batched `asta papers` |
| `asta-findings-brief` | *pre-loaded* agent artifacts | findings/decision **report** + **quiz** | `asta analyze-data` (DataVoyager), `asta generate-theories` (Theorizer) → `asta artifacts --format md` |

## Install (register the pipelines into nlm)

```bash
for y in scripts/nlm_asta/pipelines/*.yaml; do
  nlm pipeline create "$(basename "$y" .yaml)" -f "$y"
done
nlm pipeline list      # the four should appear with source=user
```

## Use

Prereqs: `asta auth login` and `nlm login` once.

```bash
# 1. Literature → podcast (one command, via the wrapper)
scripts/nlm_asta/asta_paper_to_nlm.sh "calibration-free LIBS self-absorption" asta-paper-podcast

# 2. PDFs → cross-source synthesis
scripts/nlm_asta/asta_pdf_to_nlm.sh asta-corpus-synthesis paper1.pdf paper2.pdf

# 3. Agent results (DataVoyager / Theorizer) → findings brief
#    (after an `asta analyze-data submit ...` / `asta generate-theories ...` run
#     whose A2A result JSONs are in ./agent_out/)
scripts/nlm_asta/asta_agent_to_nlm.sh ./agent_out asta-findings-brief
```

### Manual (no wrapper)

```bash
# paper URL → pipeline
url=$(asta papers get <PAPER_ID> --format json --fields openAccessPdf,url | python3 -c 'import json,sys;p=json.load(sys.stdin);print((p.get("openAccessPdf") or {}).get("url") or p["url"])')
nb=$(nlm notebook create "my paper" | grep -oE '[0-9a-f-]{36}')
nlm pipeline run asta-paper-study-pack -n "$nb" -u "$url"

# text → studio-only pipeline
nlm source add "$nb" --text "$(asta pdf-extraction remote paper.pdf)" --title paper.pdf
nlm pipeline run asta-corpus-synthesis -n "$nb"
```

## Notes
- `nlm pipeline create` only writes the YAML into nlm's local pipeline store — no
  network. **Running** a pipeline performs live NotebookLM actions (creates
  artifacts, consumes quota).
- Valid `studio_create` `artifact_type`: `audio`, `video`, `report`,
  `flashcards`, `quiz`. `report_format` defaults to `Briefing Doc`;
  `audio_format` to `deep_dive`.
- These wrappers were authored but not executed here (they require your asta /
  NotebookLM auth); verify the `nlm notebook create` id parsing against your nlm
  version on first run.
