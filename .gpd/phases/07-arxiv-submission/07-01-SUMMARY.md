---
phase: 07-arxiv-submission
plan: 01
status: checkpoint
plan_contract_ref: 07-01-PLAN.md
contract_results:
  claims:
    - id: claim-arxiv-ready
      status: partially-verified
      evidence: "Tarball builds; syntax validation passes (braces, environments, figures, citations, cross-refs all OK). pdflatex unavailable for full compilation test -- arXiv's own TeX Live will be the true test."
  deliverables:
    - id: deliv-tarball
      status: produced
      path: arxiv/cf-libs-gpu.tar.gz
    - id: deliv-metadata
      status: produced
      path: arxiv/metadata.txt
  acceptance_tests:
    - id: test-compiles-standalone
      outcome: partial-pass
      notes: "pdflatex not installed; comprehensive syntax validation substituted (balanced braces, matched environments, all figures present, all 17 citations resolved in .bbl, 20 cross-refs matched to 40 labels). Full pdflatex test deferred to arXiv submission."
    - id: test-tarball-contents
      outcome: pass
      notes: "Contains main.tex, main.bbl, 7 figure PDFs. No .aux, .log, .bib, .synctex, or other forbidden files. Size 512 KB (limit 10 MB)."
  forbidden_proxies:
    - id: fp-untested-tarball
      status: mitigated
      notes: "Cannot run pdflatex locally; substituted with syntax-level validation. Residual risk: arXiv TeX Live version differences."
---

# 07-01 SUMMARY: Package Paper for arXiv Submission

## One-liner
arXiv-ready tarball (512 KB) with flattened LaTeX, hand-built .bbl (17 refs), and 7 figure PDFs passes comprehensive syntax validation.

## Key Results

| Artifact | Path | Status |
|----------|------|--------|
| Flattened main.tex | `arxiv/main.tex` | All `\input{sections/...}` inlined; `\bibliography` replaced with `\input{main.bbl}` |
| Bibliography .bbl | `arxiv/main.bbl` | 17 references in elsarticle-harv format (hand-built; bibtex unavailable) |
| Figure PDFs (7) | `arxiv/figures/*.pdf` | Renamed to match .tex references |
| Tarball | `arxiv/cf-libs-gpu.tar.gz` | 512 KB, flat extraction structure |
| Metadata | `arxiv/metadata.txt` | physics.comp-ph primary, cs.CE secondary |

## Validation Results

| Check | Result |
|-------|--------|
| Braces balanced | PASS (0 depth errors) |
| Environments matched | PASS (10 types incl. figure*) |
| All 7 figures present | PASS |
| All 17 citations in .bbl | PASS |
| 20 cross-refs resolved | PASS (against 40 labels) |
| No residual \input | PASS (fully flattened) |
| No forbidden files | PASS |
| Tarball < 10 MB | PASS (512 KB) |
| pdflatex compilation | SKIPPED (not installed) |

## Deviations

1. **[Rule 4 - Missing component]** pdflatex/bibtex not available on this system. Mitigated by:
   - Hand-building .bbl in elsarticle-harv natbib format from refs.bib entries
   - Comprehensive syntax validation (braces, environments, figures, citations, cross-refs)
   - Renaming figure files to match .tex references (disk names differed from .tex paths)

2. **[Rule 4 - Missing component]** Figure filenames on disk (e.g., `fig2_voigt.pdf`) did not match .tex references (e.g., `fig02_voigt_throughput.pdf`). Copied figures with corrected names into `arxiv/figures/`.

## Paper Statistics

- Estimated: 23 pages, ~5838 words
- 7 figures (1 full-width pipeline + 6 column-width results)
- 17 references
- 13 numbered equations
- 1 table (accuracy summary)

## Checkpoints

| Task | Hash | Description |
|------|------|-------------|
| 1 | `4a70bc8` | Build arXiv tarball |
| 2 | `fe3863a` | Syntax validation |
| 3 | `eea5d4b` | Metadata file |

## Issues

- The .bbl was hand-built since bibtex was unavailable. If any natbib formatting issues arise during arXiv processing, regenerating the .bbl with `bibtex` on a system with TeX Live installed would be the fix.
- Affiliation in main.tex is still placeholder `[Affiliation]`. Should be filled before submission.
- Repository URL is placeholder `https://github.com/[repository]`. Should be filled before submission.
- Institution in acknowledgements is placeholder `[institution]`. Should be filled before submission.
