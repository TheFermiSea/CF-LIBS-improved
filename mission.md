# Mission — Close the CF-LIBS Identifier F1 Gap

## Headline question

**Which methodology and algorithm changes are needed to take the CF-LIBS multi-label element identification benchmark from hybrid_union F1 ≈ 0.75 (current, verified n=33 cross-shard) up to the project-target F1 ≥ 0.85 on the Aalto LIBS + BHVO-2 USGS + Vrabel 2020 composite test slice — and which of those changes account for the largest share of the gap?**

## Why this matters

- The 0.10 absolute F1 gap is a real obstacle to publishing CF-LIBS as a competitive identification pipeline. The latest peer-reviewed comparators (Mezoued ALIAS on SRM1412 reference glass: 15 of 20 elements detected = 75 % detection rate; *Scientific Data* Vrabel 2020 is a *classification* benchmark, not multi-label ID) suggest 0.75 is within the open-vocabulary CF-LIBS regime, but the project's internal validation target is F1 ≥ 0.85 (per `project_brain` notes).
- We have already separated 5 identifiers' cross-shard F1 cleanly (Appendix A of `docs/adr/ADR-0001-empirical-analysis-2026-05-13.md`). The remaining question is *which methodology-level levers close the rest of the gap.*
- Several plausible levers are independent and measurable in isolation; the right experimental design is a knob-by-knob ablation, not an end-to-end refactor.

## Operational definitions

- **F1** — micro-F1 over (spectrum, element) pairs treating element sets as multi-label predictions vs. multi-label truth. Reported by `id_summary.json["overall"][<workflow>]["micro_f1"]`.
- **Headline ranking** — sorted by `F1_mean` across 3 disjoint Vrabel shards (1/3, 2/3, 3/3) of the standard composite test slice (Aalto LIBS + BHVO-2 USGS + Vrabel 2020).
- **Identifier vocabulary** — the candidate element set the identifier scores against. *Open vocabulary* = full 83-element periodic-table-restricted registry. *Restricted vocabulary* = the union of element sets actually appearing in the dataset's truth labels.
- **Shot averaging** — replacing per-shot spectra with the mean of K shots from the same sample before running identification. K is `--vrabel-max-shots`.
- **Apples-to-apples comparator** — a peer-reviewed LIBS multi-label element identification result whose data slice, vocabulary, shot-averaging, and cross-validation methodology match ours within the bands tolerated by the ablation knobs below.

## In scope

- Path A multi-knob ablation: measure F1 delta of each of {`--quick` removed, restricted vocabulary, K-shot averaging, full LOOCV folds, expanded basis-FWHM grid (Exp 3)} independently and in combination.
- Comparison of hybrid_union against its components (alias, spectral_nnls) under each ablated condition.
- Per-element drag analysis (which elements pull F1 down for each identifier, with cross-shard variance).
- Literature comparator audit (open-vocabulary vs restricted-vocabulary published numbers; matched-methodology comparators only).

## Out of scope (for this mission)

- New identifier algorithms (e.g., LLM-driven evolutionary search, CNN-on-raw-spectrum). Those belong to the GPD-Research-Publication-Pipeline epic (`CF-LIBS-improved-mm2`).
- Composition workflow shootout (Exp 2 in the existing autonomous-plan). That's a different scope question and should be its own mission once F1 is stabilized.
- Hardware / instrument-design changes (LIBS resolving power, gating, plasma temperature). Those are upstream of any CF-LIBS algorithm.
- JAX-port performance optimization (ADR-0001 patterns T1-1 / T2-3 / T2-7). Those affect wall-time, not F1 mean.

## Success criteria

1. **Quantified ablation table**: a single Markdown table with one row per Path-A knob, reporting (a) F1 delta vs current baseline, (b) cross-shard std of that delta, (c) cost in wall-time per iter, (d) cost in vocabulary opacity.
2. **Ranked recommendation**: which knobs, in which order, the project should adopt to maximize F1 gain per implementation hour.
3. **Matched-methodology literature comparator** showing that, under the same ablation settings, the project's F1 either equals or beats the closest peer-reviewed CF-LIBS-class identifier.
4. **No fabricated numbers** — every reported F1 in the synthesis comes from a real `id_summary.json` on disk under `/cluster/shared/cf-libs-bench/results/exp00*-*/`. Every literature comparator has an Asta-find citation.
5. **All raw data preserved** — id_records.csv, id_summary.json, manifest.jsonl, and per-iter parquet outputs survive in NFS (`/cluster/shared/cf-libs-bench/results/`) at experiment end.

## Constraints

- Each ablation experiment must complete within 1.5 hours wall-time across the 3 cluster shards (vasp-01/02/03). This rules out re-running with `--quick` removed if a single iter takes >30 min; in that case the knob has to be tested via a sub-sampled dataset.
- The `parameter_sweep.py` wrapper now has the round-robin fix (PR #150); use it directly via `asta experiment` rather than bash loops.
- No NotebookLM auto-update of `project_brain` until at least 2 ablation iterations land — avoid spamming the 300-source limit with intermediate runs.

## Discovered constraints (recorded during execution)

(empty — populate as the research-step `execute` workflow finds them)

## Background

- Current verified baseline (commit `9a5558a`): hybrid_union F1 = 0.7502 ± 0.0335 across n=33 spectra split into 3 disjoint Vrabel shards. Documented in `docs/adr/ADR-0001-empirical-analysis-2026-05-13.md` §Appendix A.
- The parameter_sweep wrapper bug (`CF-LIBS-improved-yfbg`, closed by PR #150) silently routed all iters to cell 0; the round-robin fix is on `dev`.
- Path A multi-knob study was conceptually planned in `docs/exp001-autonomous-plan.md`; this mission promotes it to a typed research DAG.
- The Asta-driven research loop (`docs/workflows/asta-research-loop.md`) is the operating policy: lean on Asta skills (`experiment`, `analyze-data`, `generate-theories`, `literature-find`, `literature-report`) before homegrown alternatives.
