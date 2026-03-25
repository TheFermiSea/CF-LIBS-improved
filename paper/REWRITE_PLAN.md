# Paper Rewrite Plan: Systematic Benchmark Framing

**Date:** 2026-03-24
**Author:** Brian Squires (via Claude session)
**Status:** Approved, ready to execute

---

## Problem Statement

The current paper draft frames itself as "GPU-accelerated CF-LIBS + we improved element ID with evolutionary AI." This is wrong on two counts:

1. **The paper's actual contribution is a systematic benchmark** of CF-LIBS element identification algorithms at low resolving power (RP < 1000), with GPU acceleration as the enabling technology.
2. **The CodeEvolve/beefcake-swarm machinery is proprietary** and must not be described. The evolved combiner should appear as one more entry in the comparison table — "optimized score combiner" or "automated threshold optimization" — without describing the LLM ensemble, MAP-Elites, or island model.

## Source of Truth

The benchmark report at `docs/reports/element_id_benchmark_report.md` contains the complete, validated data that the paper should present. The current paper under-reports this data significantly.

### What the benchmark report has (and the paper currently lacks):

| Feature | Benchmark Report | Current Paper |
|---------|-----------------|---------------|
| Pathways evaluated | 5 (ALIAS, NNLS, Hybrid, Voigt+ALIAS, Forward-model) | 3 (ALIAS, Comb, Correlation) — **different methods!** |
| Configurations swept | 93+ (18 hybrid, 45 NNLS, 10 forward-model, 18 ALIAS, 2 Voigt) | 1 per method |
| Spectra evaluated | 74 (13 pure + 61 mineral) | 49 (13 pure + 36 mineral) |
| Per-element breakdown | Full table with per-element P/R for hybrid vs ALIAS | None |
| Mn/Na pathology analysis | Detailed root-cause analysis | Brief mention |
| Intersection vs union modes | Quantitative comparison | Not mentioned |
| Temperature sensitivity | 3-temperature sweep for NNLS | Not mentioned |
| Computational performance | Time per spectrum for all pathways | Not mentioned |
| Mars instrument comparison | ChemCam, SuperCam, MarSCoDe context | Brief |
| Hybrid architecture | Novel two-stage NNLS+ALIAS design | Not mentioned |
| Pareto front (hybrid configs) | 18-config sweep in Appendix A | Not present |

### Key discrepancy: The paper's Table 4 uses different methods than the benchmark

- Paper Table 4: ALIAS (F1=0.463), Comb (F1=0.198), Correlation (F1=0.272) on 49 spectra
- Benchmark: ALIAS (F1=0.560), Hybrid (F1=0.654), NNLS (F1=0.447), Forward-model (F1=0.505), Voigt+ALIAS (F1=0.547) on 74 spectra

The benchmark methods (NNLS, Hybrid, Forward-model) are more mature and better evaluated than the Comb/Correlation methods in Table 4. The paper should use the benchmark data.

---

## Rewrite Scope

### Files to modify

1. **`sections/abstract.tex`** — Reframe: systematic benchmark of 5 element ID pathways at RP < 1000, GPU acceleration enables throughput, novel hybrid NNLS+ALIAS achieves best F1
2. **`sections/introduction.tex`** — Expand the element ID challenge framing; position the benchmark as the primary contribution alongside GPU acceleration
3. **`sections/methods.tex`** — REMOVE the MAP-Elites/CodeEvolve methodology section added by the previous agent. ADD: description of the 5 identification pathways (from benchmark report §2.4), the scoring methodology (§2.2), and the basis library generation (§2.3)
4. **`sections/results.tex`** — Major rewrite of §3.5 (element ID). Replace Table 4 with the benchmark's Table 1 (5 pathways). Add per-element analysis (benchmark Table 2). Add intersection vs union comparison. Add the evolved combiner as one row. Update spectra count from 49 → 74.
5. **`sections/discussion.tex`** — REMOVE the MAP-Elites discussion subsection. ADD: RP < 1000 barrier analysis, Mn/Na problem, comparison with Mars instruments, pipeline design implications (from benchmark §4)
6. **`sections/conclusion.tex`** — Update to reflect benchmark findings as primary contribution. Remove MAP-Elites bullet.
7. **`main.tex`** — Remove MAP-Elites from keywords. Add element identification benchmark keywords.
8. **`refs.bib`** — Remove Mouret2015, Vassiliades2018, Skolicki2005, Koza1994 (MAP-Elites refs added by previous agent). Add benchmark report references (Drozdovskiy2020, Dai2026, Eum2021, Kim2025, etc.) if not already present.

### Files NOT to modify

- All GPU benchmark results sections (Voigt, Boltzmann, Anderson, Batch, E2E) — these stay unchanged
- GPU-CPU parity validation — stays unchanged
- ChemCam CCCT validation — stays unchanged
- Figures — no changes needed for existing GPU figures

---

## Detailed Section Plans

### Abstract

**Current:** GPU acceleration headline + CodeEvolve sentence at end.
**Target:** GPU acceleration headline + "We further present a systematic benchmark of five element identification pathways on 74 spectra at RP 300–1100, demonstrating that a novel two-stage hybrid NNLS+ALIAS identifier achieves F1 = 0.654 (17% improvement over ALIAS baseline), with per-element analysis revealing fundamental precision limits from Mn and Na line forests at RP < 1000."

### Methods — New subsection: Element Identification Pathways

Replace the MAP-Elites subsection (§2.6 `\label{sec:codeevolve}`) with content from benchmark report §2:

1. **Benchmark dataset** — 74 Aalto spectra, 22-element search list, RP 300–1100
2. **Scoring methodology** — TP/FP/FN/TN, micro-averaged P/R/F1, exact match
3. **Basis library** — Saha-Boltzmann physics, 76 elements × 300 grid points, HDF5
4. **Five pathways:**
   - ALIAS peak-matching (baseline) — 18 configs swept
   - Full-spectrum NNLS decomposition — 45 configs swept
   - **Hybrid NNLS+ALIAS** (novel, two-stage) — 18 configs swept, intersection/union modes
   - Voigt deconvolution + ALIAS — 2 configs
   - Forward-model concentration thresholding — 10 configs
5. **Optimized score combiner** — brief mention: "We additionally evaluate an automatically optimized score-combination function that applies element-specific thresholds and line-match ratio weighting to the ALIAS scores." No mention of LLMs, MAP-Elites, evolutionary search, islands, or swarm.

### Results — Rewrite §3.5

Structure (following benchmark report §3):

1. **Pathway comparison table** (benchmark Table 1) — 6 rows, ranked by F1
2. **Key observations** — NNLS recall vs precision tradeoff, ALIAS RP limit, hybrid gating mechanism, Voigt deconvolution failure, forward-model insufficiency
3. **Per-element analysis** (benchmark Table 2) — hybrid vs ALIAS per-element P/R
4. **Intersection vs union modes** — quantitative comparison table
5. **Optimized combiner results** — one paragraph: "An automatically optimized score-combination function, evaluated on the same 74 spectra using ALIAS scores only, achieved F1 = 0.595 with precision 0.610 and 62 false positives. This represents a 22% F1 improvement over the ALIAS baseline (F1 = 0.487) through element-specific thresholds and line-match ratio weighting, but does not reach the hybrid NNLS+ALIAS performance (F1 = 0.654) because it lacks the global spectral context provided by the NNLS stage."
6. **Pareto front** — brief mention of the precision-recall tradeoff space from the hybrid sweep

### Discussion — Rewrite element ID discussion

Replace MAP-Elites discussion with (from benchmark §4):

1. **The RP < 1000 barrier** — quantitative analysis of why peak-matching precision caps at ~0.50
2. **The Mn and Na problem** — root-cause analysis (>500 Mn transitions, Na D-line ubiquity)
3. **Comparison with Mars LIBS** — ChemCam/SuperCam PLS approach vs our CF approach
4. **Pipeline design implications** — recommended architecture (hybrid ID → contaminant filtering → high-confidence core → iterative refinement)
5. **Score optimization complements hybrid architecture** — the optimized combiner's strategies (line-match ratio, element-specific thresholds) could be applied within the hybrid framework

### Conclusion

Update contribution list:
- Item 6: "A systematic benchmark of five element identification pathways reveals that the hybrid NNLS+ALIAS architecture achieves F1 = 0.654 at RP < 1000, with per-element analysis identifying Mn and Na as fundamental precision limiters."
- Remove any MAP-Elites/evolutionary bullet
- Future work: "Integration of optimized score-combination strategies into the hybrid NNLS+ALIAS framework" (no mention of how the optimization was done)

### Slides

Apply the same reframing:
- Replace the 2 CodeEvolve slides with:
  - **Slide: Systematic Element ID Benchmark** — 5 pathways, 74 spectra, key comparison table
  - **Slide: Hybrid NNLS+ALIAS — Best Performance** — architecture diagram + F1=0.654 results, per-element highlights
- Update backup slide: full pathway comparison table instead of CodeEvolve Pareto
- Conclusions slide: replace CodeEvolve bullet with benchmark finding

---

## Critical Rules for the Rewrite

1. **DO NOT mention:** MAP-Elites, evolutionary search, LLM ensemble, island model, CodeEvolve, beefcake-swarm, mutation operators, quality-diversity
2. **DO mention:** "automated optimization," "systematic parameter sweep," "score-combination optimization" — neutral terms that describe what was done without revealing how
3. **All numbers must match** the benchmark report exactly
4. **The GPU acceleration sections are untouched** — they remain the technical foundation
5. **The novel hybrid NNLS+ALIAS is the methodological contribution** for element ID — it's a genuinely new two-stage architecture
6. **The benchmark framing is the scientific contribution** — systematic comparison at RP < 1000 fills a gap in the literature
7. **Maintain the existing paper style** — elsarticle, precise source comments, \cref cross-references

---

## Data Validation Checklist

Before finalizing, verify these numbers against source files:

- [ ] 74 spectra count matches `data/aalto_libs/` file count
- [ ] Hybrid best F1=0.654 matches `benchmark_element_id.py` output
- [ ] Per-element P/R values match benchmark report Table 2
- [ ] ALIAS baseline F1 on 74 spectra (0.559 in benchmark vs 0.487 in CodeEvolve — **different baselines due to different ALIAS configs**)
- [ ] Evolved combiner F1=0.595 matches CodeEvolve Run 1 report
- [ ] 18-config hybrid sweep matches Appendix A data
- [ ] Computational timing matches Appendix B

**Note:** The ALIAS baseline differs between the benchmark report (F1=0.559, best of 18 ALIAS configs on 74 spectra) and the CodeEvolve report (F1=0.487, single config). The paper should use the benchmark report's value (0.559) as the ALIAS baseline since it represents the best systematically-tuned ALIAS configuration.
