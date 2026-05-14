# CF-LIBS Research Findings

A structured database of investigation outcomes — every hypothesis we tried,
what we learned, and what to do next. The goal of this directory is to make
**forward progress** across sessions instead of re-deriving the same
conclusions.

If you are reading this with no context: start with the
["Top open questions / what to work on next"](#top-open-questions--what-to-work-on-next)
section below. Each open question links to the supporting findings.

---

## TL;DR — the most consequential findings (2026-05-14)

| ID | Status | Surprise | Headline | What it implies |
|----|--------|---------:|----------|-----------------|
| [empirical-05](2026-05-14-empirical-05-silicon-is-the-headline-miss-789-of-si-containing-spectra-miss-si-under-all.md) | confirmed | n/a | **Si is missed in 15/19 spectra by ALL 5 identifiers** | Single largest recall residual. Forensic on `vrabel_s039` (the 1 Vrabel spectrum that catches Si) is the highest-leverage next step. |
| [empirical-01](2026-05-14-empirical-01-universal-misses-53-pairs-53174-true-spectrum-element-pairs-universally-missed-by.md) | confirmed | n/a | **53/174 true (spectrum, element) pairs missed by every identifier** | Algorithmic combination of current 5 cannot reach literature ceiling on this corpus. |
| [asta-28](2026-05-14-asta-28-the-016-macro-f1-gap-to-the-literature-ceiling.md) | confirmed | +0.28 | **Super-union of all 5 identifiers caps at recall 0.667 (literature target 0.85)** | The 0.16 macro-F1 gap is a hard physical/basis-line ceiling, not an algorithm-combining problem. |
| [empirical-02](2026-05-14-empirical-02-top-missed-elements-top-universally-missed-elements-si-15-mg-11-al.md) | confirmed | n/a | **Universally-missed elements: Si:15, Mg:11, Al:9, Ti:7** | Misses are **major rock-forming** elements, not trace/REE. Kills "fix the long tail" strategy. |
| [empirical-03](2026-05-14-empirical-03-vrabel-rp-confound-4953-universal-misses-are-on-vrabel2020_soil_benchmark-rp30000.md) | confirmed | n/a | **49/53 universal misses are on Vrabel soil (rp=30000)** | The 'universal miss' signal is almost entirely rp=30000. |
| [asta-24](2026-05-14-asta-24-the-extremely-low-f1-score-of-the-comb.md) | confirmed | +0.32 | **comb F1=0.03 is FN-dominated (mean FN=7.06, FP=0.36)** | Threshold is globally too strict. Loosening comb's thresholds is the least controversial fix on the board. |
| [asta-08](2026-05-14-asta-08-the-macro-f1-gap-from-spectral_nnls-044-to-hybrid_union.md) | weakly_confirmed | +0.04 | **hybrid_union cuts FPs 5.72→1.48 AND raises recall 0.55→0.65** | Consensus voting is paying off on both axes. Strongest argument for keeping `hybrid_union` as default. |
| [asta-20](2026-05-14-asta-20-the-performance-of-pure-line-matching-algorithms-comb-and.md) | refuted | -0.45 | **comb performance is NEGATIVELY correlated with rp (Spearman -0.59, p=0.0003)** | Finer rp overloads peak-matching. Counter-intuitive; suggests rp-aware threshold coupling. |
| [asta-10](2026-05-14-asta-10-the-current-macro_f1-ceiling-of-069-for-hybrid_union.md) | refuted | -0.59 | **hybrid_union precision (0.75) ≈ recall (0.70), no significant gap** | The "just push recall" strategy is dead. Both metrics must improve in lockstep. |
| [asta-06](2026-05-14-asta-06-the-alias-algorithms-strict-precision-constraint-precision10-results.md) | confirmed | +0.28 | **alias recall 10.5% on matrix vs 1.4% on trace metals (chi² p=0.029)** | alias high_recall mode (PR #159) is the right direction; question is whether FP rejection (#166) keeps precision high enough. |

---

## Top open questions / what to work on next

These are the prioritized next steps as of 2026-05-14, derived from the
findings above. **Pick the highest-numbered one you can make progress on
and stop the inner-loop swarm if it's grinding low-value work.**

### 1. Forensic: why does `vrabel_s039` catch Si when 14 other Vrabel spectra don't?

Si is present in 19 spectra. Only 4 catch Si under any identifier (3 BHVO-2,
1 Vrabel — `s039`). The 14 Vrabel spectra that miss Si are at the *same* rp
and similar matrix as `s039`. If we can identify what's different about
`s039` — peak SNR, concentration, neighbor-line crowding, baseline behavior
— we may unlock the largest single block of recall in the corpus (15 pairs,
~9% of all true pairs).

**Concrete next step:** load the 4 Si-catching spectra and the 15 missing
ones, plot Si I 288.16nm and Si I 251.61nm regions, diff annotations. Should
take <1 hour. See [empirical-05](2026-05-14-empirical-05-silicon-is-the-headline-miss-789-of-si-containing-spectra-miss-si-under-all.md).

### 2. Decide whether the comb threshold loosening is on the roadmap

[asta-24](2026-05-14-asta-24-the-extremely-low-f1-score-of-the-comb.md) is
the strongest single confirmed signal (+0.32 surprise). Mean FN per
spectrum is 7.06 vs FP 0.36 — comb is leaving an enormous amount of recall
on the table.

But PR #166 went the *opposite* direction (tighter FP rejection for tier-2
Mn/Na/K). Before doing anything: confirm that #166 was scoped to the
*ensemble*'s tier-2 stage and not to comb's own threshold floor. If PR #166
did NOT touch comb's `threshold_percentile=85` / `min_correlation=0.7`, then
loosening those is independent and orthogonal — schedule it.

**Concrete next step:** scan the codebase for where comb's
`threshold_percentile` is set; if it's a constant, propose an opt-in
`high_recall` flag mirroring [PR #159](https://github.com/.../pull/159)'s
pattern for alias. See [asta-24](2026-05-14-asta-24-the-extremely-low-f1-score-of-the-comb.md)
and [asta-19](2026-05-14-asta-19-the-combinatorial-line-matching-algorithm-comb-experiences-combinatorial-candidate.md).

### 3. Run the next Asta autodiscovery iteration with a corpus that has rp variance

Three of the 30 experiments (asta-04, asta-11, asta-22, asta-29) were
*untestable* because the n=33 corpus has no rp variance worth speaking of
(essentially two values: 9433 BHVO-2, 30000 Vrabel). Several more (asta-15,
asta-17) were untestable because `aa1100_substrate` isn't in the corpus.

Before running another autodiscovery, either:
- (a) curate a benchmark with at least 5 distinct rp buckets, OR
- (b) explicitly scope hypotheses to be testable on the n=33 corpus.

Without one of these, every "rp dependency" or "aa1100 self-absorption"
hypothesis Asta proposes will get a null result. See
[asta-22](2026-05-14-asta-22-spectra-with-lower-estimated-resolving-power-rp_estimate-cause.md).

---

## Directory layout

```
docs/research/findings/
├── README.md                                  ← you are here
├── 2026-05-14-asta-01-…md                     ← per-finding markdown (one file each)
├── 2026-05-14-asta-02-…md
├── …
├── 2026-05-14-asta-30-…md
├── 2026-05-14-empirical-01-…md                ← empirical (non-Asta) findings
├── 2026-05-14-empirical-02-…md
└── …
docs/research/findings.jsonl                   ← machine-queryable index
                                                  (one finding per line)
```

Markdown is for humans. The JSONL is for querying:

```bash
# Find all "refuted" hypotheses
jq -c 'select(.status == "refuted") | {id, hypothesis: .hypothesis[:80]}' \
  docs/research/findings.jsonl

# Find findings cross-linked to PR #166
jq -c 'select(.relates_to_prs | tostring | contains("166"))' \
  docs/research/findings.jsonl

# Find everything tagged "silicon"
jq -c 'select(.tags | contains(["silicon"]))' \
  docs/research/findings.jsonl

# Sort Asta findings by surprise magnitude (most informative first)
jq -s 'sort_by(.surprise // 0 | fabs) | reverse | .[:5]' \
  docs/research/findings.jsonl
```

---

## Status legend

| Status | Surprise range | Meaning |
|--------|----------------|---------|
| `confirmed` | `+0.25` and above | Strong posterior shift toward TRUE. Treat as a working assumption; future experiments should be built ON this. |
| `weakly_confirmed` | `+0.10` to `+0.25` | Mild posterior shift. Interesting but not decisive. |
| `neutral` | `-0.10` to `+0.10` | No update. Often means the experiment was untestable on the available data. |
| `weakly_refuted` | `-0.10` to `-0.40` (with caveats) | Mild posterior shift toward FALSE. Worth investigating an alternative formulation. |
| `refuted` | `-0.40` and below | Strong posterior shift toward FALSE. Stop pursuing this exact hypothesis; reformulate. |

Asta's own `is_surprising` flag is a separate signal — it tracks whether
the posterior was *outside* the prior's likely range, irrespective of
direction. We record it in the frontmatter but use surprise magnitude for
status because it's directionally informative.

---

## How to add a new finding

### When to add one

- Any time an Asta autodiscovery run completes — document EVERY experiment,
  even neutral ones, because "neutral" is itself a finding (often: corpus
  doesn't have the data to test it).
- Any time a manual empirical analysis (e.g. a bd issue's investigation
  produces a quantitative claim) yields a result that future work would
  benefit from knowing.
- Any time a PR closes that *aligns with* an existing finding — update that
  finding's `relates_to_prs` list.

### How to add one

1. Create a new file with the naming convention:
   ```
   YYYY-MM-DD-{asta|empirical|investigation}-NN-short-slug.md
   ```
   where `NN` is a 2-digit ordinal within that day/source.

2. Use the YAML frontmatter schema (see existing files for examples):
   ```yaml
   ---
   id: 2026-05-14-asta-31
   date: 2026-05-14
   source: asta-autodiscovery-<run-id>     # OR empirical-analysis-<date>
   experiment_node: node_31_0              # Asta only
   status: refuted | weakly_refuted | neutral | weakly_confirmed | confirmed
   surprise: -0.69                         # Asta only
   prior: 0.71                             # Asta only
   posterior: 0.27                         # Asta only
   is_surprising: true                     # Asta only
   tags: [tag1, tag2]
   relates_to_prs: [#166, #168]
   ---
   ```

3. Body sections (in order):
   - `## Hypothesis` (or `## Claim` for empirical)
   - `## Asta result` (or `## Source` for empirical)
   - `## Asta analysis` (verbatim from Asta) — Asta only
   - `## What this means for us` (the **reflection** — what to actually do)
   - `## Cross-references` — PRs, related findings, bd issues
   - `## Pull full code/output from Asta` (with the `asta autodiscovery
     experiment <run-id> node_<i>_0` command) — Asta only

4. Append a one-line JSON record to `docs/research/findings.jsonl` with at
   minimum: `id`, `date`, `source`, `status`, `tags`, `relates_to_prs`,
   `file`. Asta records should also include `surprise`, `prior`, `posterior`,
   `is_surprising`, `hypothesis`. Empirical records should include `title`,
   `claim`, `what_to_do_next`.

5. If the new finding makes the TL;DR table at the top of this README more
   useful, update the table. Keep the table to ~10 rows.

### How to update an existing finding

- **Re-running a refuted hypothesis with different framing:** create a NEW
  finding, link to the original in `relates_to`. Don't edit history.
- **New PR closes a related issue:** add the PR to the existing finding's
  `relates_to_prs` list, both in the frontmatter and in the cross-references
  body section, AND in `findings.jsonl`.

---

## Catalog (auto-pinned high-value findings)

### Confirmed (these are working assumptions now)

- [asta-06](2026-05-14-asta-06-the-alias-algorithms-strict-precision-constraint-precision10-results.md): alias has a 7.8× recall gap between matrix and trace metals
- [asta-07](2026-05-14-asta-07-the-f1-performance-gap-between-hybrid_union-and-spectral_nnls.md): hybrid_union recovers alkali/alkaline-earth FNs from spectral_nnls
- [asta-09](2026-05-14-asta-09-the-low-recall-ceiling-of-the-precision-dominant-alias.md): alias FN rate 97.9% on Na/K/Ca/Mg vs 88.5% on others
- [asta-24](2026-05-14-asta-24-the-extremely-low-f1-score-of-the-comb.md): comb F1=0.03 is FN-bound; threshold globally too strict
- [asta-28](2026-05-14-asta-28-the-016-macro-f1-gap-to-the-literature-ceiling.md): super-union recall ceiling is 0.667, not 0.85
- [empirical-01..06](.): the n=33 universal-miss pattern (53 pairs, Si-heavy, Vrabel-dominated)

### Refuted (stop pursuing these specific framings)

- [asta-01](2026-05-14-asta-01-the-spectral_nnls-identifier-exhibits-significantly-lower-f1-scores.md): "spectral_nnls degrades on Vrabel soils" — actually performs better
- [asta-10](2026-05-14-asta-10-the-current-macro_f1-ceiling-of-069-for-hybrid_union.md): "hybrid_union is recall-bound" — precision ≈ recall, no gap
- [asta-13](2026-05-14-asta-13-the-correlation-identifier-sacrifices-the-perfect-precision-of.md): "correlation trades precision for recall" — both algorithms are near noise floor
- [asta-18](2026-05-14-asta-18-the-hybrid_union-algorithm-suffers-a-significant-degradation-in.md): "hybrid_union degrades on Vrabel" — actually improves
- [asta-19](2026-05-14-asta-19-the-combinatorial-line-matching-algorithm-comb-experiences-combinatorial-candidate.md): "BHVO-2 is uniquely hard for comb" — comb fails everywhere
- [asta-21](2026-05-14-asta-21-the-hybrid_union-algorithm-is-robust-to-matrix-complexity.md): "spectral_nnls is matrix-sensitive" — no identifier×dataset interaction
- [asta-25](2026-05-14-asta-25-the-physical-presence-of-iron-fe-in-a.md): "Fe-dense spectra confuse spectral_nnls" — actually helps it
- [asta-26](2026-05-14-asta-26-the-high-failure-rate-of-aa1100_substrate-identifications-under.md): "alias failures correlate with low rp" — opposite direction
- [asta-27](2026-05-14-asta-27-spectral_nnls-exhibits-a-severe-lack-of-cross-dataset-stability.md): "spectral_nnls is unstable on complex matrices" — worst on simplest dataset
- [asta-30](2026-05-14-asta-30-alkali-and-alkaline-earth-metals-eg-na-k-li.md): "hybrid_union misses alkalis" — it actually recovers them well

### Neutral / untestable on n=33 (need different corpus)

- [asta-04](2026-05-14-asta-04-identification-failures-in-the-alias-algorithm-such-as.md), [asta-11](2026-05-14-asta-11-identification-failures-via-physical-mechanisms-spectra-that-trigger.md), [asta-15](2026-05-14-asta-15-the-high-failure-rate-of-alias-on-the.md), [asta-17](2026-05-14-asta-17-the-strict-precision-of-alias-leads-to-aa1100_substrate.md), [asta-22](2026-05-14-asta-22-spectra-with-lower-estimated-resolving-power-rp_estimate-cause.md), [asta-29](2026-05-14-asta-29-tunable-threshold-floors-vs-physical-limits-spectrum-level-identification.md): all need a corpus with rp variance or aa1100_substrate spectra to test.
- [asta-03](2026-05-14-asta-03-the-correlation-identifiers-superior-f1-score-over-the.md): needs `basis_fwhm_mismatch_nm` logged into annotations.

---

## Cross-reference: PRs closed this session

| PR | Title (short) | Related findings |
|----|---------------|------------------|
| #151, #153, #154 | wire `get_wavelength_tolerance` into alias / comb / correlation | [asta-03](2026-05-14-asta-03-the-correlation-identifiers-superior-f1-score-over-the.md) (FWHM annotations) |
| #159 | opt-in `alias` high-recall mode | [asta-06](2026-05-14-asta-06-the-alias-algorithms-strict-precision-constraint-precision10-results.md), [asta-09](2026-05-14-asta-09-the-low-recall-ceiling-of-the-precision-dominant-alias.md) |
| #164 | opt-in hybrid 2-of-3 majority consensus | [asta-07](2026-05-14-asta-07-the-f1-performance-gap-between-hybrid_union-and-spectral_nnls.md), [asta-08](2026-05-14-asta-08-the-macro-f1-gap-from-spectral_nnls-044-to-hybrid_union.md), [asta-12](2026-05-14-asta-12-certain-trace-and-alkali-elements-eg-na-k.md), [asta-14](2026-05-14-asta-14-constructing-a-strict-pure-line-matching-ensemble-by-taking-the.md), [empirical-06](2026-05-14-empirical-06-zero-caught-by-all-5-zero-spectrum-element-pairs-caught-by-all-5.md) |
| #166 | tighter Tier-2 FP rejection for Mn/Na/K | [asta-06](2026-05-14-asta-06-the-alias-algorithms-strict-precision-constraint-precision10-results.md), [asta-12](2026-05-14-asta-12-certain-trace-and-alkali-elements-eg-na-k.md), [asta-19](2026-05-14-asta-19-the-combinatorial-line-matching-algorithm-comb-experiences-combinatorial-candidate.md), [asta-24](2026-05-14-asta-24-the-extremely-low-f1-score-of-the-comb.md) — **tension**: asta-24 argues for *looser* comb thresholds; verify scope of #166. |
| #168 | partition-function kernel basis (log10→natural log) — 18-OOM fix | [asta-28](2026-05-14-asta-28-the-016-macro-f1-gap-to-the-literature-ceiling.md), [empirical-01](2026-05-14-empirical-01-universal-misses-53-pairs-53174-true-spectrum-element-pairs-universally-missed-by.md), [empirical-02](2026-05-14-empirical-02-top-missed-elements-top-universally-missed-elements-si-15-mg-11-al.md) — physical modeling improvement; may close part of the 0.16 macro-F1 gap |
| #169 | multi-metric F1 gate (schema v2) | [empirical-03](2026-05-14-empirical-03-vrabel-rp-confound-4953-universal-misses-are-on-vrabel2020_soil_benchmark-rp30000.md) — partway to per-rp-bucket reporting |

---

## Provenance

- Asta autodiscovery run: `1f79815f-78b6-4f74-9c06-27127067c326`
- 30 experiments, 18 marked `is_surprising` by Asta, 5 strongly confirmed
  (surprise ≥ +0.25), 13 refuted (surprise ≤ -0.49), 12 neutral / weakly
  signal'd.
- Empirical n=33 analysis: derived from the verifier benchmark export
  used by PR #169's schema-v2 gate.
- Generation date: 2026-05-14.
