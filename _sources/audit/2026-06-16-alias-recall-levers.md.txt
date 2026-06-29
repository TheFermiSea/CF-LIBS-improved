# ALIAS recall levers — research, implement, benchmark-gate, compare

**Date:** 2026-06-16 · **Branch:** `id-benchmark-overhaul`
**Goal:** raise ALIAS recall (paper-faithful baseline P=0.844 **R=0.434** F1=0.573 on `w5_paper_v2`,
92 spectra, 11 candidates, 2% 3-way detection floor) without sacrificing precision.
**Method:** workflow `we8v1x1ov` (paper deep-read + literature via asta-documents/web + per-element
failure diagnosis → ranked candidates → worktree-isolated implement+benchmark-gate), then **re-measured
the contaminated candidates by hand on the correct baseline** (several worktrees branched off a stale
pre-fix base `4de7e12b`, so their verdicts were invalid).

## Result — one kept lever

| config (w5, 92 spectra, 2% floor) | P | R | F1 | vs baseline |
|---|---|---|---|---|
| baseline (intensity peak mode, C_th 0.5) | 0.844 | 0.434 | 0.573 | — |
| **2nd-derivative peak detection (now default)** | **0.888** | **0.448** | **0.596** | R +0.014, P +0.044, **F1 +0.023** |

**Kept:** paper §3.1 / Fig 2 peak detection — `find_peaks` (prominence) **on the negative
2nd-derivative spectrum** (curvature-domain MAD prominence + nearest intensity max within ±2 +
amplitude floor), replacing the previous "find_peaks on intensity, d2 as a gate". Recovers 3 true
elements (TP 92→95) AND removes 5 FPs (17→12) — recall *and* precision up. Paper-faithful and strictly
better, so it is the default (`peak_mode="second_derivative"`; legacy `"intensity"` retained opt-in).
All ALIAS/integration tests green (71 passed, 3 xpassed).

## Rejected (all benchmark-gated on the correct baseline)

| lever | result | why rejected |
|---|---|---|
| **k_shift calibrated-residual** (subtract per-element median shift, paper eq 5) | P 0.844→0.793, R 0.434→0.434, F1 →0.561 | residual-subtraction raised k_shift for confounders as much as true elements → +7 FP, no recall gain |
| **per-element (T,nₑ) k_sim-selection** (paper §3.8 "best pair") | recall *falls* (valid same-worktree A/B: R 0.259→0.241) | picking the k_sim-maximizing (T,nₑ) per element overfits the matched set; commits a worse emissivity vector for k_rate/k_shift |
| **paper-literal fusion** (intensity-seeded ±Δλ/2) | F1 0.573→0.444 (prior turn) | tighter bins fragment line-rich elements → k_rate down. Repo chain-fusion empirically better. |
| **10-line-bin emissivity threshold** (paper §3.6) | inconclusive on correct baseline | worktree measured on stale base; not re-run (lower priority after the above) |

## Operating-point frontier (not a fix — a dial)

`C_th` is user-defined (paper default 0.5). Lowering it slides down the P-R curve:

| C_th | P | R | F1 |
|---|---|---|---|
| 0.50 (paper default) | 0.888 | 0.448 | 0.596 |
| ~0.45 | 0.794 | 0.509 | 0.621 (measured pre-2nd-deriv) |

So higher recall is available by lowering `C_th` at a precision cost — exposed via the
`detection_threshold` constructor arg for callers who want a recall-weighted operating point.

## Takeaway

Recall on this hard 11-candidate multi-element corpus is largely **structural** at the paper's
`C_th=0.5` (the paper itself misses borderline elements, e.g. Mn at k_det 0.41 in its Fig 6). The
2nd-derivative detector is the one paper-faithful change that lifts it cleanly. Further recall gains
require either a recall-weighted `C_th` (precision tradeoff) or corpus/instrument changes (higher RP),
not more identifier tuning.

## Research artifacts

Full per-element failure diagnosis, paper deep-read, and literature scan in workflow `we8v1x1ov`
output. Paper PDF: `docs/refs_noel2025_alias.pdf`. The faithful-fix baseline: `2026-06-16-alias-faithful-fix.md`.
