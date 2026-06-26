# Complete-DB Reset — Status & Next-Phase Plan (2026-06-26)

After the atomic DB was rebuilt from the gold-standard `ASD59_dump.sql`, **every
prior benchmark/metric was invalidated** (they were computed on a DB that held
~30% of lines, missing IPs, and no stage III). This is the state of the reset.

## 1. Foundation — DONE, validated, committed + pushed (branch `v4/m5-kurucz-atomic`)

| Item | Status | Evidence |
|---|---|---|
| Atomic DB rebuilt from gold standard | ✅ | 203,695 lines / 62,752 levels (I-III) / 324 IPs; validated vs live NIST (18/18 levels, 17/17 IPs) — `docs/v4/atomic_db/COMPLETENESS-VERIFICATION.md` |
| Partition-function overhaul | ✅ | direct-sum over complete DB authoritative; g0 fixed (Fe I 1→9); U(T) within ~10% of Barklem/Irwin; IPD handles high-T; 31/31 tests (`scripts/validate_partition_vs_literature.py`) |
| `get_transitions` NULL-`aki` crash | ✅ fixed (`2f15cfa`) | complete DB's 74k observation-only lines crashed every forward-model path |
| `_level_cache` stale-after-reingest | ✅ fixed (`c22fe3b`) | token-versioned like `_spec_cache` |
| Forward model (Saha-Boltzmann) | ✅ NIST parity PASS | Fe ionization fractions within 5% (`scripts/validate_nist_parity.py --db ASD_da/...`) |

## 2. Re-benchmark on the complete DB (first VALID numbers)

| Pipeline | Result |
|---|---|
| **DED constrained (Ti-6Al-4V, the real goal)** | Al 0.82 / Ti 1.43 / V 0.87 wt% rmsep, V nominal bias +0.73, 27/27 converged |
| **Solver head-to-head (SuperCam labcal, n=15)** | **iterative 2.31** / ilr 11.0 / csigma 16.8 (median mass RMSE) — iterative wins |
| **NIST forward-model parity (Fe)** | PASS (I −4.5%, II +1.7%, III exact) |
| **Open-element ID (small_v1, 12 spectra)** | ALIAS F1 0.063 (recall collapse), Comb 0.329 (over-detects), Corr 0.327 — **mis-tuned for the dense catalog** |

## 3. The hard finding — open-element ID needs RE-ARCHITECTING (not re-tuning)

The 7× denser catalog makes open ID fundamentally harder (more lines → more
chance matches). Root causes + why simple knobs fail:
- **ALIAS:** `k_det ∝ k_rate = matched_emissivity / total_emissivity_above`. The
  larger catalog inflates the expected-line denominator → line-rich *true*
  elements (Fe/Ni) get penalized while line-poor absent ones pass by chance. The
  k_det values are systematically low AND poorly separated. Threshold C_th 0.5→0.3
  only recovered recall 0.04→0.21 (band-aid, precision dropped); `max_lines` had
  no effect. **Fix = a chance-corrected detection metric:** normalize k_det by the
  *expected random-match rate given the catalog density in-window* (so a match is
  only credited above what the dense catalog would produce by coincidence), and/or
  a high-recall-Comb → precision-filter (NNLS/BIC) hybrid.
- **Comb:** recall 0.96 already; over-predicts (precision 0.20). Tightening gates
  was marginal (F1 0.329→0.341). Better used as the high-recall front of a hybrid.

**This is deep and SECONDARY to the DED goal** (DED uses a known element set, so
identification is bypassed).

## 4. Band-aid inventory (from workflow `w80qmlwsk`): 107 total, 72 DB-compensation

High-severity, now mostly NON-FIRING on the complete DB (cosmetic to remove, low
accuracy value): missing-IP `15.0 eV` fallback (iterative.py ×8 + quality.py:347),
generic `{1:25,2:15}` partition tier-c, bayesian `log(25/15/10)` placeholder,
`min_relative_intensity` SQL floor. Behavior-changing (real value): **stage-III
enablement** (iterative.py:489 `(1,2)`, alias.py ×3 — solver run confirmed "stage 3
not supported; skipping"), Stark-width n_e diagnostic (vs the 1-atm pressure-balance
band-aid). Cleanups done: partition `_level_cache`/docstrings.

## 5. Suggested next phase (benefits from user direction on approach + priority)

1. **DED accuracy (real goal):** Stark-width n_e diagnostic (replace pressure-balance);
   stage-III Saha for hotter regimes; instrument calibration as a first-class input.
   Cr under-estimate (Saha unobserved-stage) — `reference_ded_synthetic_benchmark`.
2. **Open-ID re-architecting:** chance-corrected k_det + hybrid (above). Larger effort.
3. **Honest band-aid removal:** the non-firing solver fallbacks → loud/error (the
   complete DB guarantees coverage), unit-test-gated.
4. **Latency (deferred):** ALIAS RANSAC wl-calibration is ~84 min on the full 288
   corpus (7× reference lines) — index/cap the calibration reference set.

**Bottom line:** the atomic data, partitions, forward model, and the DED-constrained
pipeline (the real goal) are CORRECT on the gold-standard catalog. The remaining work
is deep (ID re-architecting, DED-accuracy physics) + cosmetic (non-firing cleanups).
