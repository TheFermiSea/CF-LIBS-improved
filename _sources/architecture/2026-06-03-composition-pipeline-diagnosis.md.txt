# Composition Pipeline Diagnosis & Ranked Remediation Plan

**Date:** 2026-06-03
**Worktree:** `/tmp/cflibs-comp` @ `fix/composition-pipeline-2026-06-03` (HEAD `616dec8`)
**DB under test:** `/tmp/cflibs-comp/ASD_da/libs_production.db` (28,727 lines / 175 species / 9,448 energy levels / 146 partition-function rows)
**Method:** Diagnose → adversarially Verify → synthesize. Every blocker below survived an independent adversarial reproduction. Findings the verifier *downgraded* (severity or root cause) are flagged inline.

---

## 1. Headline Verdict

**The atomic database is NOT broken in the way the user feared — the line/level/IP *inputs* are clean and NIST-consistent — but it carries two genuine, separately-real data defects that DO bias composition on the manifold/Bayesian/forward paths: (C) the polynomial partition-function table is 1.7–2.6× too low at LIBS temperatures for iron-group workhorse species, and (A4) a 20× Stark-width unit/reference-density convention mismatch between the data and every runtime consumer. The default CPU `invert`/`analyze` solver is structurally immune to (C) (it prefers direct-sum) and to the (A4) *tolerance* path (alias passes `transition=None`), which is why the user's "DB isn't validated against truth" hypothesis is *half right*: the validation gate that exists (`validate_partition_functions.py`) silently tests the *correct* code path (direct-sum) and never exercises the broken polynomial artifact that production JAX paths actually consume.**

**Peak-ID underperforms for ALGORITHM reasons, not data reasons.** The recall collapse (alias macro-F1=0.185, recall=0.144) is driven by ALIAS gates: a Boltzmann-consistency fit that pools all ionization stages into one regression (collapsing R² to ~0.01), then a fixed R²≥0.85 hard-reject. Fixing the DB will NOT move peak-ID F1 — U(T) and Stark widths do not enter the production identifier's line-matching at all. The literature offers **no published macro-F1** for physics-based per-element line-ID; the defensible target is the project's own protocol floor: **per-element recall ≥ 0.6, macro-F1 ≥ 0.7**. The production-best identifier `hybrid_union` already sits at **0.688**, just below that bar; the open levers are the ALIAS ion-stage + R² gates (alias arm) and the spectral_nnls relative-magnitude detection gate (precision arm).

**Bottom line for fix ordering:** the highest impact-per-effort items are the two ALIAS gate fixes (cheap, directly recover recall on the alias arm) and the spectral_nnls relative-magnitude gate (cheap, cuts the 5.67 FP/spectrum). The DB fixes (PF re-fit, Stark ÷20) are real and worth doing for *composition correctness on the manifold/Bayesian/forward paths*, but they are larger and do not touch the F1 baseline.

---

## 2. Database Validation Results vs Truth

This section directly answers the user's hypothesis that the DB isn't validated against truth. **Verdict: the line/level/IP inputs ARE consistent with truth; the partition-function polynomial and Stark-width columns are NOT.**

### 2.1 Partition functions — BROKEN (prior-audit defect C, CONFIRMED REAL)

Three-way cross-check (stored polynomial vs DB direct-sum vs Barklem & Collet 2016), at LIBS temperatures, all numbers reproduced by the verifier:

| Species | T (K) | U_poly / B&C16 | U_directsum / B&C16 | poly / directsum |
|---|---|---|---|---|
| Fe I | 8000 / 10000 / 12000 | **71.9% / 56.7% / 44.2%** | 99.6% / 98.3% / 95.1% | 0.58 @10kK |
| Ti I | 8000 / 10000 / 12000 | **54.1% / 38.5% / 28.2%** | 95.6% / 90.9% / 84.9% | 0.42 @10kK |
| Cr I | 8000 / 10000 / 12000 | **59.1% / 39.2% / 26.9%** | 97.7% / 94.1% / 89.1% | 0.42 @10kK |
| Ni I | 8000 / 10000 / 12000 | 95.4% / 92.6% / 87.7% | 99.7% / 98.8% / 96.5% | 0.94 @10kK |
| Cu I | 8000 / 10000 / 12000 | 99.0% / 97.0% / 90.6% | 99.5% / 97.0% / 90.3% | 1.00 @10kK |

- **Direct-sum from the DB's own `energy_levels` tracks B&C16 to within 0.4–5% across 8000–10000 K** → the energy_levels table is essentially complete and is NOT the cause. The broken artifact is the *polynomial fit*.
- **28 of 146 polynomial rows** have `min(U_poly/U_directsum) < 0.80` somewhere in 6000–12000 K; **0 rows overstate**. Since direct-sum is a strict lower bound on true U, `poly < directsum` is physically impossible for a correct fit → definitive red flag.
- **Source correlation is the smoking gun:** `Irwin1981` 11/12 bad, `NIST_ASD_fit` 17/99 bad, `direct_sum_fit_v1` **0/35 bad**. The fix is data regeneration (re-fit via the same recipe that produced the clean `direct_sum_fit_v1` rows), not a code/math change.
- **Saha ratio error** (U_II/U_I vs direct-sum): Fe +12/+23/+34%, Cr +73/+108/+138%, Ti +36/+61/+86%, Ca +37/+77/+122% at 8000/10000/12000 K.
- **Composition impact (non-cancelling):** inter-element concentration ratios skewed −31% to +60% at 0.8 eV (e.g. C(Si)/C(Fe) +59%, C(Cu)/C(Fe) +60%, C(Ca)/C(Fe) −31%) because the U-bias is species-dependent.

**Path asymmetry (this is the load-bearing scoping fact):**
- SAFE (prefers direct-sum, bypasses bad poly for the 144/146 species with levels): `cflibs/plasma/saha_boltzmann.py:184-187`, `cflibs/inversion/solve/iterative.py:255-278` & `:754-757` (the **default `invert`/`analyze`/`bayesian` CPU solver**), `closed_form.py:81`.
- BROKEN (polynomial-only, NO direct-sum fallback): `cflibs/manifold/batch_forward.py:409-420` (verified: `polynomial_partition_function_jax`, U in Boltzmann denominator at :420), `cflibs/manifold/generator.py:362`, `cflibs/plasma/anderson_solver.py:255` (test/benchmark-only, 0 production refs), and the Bayesian atomic loader.

**Commands:** `PYTHONPATH=/tmp/cflibs-comp .venv/bin/python scripts/validate_partition_functions.py --db-path ASD_da/libs_production.db --no-download` ; custom 3-way cross-check `/tmp/pf_check.py`, `/tmp/pf_cross.py`, full 146-row audit `/tmp/pf2_audit.py`, B&C16 truth `/tmp/pf2_truth.py`, path trace `/tmp/pf_path.py`, forward impact `/tmp/pf_impact.py`.

> **Verifier downgrade:** PF-1/2/3/4 were all confirmed REAL and BLOCKING but **severity downgraded from "blocker" to "major"** because the default CPU composition path is genuinely immune (direct-sum-first for all 12 workhorse species). The blast radius is the manifold/JAX/Bayesian sub-paths, not the whole pipeline.
> **Refuted sub-claims:** (a) "direct-sum matches B&C16 to <2% everywhere" is overstated — Ti I direct-sum is 91% and Cr II is 75–84% at 10–12 kK (those levels ARE incomplete), so a re-fit cannot beat the underlying level-table completeness at the hot edge. (b) The PF-3 citation "kernels.py:278/294 compute_spectrum_jax" is WRONG — `cflibs/plasma/kernels.py` has no such symbol; the real polynomial kernel is `batch_forward.py:409` (corrected and re-verified here).

### 2.2 DATA-5 (the "26% partition-function failure" headline) — DOWNGRADED to not-a-defect-as-stated

`run_nist_validation.py` reports "1482/2014 within 5%" (= 26% fail), which Phase-1 read as corroborating defect C. **The verifier refuted the attribution:** that 26% is **fixture drift**, not a polynomial-vs-truth gap. `run_nist_validation.py` compares the *current direct-sum* against a stale local fixture `tests/data/nist_reference/partition_functions.json` (whose own metadata says it was "computed by summing g·exp(−E/kT) over levels in libs_production.db" but no longer matches a current direct-sum — e.g. Ca I @12000 K: prod_calc 6.43 vs fixture 1.18, +447%). It does not measure the polynomial at all. Follow-up: regenerate or delete the stale fixture. **This does not change the PF verdict — defect C is independently confirmed via B&C16 — it just removes a misleading number.**

### 2.3 Stark widths — BROKEN convention (prior-audit defect A4, CONFIRMED REAL, direction REFUTED)

- **Provenance:** only 244/28,331 (0.86%) Stark widths are line-specific literature (`stark_w_source='stark_b'`); 81% are λ²-scaled from ONE reference width per species (`konjevic_lambda_sq_scaled`), 16% interpolated, 2% hydrogenic. Width carries no line-resolved physics for 99% of lines (Fe II: 4730 distinct values / 4809 lines = deterministic λ² function).
- **The 20× convention mismatch (verified end-to-end):** the data is stored & tested as **FWHM at n_e=1e17** (`scripts/ingest_stark_b.py:7-9,94` "n_e=1.0e17"; `tests/test_stark_provenance.py:85-96`), but every runtime consumer treats it as **HWHM at n_e=1e16** (`cflibs/radiation/stark.py:17` `REF_NE=1.0e16`, `:77`, `:102` FWHM=2×HWHM; `cflibs/inversion/common/element_id.py:184` docstring literally says "HWHM at REF_NE=1e16"; `kernels.py:377`; `batch_forward.py:86-87`). Net inflation = ×10 (density) × ×2 (HWHM→FWHM) = **×20**, reproduced exactly: Al I 396.15 stored 4.5 pm → runtime omega_stark = **90.0 pm** at 1e17/10000 K; Ca II K 8.4 → 168 pm; Fe II 430.317 33 → 660 pm.
- **The original audit's "Al ×10 *under*" is REFUTED in direction:** with current data+runtime the line is ~90 pm = ~2.25× *over* (vs Konjević ~40 pm), because the ×20 convention bug overpowers the too-small stored value.
- **Two passing test files encode contradictory conventions** (`tests/test_stark.py` HWHM@1e16 vs `tests/test_stark_provenance.py` FWHM@1e17) — a guardrail gap; both pass because they test disjoint things.

**Commands:** DB provenance `GROUP BY stark_w_source`; numeric repro via `stark_hwhm`/`get_wavelength_tolerance` at 1e17/10000 K; convention grep across `stark.py`/`kernels.py`/`batch_forward.py`/`element_id.py` + the two test files.

> **Verifier downgrade:** A4-CONV-2 confirmed REAL + BLOCKING but **severity blocker→major**; the "peak-ID F1" framing is overstated. The production-best identifier (`hybrid_union`/alias, F1=0.688) is immune because alias passes `transition=None` → omega_stark=0 (`alias.py:3088-3093`, verified). The load-bearing harm is the **forward-model / Voigt / manifold composition path** (`apply_stark=True` after A2 → every Stark line over-broadened ×20 at ps-LIBS densities), plus inflated tolerances on the weak comb/correlation arms. A4-PROV-1 (λ²-scaling) downgraded to **minor** — the default solver and top identifier never read `stark_w` at all.

### 2.4 Lines table vs NIST ASD — CLEAN (checked, found OK)

- **Zero structural defects across all 28,727 lines:** no NULL/NaN/negative/zero ei,ek; no zero/negative gi,gk; no ek≤ei; no wavelength-unit errors (no Ångström-as-nm).
- **Energy–wavelength self-consistency:** `|(E_k−E_i) − hc/λ| / (hc/λ)` < 1% for **100% of lines**, worst single line 0.16% (Al III 569.66). Median residual 2.8e-4 = the expected air-vs-vacuum refractive-index offset.
- **13/13 hand-checked strong workhorse lines** found at correct wavelength to <0.001 nm with Aki/ek matching NIST (Ca II H/K, Mg I 285.21, Na I D1/D2, Al I 396.15, Mn I 403.08, Cr I 425.43, Ni I 341.48, K I 766.49 all 1.00× Aki). Si I 288.16 (1.15×) and Ti II 334.94 (1.68×) are within the spread of competing NIST source datasets, not errors.
- **The real peak-ID hazard is physical line density, not data errors:** 206 cross-element strong-line (Aki>1e7) wavelength collisions within 0.005 nm among workhorse elements (e.g. Cr I 404.878 vs Mn I 404.876). This is intrinsic LIBS physics — it explains low alias recall and the spectral_nnls FP rate, NOT bad atomic data.

**Commands:** global null/range/consistency queries via direct sqlite3-in-python; 13-line by-hand NIST spot-check; collision scan.

### 2.5 species_physics (ionization energies) & energy_levels — CLEAN (checked, found OK)

- **All 10 workhorse first ionization potentials match NIST within 0.01%** (Fe 7.9020 vs 7.9025; Ca, Si, Al, Mg, Na, K, Ti, Mn, Cr all |diff|<0.01%); II→III within 0.08%. Zero NULL/zero/negative IPs or masses; IPs monotonic with sp_num for every element.
- **energy_levels (9448 rows / 144 species):** zero negative/NULL energies, zero g≤0, max energy 24.86 eV (no cm⁻¹ contamination), zero exact-duplicate rows. The 79 "same-energy-different-g" groups are physically real fine-structure multiplets (verified Mg I 7.4717 eV → J-split g=1,3,5 cross-checked against line gk).
- **Minor gaps (all negligible at ps-LIBS T):** 1,075/28,727 lines (3.7%, 97% with E_k>5 eV) reference an upper level absent from energy_levels → slight direct-sum under-count, Boltzmann-suppressed. Na II has 0 levels/0 PF but 0 lines and IP 47.3 eV (never populates). 18 sp≤2 species have <10 levels but all (except Na II) have a polynomial-PF fallback.

**Commands:** workhorse IP-vs-NIST diff%; IP monotonicity scan; energy_levels range/duplicate/multiplet checks; orphan-line count.

### 2.6 Tooling gaps found

- **`scripts/validate_partition_functions.py` does NOT validate the stored polynomial** — its "[Ours]" column is computed via `direct_sum_partition_function` (line 254, verified), the *correct* path, then compared to B&C16. This is why the broken polynomial shipped: the only nominal gate silently exercises a different (correct) code path. **Fix: add a 3rd column evaluating `partition_function_for(...).at(T)` (the stored polynomial) and gate CI on poly-vs-reference.**
- **`tests/test_partition_function_parity.py` passes tautologically** — it compares the polynomial against a *deliberately truncated* direct-sum (`PARITY_TRUNCATION_EV`) tuned to Irwin's observed-only convention, so 11 tests pass while the polynomial is 30–60% off B&C16.
- **`scripts/validate_kramida_2024.py` is BROKEN against the schema** — queries column `ion_stage`; actual column is `sp_num` (`scripts/validate_kramida_2024.py:59`); also needs absent cluster DBs. Cannot run as a lines-table validator.
- **`scripts/validate_nist_parity.py` does NOT validate per-line values** — only ionization fractions + spectrum shape, and its lone Fe reference is hardcoded at T=0.8 eV (`:33-38`), so running at the requested T=1.0 eV prints "differences" that are pure Saha shift, not data defects.
- **`scripts/compare_stark_vjbh.py` is NOT a Stark validator** — it is a before/after composition-benchmark delta tool requiring two `composition_records.parquet` artifacts (absent); cannot confirm/refute A4.
- **`scripts/benchmark_synthetic_identifiers.py` emits only presence P/R/F1/FPR, NOT composition metrics** — yet runbook ADR-0001 §6 (line 174) gates "within 5% … (Aitchison distance, ILR error, top-K recall)" against it, so §6 is unenforceable as written (bead qsnf). **Does NOT corrupt the F1 baseline** (separate `UnifiedBenchmarkRunner` harness, real data n=33).

---

## 3. Peak-ID Diagnosis

### 3.1 Root causes of the F1 gap (algorithm, not data)

Reproduced on a bounded synthetic corpus (224.6–265.3 nm UV, T~0.8–1.3 eV). For a **pure-Fe** spectrum, Fe — the only true element — is hard-rejected by stacked ALIAS gates:

1. **Ion-stage pooling (blocker, CONFIRMED, verifier kept severity=blocker).** `_collect_boltzmann_observations` (`alias.py:2728-2756`, verified) appends a `LineObservation` for every matched line and records `ionization_stage` but **never groups by it**; `_boltzmann_consistency_check` (`:2814-2818`) runs ONE `ln(Iλ/gA)`-vs-E_k regression over the pooled set. Verifier reproduced: Fe I-only R²=0.997, Fe II-only R²=0.999, **pooled R²=0.012** (matches Phase-1's 0.012). Collapse is driven specifically by the Saha inter-stage population offset. This is the single biggest algorithmic recall killer for iron-group (Fe/Cr/Ti/Mn/Ni) elements whose UV lines span stages.
2. **Fixed R²≥0.85 hard gate (major, CONFIRMED, verifier downgraded blocker→major).** `alias.py:839-840` defaults `boltzmann_r2_min=0.85`, `r2_gate_mode='fixed'`; `:1781-1782` converts a sub-0.85 fit into `detected=False` (verified `_r2_gate_rejects` at `:2614-2615`). Verifier reproduced: pure-Fe Fe R²=0.342–0.772 → rejected at default; disabling/lowering the gate flips Fe to `detected=True`. Downgraded because non-strict presets already ship (`v2`/`high_recall_v2`/`consensus_voter` use `adaptive_t`), and the production `hybrid_union` is NNLS-weighted; **a clean lift also requires the ion-stage fix and the cold-biased temperature estimator** (the adaptive_t cold floor only relaxes below 5500 K; ps-LIBS spans 5800–15000 K).
3. **spectral_nnls relative-magnitude detection gate (major, CONFIRMED, verifier corrected the mechanism).** spectral_nnls' 5.67 FP/spectrum is real (precision 0.42 is the bottleneck), but the verifier **refuted the "collinear basis → near-singular AᵀA" root cause**: the Gaussian basis is near-orthogonal (max off-diagonal corr 0.017, cond ~38, full rank) and the proposed Voigt fix *increases* collinearity. The true mechanism is matched-filter projection of a structured residual onto absent-element basis vectors, with the raw `element_snr≥3.0` rule (`spectral_nnls.py:481`) firing on tiny coefficients (~2.6% of total signal). **The load-bearing fix is the parenthetical secondary one: gate on coefficient magnitude relative to total (~5% floor), which cut FPs 2→0 in repro while keeping both true elements.** The Voigt-basis rewrite (LIT-F1-2 / A3) is a follow-up, gated on Stark-width accuracy (A4).

### 3.2 Bead triage (genuinely open vs stale-OPEN)

| Bead | Claim | Status | Verdict |
|---|---|---|---|
| n3rf.4 | resonance filter strands all-resonance elements | **stale-OPEN (FIXED)** | `alias.py:2683-2696` pre-scan, landed `4794d04`; Al recall 0.0→0.5. Close. |
| jbfg.1 | JAX/CPU FISTA parity | **stale-OPEN (FIXED)** | `max_iter=50000` default (`alias.py:306/377/433`); gap closed 0.060→0.017. Close. |
| n3rf.2 | alias_high_recall regresses | **stale-OPEN (FIXED in code)** | `b809d4b` high_recall inherits v2 gates; Phase-7 number not recorded. Close after confirming. |
| wzus | min-3-lines + per-element R² gate | **stale-OPEN (IMPLEMENTED)** | `alias.py:679-683` + `:638-651`. Close. |
| 5thd | comb is FN-bound | **OPEN, downgraded** | Symptom real (`comb.py:1160` dilution denominator), but **verifier refuted FN-bound thesis**: real-data comb is *precision*-bound (P=0.018, recall=0.091); the binding gate is the amplitude gate + `max_lines_per_element`, not the denominator; the proposed denominator fix inflates FP scores. comb is NOT in `hybrid_union`. **Non-blocking.** |
| s1qr.2 | comb FP (7-element false match) | **OPEN, fold into 5thd** | Not reproducible via forward-model path; its committed fix landed in the wrong function. Re-scope. |
| qsnf | synthetic harness lacks composition metrics | **OPEN, independent** | Real (ADR-0001 §6 unenforceable) but does NOT block F1 work. Amend §6 to gate on presence metrics. |

### 3.3 Literature target

There is **no published macro-F1 for physics-based LIBS line/element identification.** Labutin/Zaytsev/Popov 2013 (correlation) and Noel et al. 2025 (ALIAS) report only qualitative success; Gajarska 2024 (comb) reports none and explicitly notes the method is "highly prone to false positives." The 0.88–0.99 "F1" numbers in LIBS literature are sample *classification*, a different task — do NOT adopt them. **Defensible target = project protocol floor: per-element recall ≥ 0.6, macro-F1 ≥ 0.7.** `hybrid_union` is at 0.688. The repo methods are faithful to their cited sources (sound-but-buggy, not mis-designed); the gap to literature is execution, not algorithm selection. (Minor: the ALIAS docstring cites the wrong arXiv ID 2501.01057 — an unrelated HPC paper; the real source is Noel et al., Spectrochim. Acta B 2025.)

---

## 4. Confirmed-Blocker Table (ranked by impact × confidence ÷ effort)

Only items the Verify phase confirmed as real **and** blocking are included. Ranked for top-down execution.

| # | ID | Title | Category | Lift | Effort | Conf | Depends on | Cite (file:line) + verifying command |
|---|---|---|---|---|---|---|---|---|
| 1 | ALIAS-BOLTZ-IONMIX-1 | ALIAS pools ionization stages into one Boltzmann fit → R²≈0.01 → hard-reject | peak_id | +0.05–0.12 alias macro-F1 (recall-bound) | small | high | — | `alias.py:2728-2756`, `:2814-2818`. Repro: per-stage R²=0.997/0.999 vs pooled R²=0.012 at T=0.9 eV, ne=1e17. |
| 2 | NNLS-GAUSS-BASIS-4 | spectral_nnls raw SNR≥3 detection rule fires on ~2.6%-signal coefficients → 5.67 FP/spec | peak_id | spectral_nnls F1 0.44→~0.52–0.58 (precision) | small | high | — | `spectral_nnls.py:481`, basis `basis_library.py:179`. Repro: relative-magnitude ≥5% gate cut FP 2→0, kept both true elements. **Fix is the relative-magnitude gate, NOT the Voigt rewrite.** |
| 3 | ALIAS-R2GATE-2 | Fixed R²≥0.85 hard gate rejects valid ps-LIBS fits | peak_id | +0.05–0.10 alias macro-F1 (with #1) | small | high | ALIAS-BOLTZ-IONMIX-1; ALIAS-TEST-EST-6 | `alias.py:839-840`, `:1781-1782`, `:2614-2615`. Repro: pure-Fe R²=0.342–0.772 rejected at default; flips to detected when gate disabled/lowered. |
| 4 | A4-CONV-2 | 20× Stark unit/reference-density convention mismatch (data FWHM@1e17 vs runtime HWHM@1e16) | physics | composition-correctness (forward/manifold/Bayesian); ~0 F1 | medium | high | — | data `ingest_stark_b.py:7-9,94`; runtime `stark.py:17,77,102`, `element_id.py:184,217`, `kernels.py:377`, `batch_forward.py:86`. Repro: Al I 4.5 pm → 90 pm at 1e17/10000 K (×20 exactly). |
| 5 | PF-1/PF-2 | Polynomial partition functions 1.7–2.6× too low at LIBS T (Irwin1981 + bad NIST_ASD_fit rows) | database | composition ratios on manifold/Bayesian/forward paths skewed −31%/+60%; ~0 F1 | medium | high | — | table audit (28/146 rows poly<0.80×directsum); consumer `batch_forward.py:409-420`, `generator.py:362`. Repro: `validate_partition_functions.py --no-download`; `/tmp/pf2_audit.py`. |
| 6 | PF-3/PF-4 | Manifold/JAX/Bayesian paths have no direct-sum fallback; validation gate tests the wrong path | database | enables #5 fix verification; defense-in-depth | medium | high | PF-1/PF-2 | `batch_forward.py:409` (poly-only), `validate_partition_functions.py:254` (gate uses direct_sum). Repro: path trace `/tmp/pf_path.py`. |
| 7 | CORR-RELGATE-5 | Correlation scores present elements with negative whole-spectrum Pearson → clipped to 0 → min_confidence rejects | peak_id | correlation F1 0.17→~0.4; ~0 on ensemble | medium | high | — | `correlation.py:507/551` clip, `:384` gate. **Verifier refuted the relative-threshold root cause** (gate never fires; needs ≥3 nonzero scores, scorer produces 1–2). Real fix = residual/sequential per-element scoring. |

**Excluded by the Verify phase (do NOT fix as stated):**
- **ALIAS-NNLS-PLOCAL-3** — verifier marked `real=false`: faithful pure-Fe repro gives Fe P_local=1.0 (not 0.000), still not detected; the named P_local/nnls collinearity gates are NOT the operative cause. The real recall driver is `_fast_screening` (`alias.py:2552`) + degenerate Boltzmann R² — already captured by #1/#3.
- **DATA-5** — verifier marked `real=false`: the 26% figure is stale-fixture drift, not a polynomial defect. (Defect C itself is captured by #5/#6.)
- **A4-PROV-1, A4-AL-3, A4-TOL-4** — confirmed real but `blocking=false`/minor: the default solver and `hybrid_union` never read `stark_w`; the genuine harm is folded into #4 (A4-CONV-2).
- **TRIAGE-5thd-comb-FN, ALIAS-TEST-EST-6** — non-blocking (comb not in production hybrid; T-estimator is a second-order amplifier feeding #3).

---

## 5. Wave-Ordered Remediation Plan

1. **Wave 0 (zero-risk cleanup):** Close the 4 stale-OPEN beads (n3rf.4, jbfg.1, n3rf.2, wzus) after confirming their landed commits; delete/regenerate the stale `tests/data/nist_reference/partition_functions.json` fixture so `run_nist_validation.py` stops reporting the phantom 26%; amend ADR-0001 §6 (qsnf) to gate on presence P/R/F1.
2. **Wave 1 (cheap peak-ID recall, alias arm):** Fix ALIAS ion-stage pooling (#1) — group `_collect_boltzmann_observations` by `ionization_stage`, fit the dominant stage (≥3 lines) or Saha-map to neutral plane, report best per-stage R². Land with the R²-gate relaxation (#3, default `adaptive_t` or ~0.55) since they must ship together; guard the cold-biased temperature estimator (ALIAS-TEST-EST-6) in the same change.
3. **Wave 2 (cheap peak-ID precision, nnls arm):** Add the spectral_nnls relative-magnitude detection gate (#2) — gate on coefficient ≥ ~5% of total signal instead of raw `element_snr≥3.0`; re-measure FP/spectrum.
4. **Wave 3 (Stark correctness):** Fix A4-CONV-2 (#4) as ONE unit — pick a single convention project-wide (recommend redefine DB column as HWHM@1e16, i.e. ÷20, OR set runtime `REF_NE=1e17` and drop the ×2), reconcile the two contradicting test files, and add a canonical data→element_id→kernels anchor assertion. **Must update `tests/inversion/common/test_get_wavelength_tolerance.py` (pins the buggy convention) in the same commit** to avoid double-counting.
5. **Wave 4 (partition-function regeneration):** Re-fit the 12 Irwin1981 + 17 bad NIST_ASD_fit rows via direct-sum on 2000–25000 K (the `direct_sum_fit_v1` recipe, 0/35 failures), monotone-constrained; verify each vs B&C16 to <5–10% before storing (#5). Add the direct-sum fallback to `manifold/batch_forward.py` / `generator.py` AtomicSnapshot so all four solver paths match (#6), and add the 3rd-column poly-vs-reference CI gate to `validate_partition_functions.py`. Accept that Cr II / hot-edge Ti I cannot beat level-table completeness.
6. **Wave 5 (follow-ups, gated on Wave 3/4):** Voigt+Stark NNLS basis (LIT-F1-2 / A3) — only after A4 widths are validated; correlation residual/sequential scoring (#7); ingest line-resolved Konjević/STARK-B/Aragón widths to replace λ²-scaling; expand `test_stark_provenance.py` ANCHORS to include Al I 396.15; fix the broken `validate_kramida_2024.py` schema (`ion_stage`→`sp_num`); fix the ALIAS arXiv citation.

---

## 6. Checked-and-Found-OK (do not re-audit)

- **Lines table (28,727 rows):** zero structural defects; 100% energy–wavelength consistent to <1% (worst 0.16%); 13/13 strong workhorse lines correct vs NIST. (§2.4)
- **species_physics IPs (175 rows):** all 10 workhorse first IPs within 0.01% of NIST; monotonic; no NULL/zero/negative. (§2.5)
- **energy_levels (9448 rows):** clean energies/degeneracies; max 24.86 eV (no cm⁻¹ contamination); fine-structure multiplets are real, not duplicates; direct-sum tracks B&C16 to <2% for Fe I/II, Ti II at 8–10 kK → the level table is NOT the cause of defect C. (§2.5)
- **Default CPU `invert`/`analyze`/`bayesian` solver:** structurally immune to defect C (direct-sum-first, `iterative.py:255-278/754-757`) and to the Stark tolerance bug (alias `transition=None`). Do not chase composition bias here.
- **`hybrid_union` immunity:** NNLS+ALIAS, alias passes `transition=None` → omega_stark=0; not gated solely by the R² gate. DB fixes will not move its F1.
- **ALIAS-NNLS-PLOCAL-3 mechanism (P_local/nnls collinearity):** refuted by faithful repro — not the recall driver.
- **spectral_nnls basis conditioning:** near-orthogonal (cond ~38, full rank) — the FP problem is the SNR detection rule, not ill-conditioning; the Voigt rewrite is NOT the load-bearing fix.

---

## 7. Open Empirical Questions

1. **Realized peak-ID lift from Waves 1+2.** Estimates (+0.05–0.12 alias macro-F1; spectral_nnls 0.44→~0.52–0.58) are bounded by reproductions on narrow synthetic spectra, not the real n=33 `UnifiedBenchmarkRunner` baseline. Must re-run the production harness after Wave 1/2 land. Will `hybrid_union` cross the 0.70 floor, or do the alias-arm gains get absorbed by the NNLS-weighted union?
2. **Composition-accuracy lift from Waves 3+4 on the manifold/Bayesian paths.** No end-to-end Aitchison/ILR number was measured (`compare_stark_vjbh.py` needs absent parquet artifacts). Build the two `composition_records.parquet` runs to quantify the ×20 Stark and 1.7–2.6× PF fixes against ground truth.
3. **Hot-edge partition-function ceiling.** Direct-sum itself is ~10% low at ≥12000 K (Cr II 75–84%, Ti I 91%) due to missing high-lying NIST levels. If ps-LIBS analyses routinely exceed ~10000 K, does the re-fit need synthetic Rydberg-level completion, or is the residual within composition tolerance?
4. **Temperature-estimator cold bias.** ALIAS est_T is 2.3× low on pure Fe (5863 vs 13774 K). How much of the R²-gate lift in Wave 1 is contingent on first fixing the unweighted-slope "dragged cold" estimator (`alias.py:2099`)?
5. **comb's place.** Is comb on the roadmap as a `hybrid_consensus` voter? If not, 5thd/s1qr.2 stay non-blocking; if yes, its 0.018 precision must be fixed (amplitude gate, not the denominator) before it can join without regressing `hybrid_union` precision (gate tolerance 0.02).
