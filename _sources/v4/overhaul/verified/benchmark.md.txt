# Adversarial Verification: Benchmark Package

**Verifier:** claude-sonnet-4-6 (adversarial pass)
**Date:** 2026-06-25
**Method:** ripgrep + Read; re-derived against literature in `scratchpad/overhaul/literature/`

---

## Finding 1 — Mole-fraction / mass-fraction mismatch

**Census severity:** critical
**REAL:** TRUE
**Confirmed severity:** critical

**Reasoning:** Independently verified all three cited locations.
`dataset.py:342-378` docstring states `true_composition` is "mass fractions (sum to 1.0)" and all
dataset adapters confirm this: `datasets/nist_steel.py` calls `as_mass_fractions()` (which divides
wt% by 100); `synthetic_corpus.py:566` stores `recipe.mass_fractions` as `true_composition`.
The iterative solver (`inversion/solve/iterative.py:109`) explicitly documents its output as
"Element concentrations (number/mole fractions, sum to 1)" and says to "Convert to mass fractions
via…". `composition_eval.py:_coerce_composition_prediction` normalises the solver output but
performs no molar-mass conversion; `_build_composition_success_record` passes the raw number
fractions as `concentrations` directly into `aitchison_distance(true_comp, concentrations)` at
line 266. There is no `to_mass_fractions` call anywhere in `cflibs/benchmark/`. The Völker (2024)
pitfall P5 in `literature/cflibs-method.md` explicitly quantifies errors up to 353% for this
omission in steel. The census claim is fully confirmed — every dataset uses mass fractions as
truth, every solver returns number fractions, and no conversion is ever applied.

---

## Finding 2 — LOD estimation uses 25th-percentile as blank proxy

**Census severity:** high
**REAL:** TRUE
**Confirmed severity:** high

**Reasoning:** `metrics.py:550-554` confirmed verbatim:
```python
low_mask = true_values < np.percentile(true_values, 25)
if low_mask.sum() >= 3:
    lod = 3 * np.std(residuals[low_mask])
```
The IUPAC definition (ISO 11843) of LOD = 3σ_blank requires the standard deviation of blank
(zero-concentration) measurements. When the minimum certified concentration in NIST steel or
USGS standards is 0.01–2 wt%, the "bottom 25th percentile" contains samples with ~0.01–0.5 wt%
analyte — not blanks. The resulting σ measures inversion residual scatter at low but nonzero
concentrations, which systematically overestimates σ_blank (because it includes composition
uncertainty, not just instrument noise). As a result, LOD is inflated and the `traces` stratum
gate misfires. No blank spectra concept exists in `BenchmarkSpectrum`. The census claim is
accurate.

---

## Finding 3 — Simplified fallback model omits partition function

**Census severity:** high
**REAL:** TRUE
**Confirmed severity:** high

**Reasoning:** `synthetic.py:604-607` confirmed:
```python
boltz = np.exp(-E_k / (KB_EV * temperature_K))
intensity = conc * rel_int * boltz * 1e4
```
The canonical intensity formula (`literature/cflibs-method.md` eq. 1.1, Ciucci 1999) is
`I ∝ (A g_k / U_s(T)) · exp(-E_k / k_B T)`. The partition function `U_s(T)` in the denominator
is entirely absent. For Fe I, `U(T)` grows from ~30 at 8000 K to ~90+ at 15000 K (roughly
3–10× over the LIBS temperature range), so the simplified model over-predicts high-temperature
intensity by this factor for each element independently — distorting relative element intensities
in a temperature-dependent way. The simplified path is triggered whenever `self.atomic_db is None`
(the module-level factory functions at lines 650, 696 default `atomic_db=None`) or as a
fallback when the full forward model raises (`_generate_with_forward_model` exception handler at
line 486 silently falls through to `_generate_simplified`). Any benchmark run that uses the
simplified model generates physically incorrect spectra where relative element intensities depend
on the wrong temperature scaling, causing the inversion (which is calibrated against real U_s(T)
behaviour) to systematically recover wrong compositions on these synthetic spectra. The census
claim is accurate.

---

## Finding 4 — McWhirter delta_E default 2.0 eV is not species-specific

**Census severity:** medium
**REAL:** TRUE (with nuance)
**Confirmed severity:** medium (no severity change; the nuance actually supports the finding)

**Reasoning:** `physical_consistency.py:463` confirms `default_delta_e_ev: float = 2.0`.
`_extract_inputs` at lines 348-350 does look for `annotations["delta_e_ev"]` and `record.delta_e_ev`
before falling back to the default — so a per-spectrum override path exists in principle. However,
a search of all inversion solvers (`rg "delta_e_ev" cflibs/inversion/`) returns zero matches other
than a false hit in a comment. No solver currently populates `delta_e_ev` in its annotation dict.
In practice, therefore, every McWhirter check in every benchmark run uses the constant 2.0 eV
default. Per McWhirter (1962), ΔE should be the resonance-line energy of the dominant emitting
species; the cubic (ΔE)³ scaling means a 2.0 eV default is ~2.6× too strict for Ca-dominated
plasmas (Ca resonance ~1.69 eV gives (1.69/2.0)³ = 0.60) and ~2.9× too loose for Si-dominated
plasmas (Si resonance ~4.15 eV gives (4.15/2.0)³ = 8.9). The override path's existence is noted
but does not mitigate the finding because it is never used. The census description is accurate.

---

## Finding 5 — Three-way duplication of composition scoring path

**Census severity:** medium
**REAL:** TRUE
**Confirmed severity:** medium

**Reasoning:** Confirmed all three independently active compute paths:
(a) `metrics.py:BenchmarkMetrics._calculate_element_metrics` — operates on numpy arrays of
multi-spectrum stacked predictions, computing RMSEP/MAE/R²/bias/LOD from scratch.
(b) `composition_eval.py:_build_composition_success_record` — per-spectrum path calling
`aitchison_distance`, `rmse_composition`, `per_element_error` from `composition_metrics.py`.
(c) `harness.py:_evaluate_one` — its own per-spectrum scoring loop (lines 379-382) also calling
`aitchison_distance`, `rmse_composition`, `per_element_error`. These are three genuinely parallel
implementations, not a single implementation re-exported. `harness.py` uses `BenchmarkCorpus`
(old corpus API) while `composition_eval.py` uses `BenchmarkSpectrum` (new API), and `metrics.py`
uses raw numpy arrays. The census claim is accurate: if Finding 1 (basis mismatch) is fixed,
the fix must be applied identically in all three paths. The duplication risk is real.

---

## Finding 6 — 8 inline CFLIBS_FF_* env-var flags

**Census severity:** medium
**REAL:** TRUE
**Confirmed severity:** medium

**Reasoning:** `synthetic_eval.py:536-543` confirmed verbatim — all 8 `os.environ.get(...)` calls
inside a closure, with magic string defaults. The `CalibrationOptions` dataclass exists at line 595
of the same file, demonstrating the team's preferred pattern for structured config. The 8 flags
control `ForwardFitIdentifier` construction parameters that have no structured container. This
is a code-quality/test-maintainability issue: it cannot be overridden via test fixtures (only via
environment mutation), preventing deterministic unit tests of the forward-fit evaluation path.
Census claim is accurate.

---

## Finding 7 — No integration test for physical-consistency gate through evaluate_composition_workflow

**Census severity:** low
**REAL:** TRUE
**Confirmed severity:** low

**Reasoning:** Standard test-gap observation. The census correctly notes that
`tests/benchmark/test_physical_consistency.py` tests `aggregate_physical_consistency` in
isolation. The integration path through `evaluate_composition_workflow` that would populate
`annotations` with real solver outputs and then fire the gate is not tested. No evidence of
such integration test was found in `tests/benchmark/`. Census claim is accurate.

---

## Additional Findings Not in Census

### A1 — Aitchison distance pads missing truth elements with epsilon, not zero

**Severity:** low
**Location:** `cflibs/benchmark/composition_metrics.py:93-103`

`aitchison_distance` takes the UNION of `c_true.keys()` and `c_pred.keys()`, padding absent
elements in either dict with `_EPSILON = 1e-12` via `_to_positive_array`. This means if the
inversion reports a false-positive element not present in `c_true` (e.g. returns "Zn" when the
sample has no Zn), the truth for Zn is set to 1e-12 rather than 0. A large false-positive
concentration (say 5% Zn) will yield a log-ratio of `ln(0.05 / 1e-12) ≈ 24.8`, which massively
inflates the Aitchison distance for that sample. This is arguably correct behaviour (a gross
false-positive IS a large error), but it is not documented and differs from the RMSE function
at line 252, which pads with 0.0 (not epsilon) — meaning RMSE is insensitive to false-positive
elements at low weight while Aitchison distance is hypersensitive to them. The asymmetry between
the two primary metrics is surprising and should be documented; a false-positive of any size
dominates Aitchison distance via the 1e-12 floor.

### A2 — Silent fallback from forward-model to simplified model loses physics correctness without warning

**Severity:** medium (overlaps Finding 3, distinct root)
**Location:** `cflibs/benchmark/synthetic.py:483-490`

```python
except Exception as exc:
    logger.warning(
        "Forward model failed (%s), using simplified model fallback",
        exc,
    )
    return self._generate_simplified(composition, temperature_K, rng)
```

The fallback from the DB-backed forward model to the partition-function-free simplified model
is triggered silently on ANY exception (including transient DB connection errors, missing species,
or numerical failures). A benchmark run that encounters, say, 5% forward-model failures would
silently mix physically correct spectra with physically incorrect ones, producing a corrupted
batch where the broken spectra are indistinguishable from valid ones in downstream results.
There is no `TruthType.SIMULATED_UNPHYSICAL` flag set on spectra generated via the fallback path,
so consumers cannot filter them out. The census identified the partition-function omission as
Finding 3 but did not flag the silent-corruption-under-failure aspect separately. The fix is to
set `truth_type=TruthType.SIMULATED_UNPHYSICAL` (or raise) when the fallback fires, so the
benchmark runner can exclude or warn on those spectra.

---

## Summary Table

| # | Title | REAL | Census sev | Confirmed sev |
|---|-------|------|-----------|---------------|
| 1 | Mole/mass fraction mismatch | TRUE | critical | critical |
| 2 | LOD 25th-percentile blank proxy | TRUE | high | high |
| 3 | Simplified model omits U_s(T) | TRUE | high | high |
| 4 | McWhirter delta_E not species-specific | TRUE | medium | medium |
| 5 | Three-way composition scoring duplication | TRUE | medium | medium |
| 6 | 8 inline CFLIBS_FF_* env-var flags | TRUE | medium | medium |
| 7 | No integration test for gate + eval loop | TRUE | low | low |
| A1 | Aitchison epsilon vs RMSE zero padding mismatch | NEW | — | low |
| A2 | Silent simplified-model fallback lacks TruthType flag | NEW | — | medium |

All 7 census findings confirmed. No false positives identified. Two additional findings discovered.
Highest confirmed severity: **critical** (Finding 1).
