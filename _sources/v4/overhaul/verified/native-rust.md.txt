# Adversarial Verification: native/cflibs-core (Rust)

**Verifier methodology:** Independent re-read of `native/cflibs-core/src/{partition.rs,comb_matching.rs,lib.rs}`,
`cflibs/plasma/partition.py`, `cflibs/inversion/identify/line_detection.py`, and `cflibs/jitpipe/host.py`
via ripgrep + targeted Read. Each finding was re-derived from first principles before verdict.

---

## Finding 1 — `batch_partition_functions` missing t_min/t_max clamp + g₀ floor

**REAL:** TRUE (the code is as described) — but severity is **OVERSTATED**.

**Corrected severity:** MEDIUM (was HIGH)

**Reasoning:** The census is correct that `partition.rs:48–59` evaluates the quartic-in-ln-T polynomial
without any clamp to `[t_min, t_max]` or floor at `g0`, and that the Python `polynomial_partition_function`
at `partition.py:929–939` does apply those guards when the kwargs are supplied. The 560× Ca I error at
100 000 K and the sub-unity Nb I value at 500 K are real if the function is called outside the fit
window without guards. However, the adversarial check reveals a critical mitigating fact the census
missed: `PartitionFunctionEvaluator.evaluate_batch` has **zero production callers** in `cflibs/`. The
method is defined at `partition.py:1106–1151` and appears only in `tests/test_comb_rust.py`. No module
under `cflibs/` invokes it. Furthermore, the NumPy fallback path inside `evaluate_batch` (line 1150)
also calls `polynomial_partition_function` without `t_min`/`t_max`/`g0`, so both the Rust and Python
paths of this function lack the guard — they are mutually consistent at the wrong behavior. The fix
recommendation remains valid, but the production impact is currently zero (the function is dead code in
the shipped pipeline). Severity is MEDIUM rather than HIGH: real latent hazard for anyone who wires
`evaluate_batch` into a new caller, but no current physics-correctness regression.

---

## Finding 2 — `kdet_filter_elements` dead under default `shift_coherence_veto=True` pipeline

**REAL:** TRUE

**Corrected severity:** HIGH (confirmed as stated)

**Reasoning:** Independent verification confirms the exact guard at `line_detection.py:2187`:
`if HAS_RUST_CORE and not shift_coherence:`. The `shift_coherence` parameter arrives from
`_apply_kdet_filter` line 1026 as `shift_coherence=shift_coherence_veto`, which defaults to `True` at
the `detect_lines_and_match_transitions` signature (line 1126). Under the default configuration,
`not shift_coherence` is `False`, so the Rust branch is never entered. The comment at
`cflibs/jitpipe/host.py:1148` explicitly states "the density-score branch
(`shift_coherence_veto=False`: Rust + density) is not ported and falls [back]", confirming this is an
intentional architectural gap, not a bug. The Rust `kdet_filter_elements` function is thus unreachable
in the default production pipeline. The census description is fully accurate.

---

## Finding 3 — No cross-backend parity test for `scan_comb_shifts` dispatch

**REAL:** TRUE

**Corrected severity:** MEDIUM (confirmed as stated, but with an important upgrade note)

**Reasoning:** `tests/test_comb_rust.py` calls `scan_comb_shifts` directly on synthetic data but does
NOT compare `_scan_comb_shifts_dispatch_rust` against pure-Python `_scan_comb_shifts` on the same
realistic input. This coverage gap is confirmed. However, the divergence risk is higher than the census
implies because of an undetected algorithmic difference (see Missed Finding A below): the greedy
matching orientation differs between Rust and Python, meaning the parity gap is not merely a
test-infrastructure problem but may be a real result divergence on edge-case inputs. MEDIUM severity
is correct but underestimates the urgency.

---

## Finding 4 — Parallelism inverted (parallel over shifts, serial over elements)

**REAL:** TRUE

**Corrected severity:** MEDIUM (confirmed as stated)

**Reasoning:** `comb_matching.rs:197–241` uses `shifts.par_iter()` to parallelize the outer shift loop
while iterating elements serially inside each task. With typical N_shifts ≈ 200–400 and N_elements
≈ 30–80 after kdet filtering, this layout produces fewer Rayon tasks with higher per-task cost than the
transposed layout would. The census's fix suggestion (flatten to `(shift, element)` pairs in a single
`par_iter()`) is algorithmically sound. Benefit depends on workload, so profiling before applying is
the right guidance. Confirmed.

---

## Finding 5 — `temp <= 1.0` guard silently accepts T < 0

**REAL:** TRUE

**Corrected severity:** LOW (confirmed as stated)

**Reasoning:** `partition.rs:48–49` returns `1.0` for any `temp <= 1.0`, including negative values.
The Python reference (`partition.py:929`) does the same: `if T_K <= 1.0: return 1.0 if g0 is None
else max(1.0, float(g0))`. Both paths agree, and LIBS plasmas are always >> 1 K in practice.
Consistent behavior with no production impact. LOW severity confirmed.

---

## Finding 6 — No end-to-end test of `_apply_kdet_filter` Rust dispatch branch

**REAL:** TRUE

**Corrected severity:** MEDIUM (confirmed as stated)

**Reasoning:** The Rust kdet branch is guarded by `HAS_RUST_CORE and not shift_coherence`
(line 2187). `tests/test_comb_rust.py` calls `kdet_filter_elements` directly, bypassing the
`_apply_kdet_filter` dispatch logic entirely. If the argument mapping or return-type conversion in
`_kdet_dispatch_rust` were broken, the test suite would silently fall back to Python and not report a
failure. The coverage gap is real.

---

## Finding 7 — Partition parity tests use unclamped Python — cannot catch Finding 1

**REAL:** TRUE

**Corrected severity:** MEDIUM (confirmed as stated)

**Reasoning:** `test_comb_rust.py:276` compares `batch_partition_functions` against
`polynomial_partition_function(float(temp), list(coefficients[0]))` with no `t_min`/`t_max`/`g0`
arguments. Both paths therefore extrapolate identically, so the test confirms raw polynomial parity
but cannot detect the clamping divergence described in Finding 1. The test at T=50 000 K for Ca I
coefficients would pass even though both paths are 560× wrong. Confirmed.

---

## Missed Findings

### Missed Finding A — ARCHITECTURE / MEDIUM

**Title:** Greedy matching orientation is TRANSPOSED between Rust and Python `scan_comb_shifts`

**Location:** `native/cflibs-core/src/comb_matching.rs:14–52` (`count_matches`) vs
`cflibs/inversion/identify/line_detection.py:1921–1989` (`_match_transitions_to_peaks` as called by
`_score_comb_for_element`)

**Evidence:**
- Rust `count_matches` (lines 14–52): outer loop over **peaks**, pick nearest unused **transition** for
  each peak.
- Python `_match_transitions_to_peaks` (lines 1950–1988, called with `used_peaks=None`): outer loop
  over **transitions**, pick nearest unused **peak** for each transition.

**Consequence:** These are transposes of the same greedy bipartite-matching approximation. On inputs
where the bipartite graph is a perfect matching (one-to-one correspondence, no ambiguity), both
produce the same count. But when multiple peaks cluster near a single transition wavelength, or when
multiple transitions fall within tolerance of one peak, the two algorithms make different greedy
choices and can produce different `matched_lines` counts — and therefore different `precision`,
`recall`, `f1_score`, and `passes` flags for the same element at the same shift. Since
`_scan_comb_shifts` uses the Rust path as the primary path when `HAS_RUST_CORE` is True and the Python
path only as a fallback, any such divergence is silent in production. The existing parity tests
(Finding 3) use only clean 1:1 synthetic data and would not catch this.

**Fix:** Add a parity test with a deliberately ambiguous input — e.g. two peaks near one transition and
one peak near two transitions — and assert that Rust and Python produce identical `matched_lines`. If
they diverge, align the greedy orientation (prefer transition-outer to match the Python semantics,
which was the original design).

### Missed Finding B — ARCHITECTURE / LOW

**Title:** F1 tie-breaking epsilon differs between Rust (`1e-12`) and Python (`np.isclose` rtol=1e-5)

**Location:** `native/cflibs-core/src/comb_matching.rs:267` vs
`cflibs/inversion/identify/line_detection.py:1845`

**Evidence:**
- Rust: `(sr.total_f1 - prev.total_f1).abs() < 1e-12` for the "treat as equal" condition.
- Python: `np.isclose(total_f1, prev_f1)` which uses default rtol=1e-5, atol=1e-8.

**Consequence:** When two shifts produce F1 values that differ by, say, 3e-7 (well within `np.isclose`
tolerance but above 1e-12), the Python path treats them as tied and tiebreaks by match count, while
the Rust path picks the strictly higher F1 directly. This can cause different `best_shift` selections
in borderline cases. LOW severity because the absolute F1 difference in a real tie scenario is
typically sub-floating-point noise, but it is a semantic divergence the parity tests do not cover.

---

## Corrected Severity Summary

| # | Dimension | Original Sev | Confirmed? | Corrected Sev | Notes |
|---|-----------|-------------|-----------|--------------|-------|
| 1 | Physics-correctness | HIGH | TRUE | **MEDIUM** | `evaluate_batch` has no production callers |
| 2 | Architecture | HIGH | TRUE | **HIGH** | kdet Rust branch dead under default config |
| 3 | Architecture | MEDIUM | TRUE | **MEDIUM** | Urgency understated (see Missed A) |
| 4 | Performance | MEDIUM | TRUE | **MEDIUM** | Confirmed |
| 5 | Complexity | LOW | TRUE | **LOW** | Consistent behavior, no prod impact |
| 6 | Test-gaps | MEDIUM | TRUE | **MEDIUM** | Confirmed |
| 7 | Test-gaps | MEDIUM | TRUE | **MEDIUM** | Confirmed |
| A | Architecture (MISSED) | — | NEW | **MEDIUM** | Transposed greedy matching orientation |
| B | Architecture (MISSED) | — | NEW | **LOW** | F1 tie-breaking epsilon mismatch |

**Highest confirmed severity:** HIGH (Finding 2: kdet_filter_elements dead under default config)
