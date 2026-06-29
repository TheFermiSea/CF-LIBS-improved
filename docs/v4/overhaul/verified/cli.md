# CLI Adversarial Verification (`cflibs/cli/main.py`)

Verifier: adversarial sub-agent  
Date: 2026-06-25  
Source census: scratchpad/overhaul/census/cli.md  
Code verified at: `/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5/cflibs/cli/main.py`  
Support files read: `cflibs/plasma/lte_validator.py`, `cflibs/inversion/solve/iterative.py`, `cflibs/benchmark/scoreboard.py`  
Literature: `saha-boltzmann-lte.md`  
Project memory: `reference_mcwhirter_delta_e_physics.md`, `project_v4_m7_refuse_to_report.md`

---

## Verified Findings

### F1 · McWhirter ΔE defaults to max(E_k) — physics wrong, gated off by default

**REAL: TRUE**  
**Confirmed severity: HIGH**

Verified in `cflibs/plasma/lte_validator.py::_delta_e_from_observations` (lines 365–397): the default path computes `delta_E_eV = float(max(energies))` where `energies = [o.E_k_ev for o in observations]`, i.e. the highest observed upper level. The physics-correct alternative (`_mcwhirter_delta_e_resonance` in `iterative.py` lines 2011–2069) selects the upper-level energy of the strongest (max A_ki) `is_resonance` line per element, capped at the max over species present. That path is only activated when `CFLIBS_MCWHIRTER_RESONANCE_DE` env var is set to `"1"/"true"/"yes"/"on"` (`iterative.py` lines 2161–2166). The docstring in `_delta_e_from_observations` acknowledges the limitation: it explicitly states that `max(E_k)` "bounds the resonance-to-upper-level transition and is far larger than the adjacent-gap value" and that using the adjacent gap "badly under-estimates the required density" — but neither interpretation aligns with Cristoforetti et al. (2010): the correct quantity is the *first resonance transition energy* (ground → first dipole-allowed excited state), which for Fe I is ~2.48 eV vs max(E_k) ≈ 7.5 eV. The cubic dependence means the n_e floor is overestimated by ~(7.5/2.48)³ ≈ 27×, causing false-reject of valid LTE plasmas. Project memory `reference_mcwhirter_delta_e_physics.md` independently confirms this via a real observed case (n_e=2.7e17 cm⁻³ falsely flagged non-LTE; resonance path gives correct PASS). The correct behavior exists and is DB-validated; it is simply off by default.

---

### F2 · `lte_n_e_ratio` warning missing direction context

**REAL: TRUE**  
**Confirmed severity: MEDIUM** (downgraded from HIGH-adjacent: cosmetic/UX, not physics-wrong)

Verified at `main.py` lines 299–304: when `lte_ok` is False, the warning prints `f"(n_e ratio = {qm.get('lte_n_e_ratio', 0):.2f})"`. The ratio semantics (actual/required) are not explained; a ratio of 0.3 means the plasma is 3× below the LTE threshold, but the user sees only "0.30" with no label. The `lte_n_e_required_cm3` field is stored in `qm` by `lte_validator.py` line 87 (`"lte_n_e_required_cm3": self.mcwhirter.n_e_required`) but is never surfaced in the CLI. The fix described in the census (expand the message to show actual and required n_e in cm⁻³) is actionable and correct. Severity remains MEDIUM — functionally silent but cosmetic/diagnostic gap.

---

### F3 · `--elements` parses as `nargs="+"` (invert) vs comma-string (analyze/batch/bayesian) — public API inconsistency

**REAL: TRUE**  
**Confirmed severity: HIGH**

Verified directly:
- `invert_parser` at line 1116–1122: `add_argument("--elements", type=str, nargs="+", default=None)` → list of space-separated tokens.
- `analyze_parser` at line 1161–1163: `add_argument("--elements", type=str, required=True, help="Comma-separated element list, e.g. Fe,Si,Ca")` → single string.
- `bayesian_parser` at line 1212–1213: `add_argument("--elements", type=str, required=True, help="Comma-separated element list")` → single string.
- `batch_parser` at line 1262–1263: `add_argument("--elements", type=str, required=True, help="Comma-separated element list")` → single string.

`analyze_cmd` (line 482), `batch_cmd` (line 770), and `bayesian_cmd` (line 644) all call `.split(",")` on `args.elements`. `_resolve_invert_elements` (lines 199–211) handles list or bare string but does NOT split comma-joined strings: if a user passes `--elements Fe,Cu` to `invert`, argparse with `nargs="+"` produces `["Fe,Cu"]` (a single-element list), which is not a `str`, so the `isinstance(elements, str)` guard at line 209 is bypassed, and `["Fe,Cu"]` is returned as a single-element list — silently failing to find element "Fe,Cu" in the database. Conversely, `--elements Fe Cu` to `analyze` triggers argparse error "unrecognized arguments: Cu". The inconsistency is real, user-visible, and scrpting-hostile.

---

### F4 · Eager scoreboard import fires on every CLI invocation

**REAL: TRUE**  
**Confirmed severity: MEDIUM**

Verified at `main.py` line 1328: `from cflibs.benchmark.scoreboard import DEFAULT_SEED as _SCOREBOARD_DEFAULT_SEED` is inside `main()` at parser-setup time, not inside `scoreboard_cmd`. Every `cflibs forward ...` or `cflibs invert ...` invocation pays this cost. Inspecting `cflibs/benchmark/scoreboard.py` top-level imports (lines 56–75): it imports `numpy`, `cflibs.inversion.pipeline.run_pipeline`, `cflibs.inversion.pipeline.build_pipeline_config`, and `cflibs.benchmark.synthetic_eval.compute_binary_metrics` at module level — all heavyweight. This contrasts with the established pattern throughout `main.py` where every other heavy import is deferred to its command handler (e.g., `forward_model_cmd` lines 117–122, `analyze_cmd` lines 475–476, etc.).

---

### F5 · batch CSV rows expose composition for refuse-to-report-failing spectra

**REAL: TRUE**  
**Confirmed severity: MEDIUM**

Verified at `_batch_row` (lines 688–706): the function always includes `**{f"C_{el}": result.concentrations.get(el, 0.0) for el in elements}` regardless of `_refuse_to_report_enabled()` or `result.overall_reliable`. The batch loop (lines 792–799) calls `_trust_report` for logging only; the resulting `warning_lines` are sent to the logger but do not suppress concentration values in the CSV row. When `CFLIBS_REFUSE_TO_REPORT=1`, a spectrum that fails the LTE/quality gate logs a "RESULT UNRELIABLE" warning but still writes its composition values to the CSV column.

---

### F6 · `_trust_json` recomputes `_trust_report` already computed by the caller

**REAL: TRUE**  
**Confirmed severity: LOW**

Verified at lines 515 and 528: `_output_analyze_result` calls `_trust_report(result, diagnostics)` at line 515 to get `(info_lines, warning_lines)`, then calls `_trust_json(result, diagnostics)` at line 528 for the JSON `"trust"` block. `_trust_json` (lines 347–362) calls `_trust_report` internally at line 350. On the JSON format path, `_trust_report` runs twice. The double computation includes two `os.environ.get("CFLIBS_REFUSE_TO_REPORT")` reads and two full quality-metrics walks. This is a minor performance inefficiency and a latent divergence risk if `_trust_report` gains side effects.

---

### F7 · Forward-model stdout silently decimates spectrum 10× without warning

**REAL: TRUE**  
**Confirmed severity: MEDIUM**

Verified at `main.py` line 193: `for wl, intensity in zip(wavelength[::10], intensity[::10])`. No WARNING, no print informing the user. The CSV path (lines 182–188) uses `np.savetxt` over the full arrays. The asymmetry is silent and user-invisible. For a default 0.01 nm grid from 200–800 nm (60,001 points), stdout yields 6,001 points. Note: the loop variable `intensity` at line 193 also shadows the outer `intensity` array from line 175 — this is harmless for the current loop body (the inner `intensity` is used only in that iteration) but is an unnecessary code smell that would bite any future edit that references `intensity` after the loop but before reassignment.

---

### F8 · `CFLIBS_REFUSE_TO_REPORT` and `CFLIBS_MCWHIRTER_RESONANCE_DE` vestigial off-default flags

**REAL: TRUE**  
**Confirmed severity: MEDIUM**

Verified: both flags are off-by-default env vars guarding physics-correct/quality-correct behavior. `_refuse_to_report_enabled()` at lines 222–235 and the `CFLIBS_MCWHIRTER_RESONANCE_DE` check at `iterative.py` lines 2161–2166. The project memory and docstrings both acknowledge these are non-regression scaffolds intended for eventual default-on promotion. The census characterization is accurate; this is a moderate flag-debt issue, not a blocker. Severity MEDIUM is appropriate.

---

### F9 · `manifold_cmd` progress callback: ZeroDivisionError for small grids + dead conditional branch

**REAL: TRUE**  
**Confirmed severity: LOW**

Verified at lines 935–939:
```python
def progress(completed, total, percentage):
    if args.progress or completed % (total // 10) == 0:
        print(...)
generator.generate_manifold(progress_callback=progress if args.progress else None)
```
(1) If `total < 10`, `total // 10 == 0`, and the first callback call with `completed > 0` raises `ZeroDivisionError`. This is a real crash path for any small manifold grid (e.g. a smoke-test with 5 parameter points).
(2) The callback is only passed when `args.progress is True` (line 939). Therefore when the callback fires, `args.progress` is always `True`, making the `args.progress or` branch in the `if` condition trivially True and the `completed % (total // 10) == 0` branch unreachable dead code. Both issues confirmed.

---

### F10 · Missing tests for forward-cmd stdout parity, element-parsing divergence, batch refuse-to-report, manifold progress ZeroDivisionError

**REAL: TRUE**  
**Confirmed severity: MEDIUM**

Not directly re-verified by code reading (this is a test-gap finding), but the census evidence is credible given the other confirmed findings (F7, F3, F5, F9 are real bugs with no existing test coverage preventing them).

---

## Additional Findings (Missed by Census)

### MF1 · `analyze_cmd` CSV and table modes also expose concentration for refuse-to-report failures (extension of F5)

**Severity: MEDIUM**

The census flagged F5 only for `batch_cmd`, but the same gap exists in `analyze_cmd`. The `_output_analyze_result` function's CSV format path (lines 531–537) always prints concentration values and sends warnings to stderr — it does not gate on `_refuse_to_report_enabled()` or `result.overall_reliable`. The table format (lines 538–551) similarly always prints concentrations. `invert_cmd`'s stdout path (lines 391–396) also always prints concentrations regardless of reliability. The "composition withheld" guarantee of `CFLIBS_REFUSE_TO_REPORT=1` is only enforced at the text-level warning ("composition withheld as non-quantitative") added by `_trust_report`; the actual concentration values are still emitted in every output path. F5 describes the batch case; this is the same contract breach in the single-spectrum paths and should be fixed together.

### MF2 · `_resolve_invert_elements` handles bare string but not comma-joined string from `invert --elements Fe,Cu`

**Severity: LOW** (sub-case of F3, but with a distinct failure mode)

`_resolve_invert_elements` at lines 209–210 converts a bare `str` to `[str]` but does not split comma-joined strings. When `invert` is called as `cflibs invert s.csv --elements Fe,Cu`, argparse `nargs="+"` gives `["Fe,Cu"]`, a single-element list of a comma-joined string, which bypasses the `isinstance(elements, str)` guard. The returned list is `["Fe,Cu"]` — a single element with a literal comma in its name — which silently fails to match any database entry. The fix (add a comma-split pass in `_resolve_invert_elements`) is distinct from the full standardization fix for F3 and should be applied as a defensive immediate patch regardless of whether the API is harmonized.

---

## Summary Table

| ID   | Census Sev | Confirmed? | True Sev | Notes |
|------|-----------|------------|----------|-------|
| F1   | HIGH      | TRUE       | HIGH     | `max(E_k)` default confirmed via lte_validator.py; resonance path gated off |
| F2   | MEDIUM    | TRUE       | MEDIUM   | Warning string confirmed; `lte_n_e_required_cm3` available but unused |
| F3   | HIGH      | TRUE       | HIGH     | `nargs="+"` vs `.split(",")` confirmed in all four parsers |
| F4   | MEDIUM    | TRUE       | MEDIUM   | Scoreboard import at line 1328 inside `main()` not `scoreboard_cmd`; heavyweight |
| F5   | MEDIUM    | TRUE       | MEDIUM   | `_batch_row` always writes concentrations; `_trust_report` only logs |
| F6   | LOW       | TRUE       | LOW      | Double `_trust_report` call on JSON path confirmed |
| F7   | MEDIUM    | TRUE       | MEDIUM   | `[::10]` confirmed; no user notification; variable shadowing also present |
| F8   | MEDIUM    | TRUE       | MEDIUM   | Both flags are off-default guards for correct behavior |
| F9   | LOW       | TRUE       | LOW      | ZeroDivisionError for `total<10` confirmed; dead branch confirmed |
| F10  | MEDIUM    | TRUE       | MEDIUM   | Test gaps credibly absent given confirmed real bugs |
| MF1  | (missed)  | NEW        | MEDIUM   | Same refuse-to-report composition-leak in analyze_cmd + invert_cmd |
| MF2  | (missed)  | NEW        | LOW      | Comma-joined string silently passes `_resolve_invert_elements` as "Fe,Cu" element |

**Highest confirmed severity: HIGH** (F1 physics correctness, F3 public API inconsistency)

All 10 census findings confirmed TRUE. No false positives found. 2 additional findings identified.
