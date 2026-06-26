# Adversarial Verification: `cflibs/evolution` Census Findings

Verified against `/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5` using rg + Read + live Python execution.

---

## Finding 1.1 — HIGH: `exec`/`eval`/`compile` bypass blocklist

**REAL: TRUE** | **Corrected severity: HIGH** (unchanged)

Live execution confirms: `scan_source('exec("import torch")')` returns `[]`, `scan_source('eval("import torch")')` returns `[]`, `scan_source('compile("import torch", "", "exec")')` returns `[]`. The `_scan_call` function at `evaluator.py:148-168` dispatches to `DYNAMIC_IMPORT_CALLS` membership check, which only contains `("__import__", "importlib.import_module", "importlib.__import__")`. `exec` and `eval` resolve to `ast.Call` with `ast.Name(id="exec")` / `ast.Name(id="eval")` — the same node type that catches `__import__` — but they simply are not in the registry. The bypass is real and exploitable by any LLM-generated candidate that wraps forbidden imports in `exec()`/`eval()`. The `__import__` case is correctly caught (verified: returns one `dynamic_import` violation). No other scanner mechanism would catch this. HIGH severity confirmed.

---

## Finding 2.1 — HIGH: `enforcement_mode="warn"` is a config fiction

**REAL: TRUE** | **Corrected severity: HIGH** (unchanged)

`EvolutionDriverConfig` is only referenced in `cflibs/evolution/config.py` and `tests/evolution/test_config.py` — confirmed by `find ... | xargs grep -l "EvolutionDriverConfig"`. No driver, no caller, no consumer of the `enforcement_mode` field anywhere in the codebase outside of tests. The config.py docstring (lines 44-48) explicitly says the field is "consumed by the evolution driver, not by `evaluator.scan_source` / `evaluator.assert_physics_only`" — but no such driver exists. The field is validated (line 87-89) and immutable (frozen dataclass), so a future external driver that naively reads `enforcement_mode == "warn"` and skips `assert_physics_only` would silently allow ML-violating candidates through the physics evaluator. HIGH confirmed: this is a security footgun with no implementation.

---

## Finding 5.1 — HIGH: No tests for `exec`/`eval`/`compile` evasion

**REAL: TRUE** | **Corrected severity: HIGH** (unchanged)

`tests/evolution/test_blocklist.py` contains tests for `__import__` (`test_dunder_import_rejected_for_forbidden_module`) and `importlib.import_module` (`test_importlib_import_module_rejected`), and a registry membership smoke test (`test_dynamic_import_calls_registry_covers_common_forms`). There is no test for `exec(...)`, `eval(...)`, or `compile(...)` evasion. The test file was read in full; grep confirms no `exec` or `eval` test. Ties directly to Finding 1.1: the gap is both real in the scanner and unprotected by regression tests.

---

## Finding 1.2 — MEDIUM: `__init__.py` vs. CLAUDE.md ML-allowance contradiction

**REAL: TRUE** | **Corrected severity: MEDIUM** (unchanged; documentation-only, no runtime impact)

`cflibs/evolution/__init__.py` line 8: "modules here must not import any ML / neural-network library." CLAUDE.md "Physics-Only Constraint" section says "Machine learning is allowed **only** in `cflibs/evolution/`." The Ruff `pyproject.toml` (lines 206-218) enforces TID251 across all of `cflibs/` with no per-file-ignores for `cflibs/evolution/` (the only TID251 exemption is `scripts/archive/hpc-campaign/train_ml_classifier.py`). So the `__init__.py` is accurate: ML is banned inside `cflibs/evolution/` by both the docstring and the static linter. CLAUDE.md's claim is misleading: ML is permitted in the *external* LLM-driven optimization driver that calls into `cflibs/evolution/`, not in `cflibs/evolution/` itself. The census characterization is correct; documentation discrepancy confirmed.

---

## Finding 2.2 — MEDIUM: `fitness_weights` lacks normalisation contract

**REAL: TRUE** | **Corrected severity: MEDIUM** (unchanged; speculative/design debt, no runtime impact without a driver)

`config.py:55-63` defines `fitness_weights` as a plain mapping of per-dataset floats with no aggregation formula. The comment says "per-dataset fitness weights" but documents no `aggregate_fitness` function or formula. No such function exists in `evaluator.py` or `config.py`. The `overfitting_penalty` field (line 67) references a variance penalty but without a formula. Since no driver exists, this is currently harmless, but the absence of a specified aggregation contract means any future driver will need to invent one. Severity MEDIUM correct (design debt, not a runtime bug in existing code).

---

## Finding 2.3 — MEDIUM: `assert_benchmark_relevance` misses binary/mode-change hunks

**REAL: TRUE** | **Corrected severity: MEDIUM** (unchanged)

Live test confirms: binary diff `"Binary files a/cflibs/data.bin and b/cflibs/data.bin differ"` with `exercised={"cflibs/data.bin"}` raises `RuntimeError` (false rejection) because the `touched` set is empty — the parser only matches lines starting with `"--- a/"` or `"+++ b/"` (evaluator.py:253-257). The new-file case (`--- /dev/null` / `+++ b/new.py`) is NOT broken: the `+++ b/` branch fires for the new-file hunk and the exercised path is found. The mode-change-only case (no `---/+++` lines at all) would also produce a false rejection. Binary-diff false rejection is confirmed real. Census finding is accurate. Note: the census also states "no bug" for renames — this is correct, since either the old or new path matching exercised_files is sufficient.

---

## Finding 4.1 — MEDIUM: `evaluation_timeout_s=5.0` unrealistically short

**REAL: TRUE** | **Corrected severity: MEDIUM** (unchanged; config default issue)

`config.py:38`: `evaluation_timeout_s: float = 5.0`. Per project memory `reference_inversion_hotspot_profile`, RANSAC wavelength calibration alone accounts for ~73% of wall-clock on a reference inversion. A full benchmark run on real spectra (aalto, chemcam, supercam matrices) would substantially exceed 5 seconds per candidate. This default, if taken literally by a driver, would cause every non-trivial evaluation to time out with `fitness=-inf` from timeout rather than actual physics failure. MEDIUM severity is appropriate (misconfigured default, not a security issue; no driver currently reads it).

---

## Finding 4.2 — LOW: `structural_mutation_cadence=10` magic number undocumented

**REAL: TRUE** | **Corrected severity: LOW** (unchanged; documentation gap)

`config.py:41-43` sets `structural_mutation_cadence: int = 10` with a comment saying "Set to a large number to effectively disable" but no citation or rationale for K=10. In NES/CMA-ES literature, structural mutations are typically infrequent (1:50 to 1:100 relative to parameter mutations). K=10 means 10% of batches trigger structural mutation, which is aggressive. Finding is technically real but low impact; no driver exists to trigger it.

---

## Finding 5.2 — MEDIUM: `assert_benchmark_relevance` tests incomplete; straggler file

**REAL: TRUE** | **Corrected severity: MEDIUM** (unchanged)

`tests/test_evolution_evaluator.py` contains only 3 tests covering the normal unified-diff path. Confirmed: no binary-diff test, no mode-change test, no `pytestmark = pytest.mark.unit`. The file is at repo-root `tests/` rather than `tests/evolution/`. The missing binary-diff test would catch the bug in Finding 2.3. Finding accurate.

---

## Finding 5.3 — LOW: `enforcement_mode="warn"` has no behaviour test

**REAL: TRUE** | **Corrected severity: LOW** (unchanged; low impact since no driver exists)

`tests/evolution/test_config.py` tests that `enforcement_mode="off"` raises `ValueError` but does not test what `"warn"` *does*. Since no driver implements warn-mode behaviour, this is a consequence of Finding 2.1 rather than an independent test gap. Severity LOW is correct.

---

## Additional Findings Not in Census

### NEW-A — LOW: `_scan_call` does not flag `exec`/`eval` called as attribute methods

While `exec("import torch")` at module level is an `ast.Call` with `ast.Name(id="exec")`, if a hypothetical candidate calls `builtins.exec(...)`, it would appear as an `ast.Attribute` chain `builtins.exec` — not in `DYNAMIC_IMPORT_CALLS`. Since `builtins` is not a forbidden prefix, `_scan_attribute` would not catch it either. This is a narrower variant of Finding 1.1 (the main finding already covers the common case). Severity LOW (requires deliberate obfuscation beyond the simple `exec(...)` case).

### NEW-B — LOW: No test for `SyntaxError` propagation in `assert_physics_only`

`assert_physics_only` docstring (evaluator.py:228-232) says "malformed candidate code surfaces as `SyntaxError`, not `BlocklistViolationError`" and instructs callers to catch both. The `__main__.py` correctly catches `SyntaxError` (line 71). However, `tests/evolution/test_blocklist.py` has no test that passes a syntactically invalid source and confirms `SyntaxError` is raised rather than being swallowed. Minor test gap; severity LOW.

---

## Summary

| Finding | Real? | Confirmed Severity | Notes |
|---------|-------|--------------------|-------|
| 1.1 exec/eval bypass | TRUE | **HIGH** | Live-executed; returns [] for exec/eval/compile |
| 1.2 CLAUDE.md contradiction | TRUE | MEDIUM | Ruff config confirms cflibs/evolution/ has no TID251 exemption |
| 2.1 enforcement_mode fiction | TRUE | **HIGH** | No callers of EvolutionDriverConfig outside tests |
| 2.2 fitness_weights no formula | TRUE | MEDIUM | No aggregate_fitness function anywhere |
| 2.3 binary diff false rejection | TRUE | MEDIUM | Live-executed; binary diff raises false RuntimeError |
| 4.1 evaluation_timeout_s=5.0 | TRUE | MEDIUM | Contradicts known inversion wall-clock |
| 4.2 structural_mutation_cadence | TRUE | LOW | Documentation gap only |
| 5.1 no exec/eval tests | TRUE | **HIGH** | test_blocklist.py confirmed; zero exec/eval test cases |
| 5.2 straggler test file | TRUE | MEDIUM | No pytestmark, no binary test |
| 5.3 warn mode no behaviour test | TRUE | LOW | Consequence of 2.1 |
| NEW-A builtins.exec attribute | NEW | LOW | Narrow obfuscation variant |
| NEW-B SyntaxError not tested | NEW | LOW | Missing test for documented behaviour |

**Highest confirmed severity: HIGH** (Findings 1.1, 2.1, 5.1)

All census findings for severity HIGH and MEDIUM were verified as TRUE. No finding was found to be FALSE. The two prior false-positive findings (dead class, Saha ionization-potential guard) do not apply here — this package has no physics equations and no complex call graphs to misread.
