# cflibs/io — Adversarial Verification Report

**Verifier:** independent re-read of source at `.worktrees/v4-m5/cflibs/io/` and `.worktrees/v4-m5/cflibs/cli/main.py` using ripgrep + Read tool.  
**Highest confirmed severity:** medium (no critical/high findings confirmed).

---

## F1 — `log_ne` exported without stating base; CSV/HDF5 label is ambiguous

**REAL: TRUE**  
**Corrected severity: medium** (unchanged)

Confirmed by reading `exporters.py:435,539` (CSV) and lines 817–829 (HDF5 summary attrs). The `MCMCResult` dataclass declares `log_ne_mean: float` as a proper field (confirmed in `results.py:45`), and the property at `results.py:66–68` (`n_e_mean`) returning `10.0**self.log_ne_mean` confirms the field stores log₁₀(n_e [cm⁻³]). The CSV column header is the bare string `"log_ne"` with no base or unit annotation; the HDF5 summary group stores attribute keys `"log_ne_mean"`, `"log_ne_std"`, etc. with no metadata indicating log₁₀ vs ln. A downstream reader has no machine-readable way to distinguish the two. The finding is correctly identified and the fix (rename to `log10_ne_cm3` + add `ne_cm3` derived value) is sound.

---

## F2 — Concentration export has no mole-fraction unit label

**REAL: TRUE**  
**Corrected severity: medium** (unchanged)

Confirmed by reading `exporters.py:379` (CSV header: `"element", "concentration", "uncertainty"`) and `exporters.py:788–799` (HDF5 `conc_group` creates `"values"` dataset with no attrs). The CF-LIBS closure equation produces number (mole) fractions; nothing in the export path annotates this. Benchmark adapters that read these files (e.g., `cflibs/benchmark/adapters_core.py`) consume the raw float without a basis check. The finding is valid; severity is medium because the value range [0,1] with Σ=1 is implicit, but cross-instrument or cross-paper comparisons require an explicit basis label.

---

## F3 — Comment says "lazy import" but imports are top-level and eager

**REAL: TRUE**  
**Corrected severity: medium** (unchanged)

Confirmed by reading `exporters.py:40–42`: the comment reads "lazy to avoid circular imports" but the three imports are unconditional top-level statements, not guarded by `if TYPE_CHECKING`. A grep for callers of `cflibs.io` in `cflibs/inversion/` returns zero results today, confirming no actual circular import currently exists; however, the imports are still eager and the claimed intent is wrong. The `ExportData` type alias at line 55 uses the imported names at module definition time, so the fix requires `TYPE_CHECKING` guard + string-quoted `Union` or `TypeAlias`. The finding is architecturally valid though the risk is latent rather than active.

---

## F4 — Duck-typed `_is_*` dispatch; ordering load-bearing, fragile for new types

**REAL: TRUE**  
**Corrected severity: medium** (unchanged)

Confirmed at `exporters.py:224–238` (helpers) and `exporters.py:325–336` (CSV dispatch chain), `exporters.py:741–748` (HDF5 dispatch). The comment at line 325–326 explicitly documents that order matters: nested-sampling is checked before MCMC "since nested results may contain MCMC-like fields." This is a real ordering dependency. The existing `TwoZoneMCMCResult` in `results.py` (confirmed by `rg` across the solve package) would need to be inserted in the right place in both the CSV and HDF5 chains to be supported. The sentinel-field fix proposed in the census is sound.

---

## F5 — `forward` CLI bypasses `save_spectrum`; round-trip from forward→invert broken

**REAL: FALSE**  
**Corrected severity: none**

The census claims "`intensity_W_m2_nm_sr` is not in the `_load_spectrum_csv` alias list." Reading `spectrum.py:40–52` directly, the alias list is:

```python
["intensity", "intensity_W_m2_nm_sr", "I", "counts", "signal", "spectrum", "flux"]
```

`intensity_W_m2_nm_sr` is at position 2 in the list (line 44). The round-trip `cflibs forward --output x.csv && cflibs invert x.csv` does NOT fail. The CLI uses `np.savetxt` with `header="wavelength_nm,intensity_W_m2_nm_sr"` and that column name IS handled by `_load_spectrum_csv`. The census finding is factually wrong about the alias gap. The architectural observation (two different serialization paths for the same format) is a valid style concern, but the claimed functional bug does not exist.

---

## F6 — `CSVExporter._export_spectrum` Python per-row loop; use `np.savetxt`

**REAL: TRUE**  
**Corrected severity: medium** (unchanged)

Confirmed at `exporters.py:593–599`. The inner loop iterates per wavelength point, per column, using Python string formatting (`self.float_format % val`). For a manifold-scale spectrum (tens of thousands of points), this is materially slower than `np.savetxt`. The `save_spectrum()` function in `spectrum.py:120–126` correctly uses `np.savetxt` for the same operation, proving the pattern is known in the codebase. The proposed `np.column_stack + StringIO + np.savetxt` fix is valid.

---

## F7 — HDF5 `dtype="S10"` silently truncates element names > 10 bytes

**REAL: TRUE**  
**Corrected severity: medium** (unchanged)

Confirmed at `exporters.py:790, 864, 915` (three concentration `elements` datasets with `dtype="S10"`) vs `exporters.py:881` (diagnostics `parameters` dataset with `dtype="S20"`). NumPy's fixed-width byte-string dtype silently truncates without warning. Standard LIBS species identifiers (e.g., `"Fe"`, `"Fe_II"`, `"Ca_I"`) are under 10 bytes, but names like `"Manganese_II"` (12 bytes) or future compound identifiers would be silently corrupted. The inconsistency with `"S20"` for the adjacent diagnostics dataset is also a maintenance trap. Using `h5py.string_dtype()` or `dtype=object` is the correct fix.

---

## F8 — `JSONExporter` allow_nan asymmetry: NaN → string, Inf → invalid JSON token

**REAL: PARTIALLY FALSE**  
**Corrected severity: low** (unchanged, but mechanism is wrong)

The census claims `_float_to_json` returns raw `float('inf')` for Infinity, relying on `json.dump(allow_nan=True)` to write the bare token. Reading `exporters.py:1066–1072`, both paths return Python strings: `"NaN"` for NaN and `"Infinity"` / `"-Infinity"` for Inf. Neither path returns a raw Python float Infinity. The census's described asymmetry ("NaN → string, Inf → invalid JSON token") does NOT exist in the code.

The real (minor) bug is different: the `allow_nan` flag controls NaN behavior (returns `"NaN"` or `None`) but does NOT apply to Inf (always returns the string `"Infinity"`). When `allow_nan=False`, a user expects non-finite values to become `None`, but positive/negative Infinity becomes the JSON string `"Infinity"` instead. This is a real inconsistency, but it is the opposite polarity from what the census described. The severity is correctly low.

---

## F9 — Exporter tests use mock fixtures; real result types never imported in tests

**REAL: TRUE**  
**Corrected severity: medium** (unchanged)

Confirmed by reading `tests/test_exporters.py:38–111`. Three mock dataclasses (`MockCFLIBSResult`, `MockMCMCResult`, `MockNestedSamplingResult`) mirror the real types structurally but are entirely separate. The real `CFLIBSResult` has `@property` methods (e.g., `ne_cm3`, `T_K_from_eV`) that are invisible to `_dataclass_to_dict` because it uses `dataclasses.fields(obj)` which skips properties. The real `MCMCResult` also has `n_e_mean` and `T_K_mean` as properties (confirmed at `results.py:65–73`). These would be silently dropped in a real-type export. The test suite does not exercise this path. An integration test using real constructors with minimal valid arguments would have caught this.

---

## F10 — No round-trip test for `forward` CLI header → `load_spectrum`

**REAL: FALSE**  
**Corrected severity: none**

This finding is contingent on F5. Since the `intensity_W_m2_nm_sr` alias IS present in `_load_spectrum_csv` (verified above), the round-trip works. While adding a regression test for the full round-trip is still good practice, it is not testing a broken path. No functional gap exists.

---

## Findings MISSED by the Census

### M1: `_dataclass_to_dict` drops `@property` fields silently — confirmed functional gap, not just a test gap

**Severity: medium**

The census identifies this as a test gap (F9) but the underlying code defect is independent of tests. `Exporter._dataclass_to_dict` (lines 179–192) uses `dataclasses.fields(obj)` which returns only declared fields, not `@property` methods. `MCMCResult.n_e_mean` (the linear electron density in cm⁻³) is a property, not a field. When a real `MCMCResult` is exported, the linear n_e is absent from the CSV/HDF5/JSON output — only `log_ne_mean` (the log₁₀ value) appears. This is a data-loss bug distinct from F1 (labeling) and F9 (test coverage): a user exporting an `MCMCResult` object directly and looking for `n_e_mean` in the output will not find it even though the property exists on the result object.

### M2: `JSONExporter._float_to_json` Infinity handling ignores `allow_nan` flag

**Severity: low**

Described in F8 corrected reasoning above. When `allow_nan=False`, NaN → `None` but Infinity → `"Infinity"` string. A caller setting `allow_nan=False` to request `None`-substitution for all non-finite values will get inconsistent behavior: NaN is substituted but Infinity is not. The `elif np.isinf(value)` branch has no `if self.allow_nan` guard.

---

## Summary Table

| # | Census Finding | REAL | Corrected Severity | Notes |
|---|---------------|------|--------------------|-------|
| F1 | `log_ne` label missing base/unit | TRUE | medium | Confirmed |
| F2 | Concentration export missing unit label | TRUE | medium | Confirmed |
| F3 | "Lazy import" comment but eager top-level | TRUE | medium | Confirmed |
| F4 | Duck-typed `_is_*` dispatch fragile | TRUE | medium | Confirmed |
| F5 | `forward` CLI → `load_spectrum` round-trip broken | **FALSE** | **none** | `intensity_W_m2_nm_sr` IS in alias list (spectrum.py:44) |
| F6 | Python per-row loop in `_export_spectrum` | TRUE | medium | Confirmed |
| F7 | HDF5 `dtype="S10"` truncates element names | TRUE | medium | Confirmed |
| F8 | NaN→string, Inf→invalid JSON token asymmetry | **PARTIALLY FALSE** | low | Inf returns string too; real bug is `allow_nan` not applied to Inf |
| F9 | Tests use mocks; real types never exercised | TRUE | medium | Confirmed |
| F10 | No round-trip test for CLI header | **FALSE** | **none** | Contingent on F5 which is false |
| M1 | `_dataclass_to_dict` drops `@property` fields | NEW | medium | Real data-loss when exporting live objects |
| M2 | `allow_nan=False` not applied to Infinity | NEW | low | Inf always → string regardless of flag |

**Confirmed critical/high findings: 0**  
**Highest confirmed severity: medium** (F1, F2, F3, F4, F6, F7, F9, M1)
