# cflibs/pds — Adversarial Verification Report

Verifier role: adversarial cross-check of census findings  
Worktree: `/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5`  
Method: ripgrep + Read on actual source; compared against literature in `overhaul/literature/`  
Date: 2026-06-25

---

## Verification of HIGH-Severity Findings

### F1-1 · Ground-truth weight-fraction basis vs CF-LIBS mole fractions

**REAL: TRUE**  
**Corrected severity: HIGH** (confirmed, no downgrade warranted)

Code verification:

- `corpus.py:65-67` — `CorpusEntry.expected_elements` docstring explicitly says "known weight fraction (0-1)".
- `corpus.py:116` — inline comment: "CCCT compositions (weight fractions of major oxides → elements)".
- `cflibs/inversion/solve/iterative.py` — `CFLIBSResult.concentrations` docstring says "Element concentrations (number/mole fractions, sum to 1)". Confirmed with `cflibs/inversion/physics/closure.py` which labels the closure output "element -> number (mole) fraction (sum=1)".
- `cflibs/pds/validation.py:80-83` — `quantified_elements` property returns `entry.expected_elements` (weight fractions) directly, with no conversion.
- No `mole_to_weight` or equivalent function exists anywhere in `cflibs/pds/`. The conversion utility `mass_to_number_fractions` exists in `cflibs/benchmark/synthetic_corpus.py` but is not accessible from `cflibs/pds/`.
- `cflibs-method.md` confirms: "CF-LIBS closure gives C_s = mole (number) fractions... Völker (2024) quantified errors up to 353% from omitting molar-mass conversion."

The mismatch is confirmed: corpus stores weight fractions, pipeline outputs mole fractions, and no conversion helper exists in the pds package. Any caller comparing `PDSValidationDataset.quantified_elements` against `CFLIBSResult.concentrations` directly would compute systematically wrong residuals. Finding is real.

---

### F5-1 · PDSCache.fetch() has no network mock and post-download validation is unexercised

**REAL: TRUE**  
**Corrected severity: MEDIUM** (downgraded from HIGH — see reasoning)

Code verification:

- `cache.py:116-127` — confirmed: `response.read()` loads whole file; no `is_cached()` call after the write; `MemoryError` is not in the caught exception tuple `(urllib.error.URLError, urllib.error.HTTPError, OSError)`.
- `tests/test_pds.py` — confirmed: `TestPDSCache` covers `is_cached`, `cached_path`, `clear`, `status` only. Zero `fetch()` tests, zero `urlopen` mocks.
- Note: `shutil` is already imported in `cache.py` (line 19), so the streaming-write fix is trivial.

Severity downgrade rationale: The census rated this HIGH, but the impact is bounded. `cflibs.pds` is an I/O utility with no downstream wiring (F2-2 is confirmed). A corrupt downloaded file would produce a clear `csv.Error` at parse time, not silent composition errors. The MemoryError risk is negligible for PDS CSV files (< a few MB). The missing `is_cached()` post-write check means an HTML error page would only be caught at parse time, not at download time — a UX annoyance, not a safety hazard. The test gap is real and should be fixed, but HIGH overstates the risk given the isolation of this code.

---

## Verification of Additional MEDIUM-Severity Findings (Selected)

### F1-2 · SCCT5/SCCT7 compositions copy-pasted from CCCT3/CCCT2 with K silently dropped

**REAL: TRUE**  
**Corrected severity: MEDIUM** (confirmed)

Code verification:

- `corpus.py:233-241` (SCCT5): `{"Si": 0.207, "Al": 0.049, "Mg": 0.110, "Fe": 0.086, "Ca": 0.053, "Na": 0.009, "Ti": 0.005, "O": None}` — K absent.
- `corpus.py:143-154` (CCCT3): identical Si/Al/Mg/Fe/Ca/Na/Ti values **plus** `"K": 0.002`.
- `corpus.py:244-253` (SCCT7): `{"Si": 0.243, ..., "Ti": 0.003, "O": None}` — K absent.
- `corpus.py:132-142` (CCCT2): identical values **plus** `"K": 0.003`.

The numerical copy-paste is exact (all non-K values match to 3 decimal places). Whether the SCCT targets share the same rock material as the CCCT targets is unverified in the code — the comment says "similar to", not "identical to", and no primary citation is given for the SCCT values. K lines (766/769 nm) are strong in LIBS and omitting a known ~0.2-0.3% K from the ground-truth denominator will produce systematic bias. Finding is real.

Adversarial note: it is plausible that the SCCT and CCCT targets are different cuts of the same rock samples and K was intentionally omitted from the SCCT reference because its concentration was below the relevant detection limit for SuperCam — but this is not documented in the code, so the finding stands as a documentation/data-quality issue.

---

### F2-2 · PDSValidationDataset is a dead-end schema bridge — not wired downstream

**REAL: TRUE**  
**Corrected severity: MEDIUM** (confirmed)

Code verification:

- `rg "PDSValidationDataset|map_chemcam_to_validation|map_supercam_to_validation"` across all of `cflibs/` excluding `pds/validation.py` and `tests/test_pds.py` → **zero results**.
- `cflibs/benchmark/` and `cflibs/validation/` have no imports from `cflibs.pds.validation`.
- `PDSValidationDataset` is used only in `tests/test_pds.py` (test-only consumption, not pipeline wiring).

Finding is confirmed. The bridge module is real code but is not wired to any downstream consumer. It is not dead in a harmful sense (it works), but its docstring claims it "Connects the PDS ingestion layer to the existing round-trip validation and benchmark infrastructure" — a claim that is currently false.

---

### F3-1 · PDSCache.fetch() reads entire file into memory before writing

**REAL: TRUE**  
**Corrected severity: MEDIUM** (confirmed, not upgraded)

Code verification:

- `cache.py:119`: `f.write(response.read())` — whole-file read confirmed.
- `shutil` is already imported at line 19; the streaming fix `shutil.copyfileobj(response, f)` is a one-liner.
- The `MemoryError` gap is real but inconsequential for the file sizes involved.
- Absence of post-write `is_cached()` check is confirmed — the function returns `dest` at line 127 without validation.

Finding is confirmed. The streaming fix is trivial given `shutil` is already imported.

---

## Findings the Census MISSED

### M1 · CCCT4 K omission inconsistent with Fabre et al. (2011)

**Severity: LOW**  
**Location:** `corpus.py:155-166`

CCCT4 (Shergottite) composition in the corpus lists Si/Al/Fe/Mg/Ca/Na/K/Ti but `K: 0.001`. This is fine. However, the Shergottite composition per Fabre et al. (2011) for CCCT-4 gives K₂O < 0.06 wt% which converts to K < 0.0005 wt fraction — an order of magnitude smaller than `K: 0.001`. This may be a rounding issue or a different source, but it is not cited. Low severity because K at 0.001 is negligible for LIBS analysis.

### M2 · PDSValidationDataset.expected_elements type annotation is unguarded

**Severity: LOW**  
**Location:** `validation.py:48-49`, `validation.py:80-83`

`PDSValidationDataset.expected_elements: Dict[str, Optional[float]]` passes `None` values for qualitatively-expected elements. The `quantified_elements` property correctly filters these out. However, any caller that iterates over `expected_elements` directly (not via `quantified_elements`) and attempts `float(v)` on a `None` value will raise a `TypeError` with no guard. Since `PDSValidationDataset` is not wired downstream, this is currently unexploited, but it is a latent API hazard.

### M3 · CCCT9 Ti alloy does not sum to 1.0 (O intentionally absent)

**Severity: NONE (confirmed correct)**

`corpus.py:179-185` — CCCT9: `{"Ti": 0.895, "Al": 0.061, "V": 0.040, "Fe": 0.004}` sums to 1.000 exactly (Ti-6Al-4V specification: 6% Al, 4% V nominal, balance Ti). O is absent because Ti alloys are metallic, not oxide matrices. This is correct. The census did not flag it, and the verifier confirms it is not a finding.

---

## Summary Table

| ID    | Census Severity | REAL?   | Confirmed Severity | Notes |
|-------|-----------------|---------|-------------------|-------|
| F1-1  | HIGH            | TRUE    | HIGH              | CF-LIBS mole fracs vs weight frac corpus, no conversion helper |
| F5-1  | HIGH            | TRUE    | MEDIUM            | Test gap real; impact bounded (I/O-only, not downstream) |
| F1-2  | MEDIUM          | TRUE    | MEDIUM            | K silently dropped in SCCT copy-pastes |
| F2-2  | MEDIUM          | TRUE    | MEDIUM            | PDSValidationDataset unwired despite docstring claim |
| F3-1  | MEDIUM          | TRUE    | MEDIUM            | Whole-file read; shutil already imported; trivial fix |
| M1    | —               | NEW     | LOW               | CCCT4 K value order-of-magnitude discrepancy vs Fabre (2011) |
| M2    | —               | NEW     | LOW               | expected_elements None-unsafe for non-quantified callers |

**Highest confirmed severity: HIGH (F1-1)**
