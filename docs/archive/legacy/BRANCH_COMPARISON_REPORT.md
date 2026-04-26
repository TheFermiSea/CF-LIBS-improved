# Branch Comparison Report: Peak Detection Improvements

**Date**: 2026-02-17  
**Branches Compared**: PR #29, #30, #31, #32  
**Status**: All branches fixed and tested ✅

---

## Executive Summary

Four parallel branches implemented different approaches to reduce peak detection false positives. After fixing all review comments, we compared implementations to identify the best parts of each branch for consolidation.

**Key Finding**: PR #30 (opus_max) provides the most comprehensive and robust foundation, with PR #32 adding valuable calibration tooling and API flexibility.

---

## Test Results Summary

| Branch | Tests Passed | Tests Skipped | Status |
|--------|--------------|---------------|--------|
| PR #29 (Opus) | 15 | 3 | ✅ All pass |
| PR #30 (opus_max) | 16 | 2 | ✅ All pass |
| PR #31 (Composer) | (via worktree) | - | ✅ All pass |
| PR #32 (Codex) | 17 | 1 | ✅ All pass |

All branches pass their test suites. Differences are in implementation approach and feature completeness.

---

## Feature Comparison Matrix

### preprocessing.py Core Features

| Feature | PR #29 | PR #30 | PR #31 | PR #32 | Winner |
|---------|--------|--------|--------|--------|--------|
| **detect_peaks_auto() wrapper** | ✅ | ✅ | ❌ | ❌ | #29/#30 |
| **Cosmic ray rejection** | ✅ (int) | ✅ (float) | ❌ | ❌ | **#30** |
| **Second-derivative confirmation** | ❌ | ✅ | ❌ | ❌ | **#30** |
| **Noise floor clamping** | ❌ | ✅ | ❌ | ❌ | **#30** |
| **Distance API** | `min_distance_nm` | `resolving_power` | `distance_px` | Both | **#32** |
| **Wavelength guards** | ✅ | ✅ | ❌* | ✅ | #29/#30/#32 |
| **Error handling/logging** | Basic | ✅ Logger | Basic | Basic | **#30** |
| **Code size** | 242 lines | 288 lines | 140 lines | 153 lines | - |

*PR #31 missing guards were fixed in review

### Key Implementation Differences

#### 1. Cosmic Ray Rejection

- **PR #29**: `min_width_pts` (integer, default 2) - simple but less flexible
- **PR #30**: `min_fwhm_pixels` (float, default 1.5) - more flexible, resolution-aware
- **PR #31/#32**: No cosmic ray rejection

**Recommendation**: Use PR #30's float-based approach for flexibility.

#### 2. Second-Derivative Confirmation (PR #30 ONLY)

Unique feature that filters peaks without positive curvature. Can reduce false positives from noise spikes.

```python
if use_second_derivative:
    d1 = np.gradient(corrected, spacing, edge_order=2)
    d2 = -np.gradient(d1, spacing, edge_order=2)
    # Filter peaks without positive curvature
```

**Recommendation**: Include as optional feature (default False) for advanced use cases.

#### 3. Noise Floor Clamping (PR #30 ONLY)

Critical for synthetic/noiseless spectra:

```python
if noise < 1e-10:
    noise = max(1e-10, float(np.nanpercentile(np.abs(intensity - baseline), 95)) * 1e-6)
```

**Recommendation**: **MUST INCLUDE** - prevents zero thresholds that break find_peaks.

#### 4. Distance Parameter API Design

- **PR #29**: `min_distance_nm` - wavelength-based, requires manual conversion
- **PR #30**: `resolving_power` - instrument-aware, automatic conversion
- **PR #31**: `distance_px` - pixel-based, simplest but manual
- **PR #32**: Both `resolving_power` + `min_distance_px` - most flexible

**Recommendation**: Use PR #32's dual-parameter approach for maximum flexibility while maintaining PR #30's instrument-aware defaults.

#### 5. detect_peaks_auto() Wrapper

- **PR #29/#30**: Convenience wrapper that estimates baseline/noise automatically
- **PR #31/#32**: No wrapper (callers must do estimation themselves)

**Recommendation**: **MUST INCLUDE** - reduces code duplication across identifiers.

---

## Identifier Implementation Comparison

### correlation_identifier.py

| Branch | Peak Detection | Caching | Performance |
|--------|----------------|---------|-------------|
| PR #29 | `detect_peaks_auto()` | ❌ | Good |
| PR #30 | `detect_peaks_auto()` | ✅ `self._peaks` | **Best** ✅ |
| PR #31 | Manual | ❌ | Good |
| PR #32 | `detect_peaks()` | ❌ | Good |

**Winner**: PR #30 (has caching - avoids redundant peak detection calls)

### alias_identifier.py & comb_identifier.py

All branches use similar approaches. PR #29/#30 use `detect_peaks_auto()` cleanly. PR #31/#32 require manual baseline/noise estimation.

**Winner**: PR #29/#30 (cleaner API usage)

---

## Additional Features by Branch

### PR #29 (Opus)
- ✅ Improvements to `line_selection.py` (resonance line identification)
- ✅ Improvements to `line_detection.py` (matching logic)
- **Scope**: 8 files, 403 insertions, 144 deletions

### PR #30 (opus_max) - **MOST COMPREHENSIVE**
- ✅ `quality.py` improvements (Saha ratio fixes)
- ✅ Peak detection caching in correlation_identifier
- ✅ Comprehensive error handling with logging
- **Scope**: 9 files, 478 insertions, 206 deletions

### PR #31 (Composer) - **SIMPLEST**
- ✅ Focused identifier-level changes
- ❌ Missing cosmic ray rejection
- ❌ Missing detect_peaks_auto wrapper
- **Scope**: 4 files, 257 insertions, 157 deletions

### PR #32 (Codex) - **UNIQUE TOOLING**
- ✅ **`scripts/calibrate_alias.py`** - calibration sweep tool (360 lines) - **UNIQUE** ✅
- ✅ CLI integration
- ✅ Flexible dual-parameter API
- **Scope**: 8 files, 523 insertions, 52 deletions

---

## Recommended Consolidation Strategy

### Phase 1: Core preprocessing.py

**Base**: Start with **PR #30** because:
1. ✅ Noise floor clamping (critical)
2. ✅ Second-derivative confirmation (unique feature)
3. ✅ Cosmic ray rejection (flexible float parameter)
4. ✅ `detect_peaks_auto()` wrapper
5. ✅ Logging support
6. ✅ Comprehensive error handling

**Enhancements from PR #32**:
- Add support for both `resolving_power` AND `min_distance_px` parameters
- Maintain backward compatibility with PR #30's API

**Enhancements from PR #29**:
- Review `line_selection.py` improvements for inclusion

### Phase 2: Identifier Files

**Use PR #30's approach** for all identifiers:
- `correlation_identifier.py` - use caching (`self._peaks`)
- `alias_identifier.py` - use `detect_peaks_auto()` cleanly
- `comb_identifier.py` - use `detect_peaks_auto()` cleanly

### Phase 3: Additional Features

**From PR #32**:
- ✅ Include `scripts/calibrate_alias.py` - valuable calibration tool
- ✅ Include CLI integration changes

**From PR #30**:
- ✅ Include `quality.py` improvements (Saha ratio fixes)

**From PR #29**:
- ✅ Review `line_selection.py` and `line_detection.py` improvements

---

## Implementation Plan

### Step 1: Create Consolidated Branch
```bash
git checkout -b feature/consolidated-peak-detection main
```

### Step 2: Merge Base (PR #30)
```bash
git merge fix/peak_opus_max_detection_pipeline --no-commit
# Resolve conflicts, keep PR #30's preprocessing.py as base
```

### Step 3: Add API Flexibility (PR #32)
```bash
git merge feature/cf-libs_codex_peak-calibration-compare --no-commit
# Add min_distance_px parameter support to detect_peaks()
# Keep PR #30's detect_peaks_auto() signature
```

### Step 4: Add Calibration Tool (PR #32)
```bash
# Keep scripts/calibrate_alias.py from PR #32
```

### Step 5: Add Quality Improvements (PR #30)
```bash
# Keep quality.py changes from PR #30
```

### Step 6: Review Line Selection (PR #29)
```bash
git merge fix/peak_opus_detection_overhaul --no-commit
# Review line_selection.py and line_detection.py changes
# Keep improvements that don't conflict
```

### Step 7: Test & Validate
```bash
# Run full test suite
pytest tests/ -v

# Run calibration sweep (if applicable)
python scripts/calibrate_alias.py --help
```

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| API incompatibility | Medium | Maintain backward compatibility, deprecate old params gradually |
| Performance regression | Low | PR #30's caching should improve performance |
| Test failures | Low | All branches pass tests; consolidation should maintain compatibility |
| Feature conflicts | Medium | Careful merge conflict resolution, prioritize PR #30 as base |

---

## Success Criteria

✅ All tests pass on consolidated branch  
✅ No performance regression (should improve due to caching)  
✅ All unique features preserved:
- Noise floor clamping
- Second-derivative confirmation
- Cosmic ray rejection
- Calibration tool
- Quality improvements

---

## Next Steps

1. ✅ **COMPLETE**: Fix all review comments on all branches
2. ✅ **COMPLETE**: Compare implementations (this document)
3. ⏭️ **NEXT**: Create consolidated branch following implementation plan
4. ⏭️ **NEXT**: Run full test suite on consolidated branch
5. ⏭️ **NEXT**: Run calibration sweep to validate improvements
6. ⏭️ **NEXT**: Create PR for consolidated implementation
7. ⏭️ **NEXT**: Close individual PRs after consolidation is merged

---

## Archival Note

This document was archived 2026-04-23. It documents the peak detection improvement consolidation process from February 2026. Refer to this for historical context on branch comparison methodology and consolidation rationale.
