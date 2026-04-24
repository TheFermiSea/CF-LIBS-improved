# Documentation Audit and Updates — 2026-04-23

## Summary

Comprehensive audit of docs/ directory against current codebase state. Updated stale references, documented major architectural changes, and added missing sections.

## Changes

### 1. Inversion Package Reorganization

- Documented that `cflibs.inversion/` is now organized into 6 sub-packages (common, preprocess, physics, identify, solve, runtime)
- Noted backward-compatible shims for old flat import paths
- Updated API_Reference.md with sub-package table and import guidance
- Updated CF-LIBS_Codebase_Technical_Documentation.md with location and structure

### 2. Physics-Only Constraint and ML Ban (Ruff TID251)

- Added to Deployment.md: "Physics-Only Constraint" section explaining hard constraint
- Added to API_Reference.md: Note about physics-only enforcement and banned modules
- Cross-referenced CLAUDE.md for full specifications
- Documented dual enforcement: Ruff TID251 static rule + AST blocklist scanner

### 3. New cflibs/evolution/ Package

- Added "Algorithm Evolution Framework" section to CF-LIBS_Codebase_Technical_Documentation.md
- Explained purpose: LLM-driven algorithm optimization (tooling only, not shipped algorithm)
- Added to index.rst toctree under "Developer Notes"
- Cross-referenced docs/Evolution_Framework.md (written by separate agent)

### 4. De-ML Pass Cleanup

- Removed old references to sklearn-based implementations
- Noted that jax.nn.softmax replaced by softmax_closure helper
- Noted that sklearn.ElasticNet replaced by L-BFGS-B optimizer in ALIAS
- No direct changes to User_Guide.md or API_Reference.md CLI examples (still accurate)

## Files Updated

- `/home/brian/code/CF-LIBS-improved/docs/index.rst` — Added Evolution_Framework to toctree
- `/home/brian/code/CF-LIBS-improved/docs/API_Reference.md` — Added inversion sub-package table, physics-only note
- `/home/brian/code/CF-LIBS-improved/docs/Deployment.md` — Added physics-only constraint section
- `/home/brian/code/CF-LIBS-improved/docs/CF-LIBS_Codebase_Technical_Documentation.md` — Updated inversion section, added evolution framework section

## Files Verified (No Changes Needed)

- User_Guide.md — CLI invocations and config schemas match current code
- Database_Generation.md — Instructions still accurate
- Manifold_Generation_Guide.md — Configuration examples still valid
- Hardware_Interfaces.md — Architecture descriptions match cflibs/runtime/
- Echellogram_Processing_Guide.md — Algorithm descriptions match cflibs/instrument/echelle.py
- REFERENCE_ANALYSIS_LIBSSA.md — Reference material (no code dependencies)

## Notes

- Backward compatibility preserved: old imports like `from cflibs.inversion.solver import X` still work
- Phase status language ("Phase 2", "Phase 3") already updated in prior commits
- All cross-references validated to avoid broken links
