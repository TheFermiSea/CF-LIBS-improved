# ADR-0002: Do not extend `_make_predictor` to the standalone non-ALIAS identifiers

**Status:** accepted (2026-05-27)

## Context

PR #205 introduced `cflibs/benchmark/unified.py::_make_predictor` and `ID_WORKFLOW_PRESETS`
to collapse six near-identical ALIAS-family builders (alias / alias_v2 /
alias_high_recall / hybrid_consensus_2of3 / 2of4_with_nnls / weighted) into a
single factory + data-table pair. PR #208 then lifted the cocktails themselves
into `ALIAS_PRESETS` in `cflibs/inversion/identify/alias.py`.

Four sibling builders in the same file remained inline:

- `_build_comb_predictor`
- `_build_correlation_predictor`
- `_build_nnls_predictor`
- `_build_hybrid_predictor`

A 2026-05-27 architecture review suggested folding these four into `_make_predictor`
for "consistency with the ALIAS family."

## Decision

**Do not extend `_make_predictor` to the four standalone non-ALIAS builders.**
Leave each as an inline 25–30-line closure that constructs its identifier
directly and calls `.identify()`.

## Why

`_make_predictor` is ALIAS-centric by design — its body is "build an ALIAS main
voter from a cocktail, optionally orchestrate sibling voters and consensus voting."
The four standalone identifiers do not fit that shape:

| Builder | Needs `AtomicDatabase`? | Needs `BasisLibrary`? | `.identify()` signature |
|---|---|---|---|
| comb | yes | no | `(wl, intensity)` |
| correlation | yes | no | `(wl, intensity, mode="classic")` |
| nnls | **no** | yes | `(wl, intensity)` |
| hybrid | yes | yes | `(wl, intensity)` |

Forcing them through `_make_predictor` would require either:

1. **One identifier-specific branch per class inside the factory body** —
   which is what `_build_sibling_voter` already does (lines 1380–1409) and which
   is itself shallow boilerplate.
2. **Generalising `_make_predictor`'s interface** until it is essentially
   `lambda context, candidates, config: callable_returning_predictor` — at which
   point the factory adds no leverage over the inline closure.

Neither path beats the status quo on the deletion test: removing the inline
closures and inlining the bodies into `build_id_workflow_registry` would *reduce*
complexity, not concentrate it. The "consistency" win is visual symmetry, not
locality or leverage.

The original signal — "four builders look different from the six factory-backed
ones" — was real, but the friction was symmetry-for-its-own-sake, not
architectural debt. Symmetry is not depth.

## Consequences

- The four `_build_*_predictor` closures stay inline. New non-ALIAS identifiers
  added later get their own inline closure in the same style.
- `_make_predictor`'s docstring already says "Currently always ALIASIdentifier
  (the only class the 6 collapsed builders use as their 'main' voter). Kept as
  an argument so future workflows (e.g. comb-led consensus) can reuse the
  factory without further refactor." That door stays open — *consensus*
  workflows with a non-ALIAS main voter would fit the factory's shape. This ADR
  only rejects extending the factory to *standalone* (non-consensus) predictors.
- Future architecture reviews should not re-surface this candidate.
