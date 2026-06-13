# jitpipe stage contracts + tolerance tiers (ADR-0004 §4 / §5.4)

The jittable pipeline (`cflibs.jitpipe`) is a parallel re-implementation of the
reference inversion pipeline (ADR-0004 D1). Each stage carries a **parity
contract** against its reference adapter (`cflibs.jitpipe.parity`), graded by a
**tolerance tier**.

## Precision policy (ADR-0004 §5.3)

All jitpipe device arrays are float64 under `JAX_PLATFORMS=cpu`/`cuda`
(`jax_enable_x64=True`; `conftest.py` forces it for tests). Metal (no fp64) is
out of scope for jitpipe. Per-line metadata uses the narrowest exact integer
dtype (i16/i8/i32/u8) that holds the DB values.

## Tolerance tiers (K / S / D / B)

| Tier | Scope | Default contract |
|---|---|---|
| **K** | atomic-data passthrough (snapshot fields consumed verbatim by kernels) | **exact** (`rtol=atol=0`, NaN-aware) |
| **S** | Saha-Boltzmann scalar physics (partition U, ionization fractions) | tight (`rtol≈1e-10`) |
| **D** | detection / identification scores (thresholded, discrete decisions) | decision-equivalence (same lines selected) |
| **B** | full-pipeline plasma parameters (T, n_e, concentrations) | loose (physics-level, set per stage in J7) |

## Per-stage contract registry

| Stage | Module | Bead | Tier | Reference adapter |
|---|---|---|---|---|
| snapshot bridge (forward) | `snapshot.to_atomic_snapshot` | J0 | K | `parity.reference_atomic_snapshot` |
| snapshot bridge (lax) | `snapshot.to_lax_snapshot` | J0 | K | `parity.reference_lax_snapshot` |
| preprocess | `preprocess.py` | J1 | D | `parity.preprocess_parity` (stub) |
| detect | `detect.py` | J2 | D | `parity.detect_parity` (stub) |
| calibrate | `calibrate.py` | J3 | D | (J3) |
| identify | `identify.py` | J4 | D | `parity.identify_parity` (stub) |
| fit (Boltzmann) | `fit.py` | J5 | S | `parity.fit_parity` (stub) |
| self-absorption | `selfabs.py` | J5 | S | (J5) |
| stark n_e | `stark.py` | J6 | S | (J6) |
| solve (iterative) | `solve.py` | J7 | B | `parity.solve_parity` (stub) |
| forward | `forward.py` | J7 | K/S | `kernels.forward_model` |

**J0 status:** the two snapshot-bridge contracts (tier K) are implemented and
tested (`tests/jitpipe/test_snapshot.py::test_forward_bridge_parity`,
`test_lax_bridge_parity`) against both legacy builders for a 15-element
candidate set. Per-stage contracts J1–J7 are registered here but their
adapters are stubs until the stages land.

## Eager-fallback semantics (spec §2, AC7)

The canonical scalar partition fallbacks (`PipelineSnapshot.canonical_fallback`)
are evaluated **eagerly at snapshot build** — what was a lazy per-solve probe in
the reference lax solver (`iterative.py:479-490`) becomes a one-time
process-level cost. Closed-shell ions get their exact U (e.g. Na II → 1.0, not
the 15.0 placeholder); other missing-data species get the canonical stage
defaults (U_I = 25.0, U_II = 15.0). The forward-kernel `partition_coeffs` are
the `partition_spec_for` direct-sum re-fit (concrete, no NaN); the lax-path
`partition_coeffs_stored` keeps the raw table poly with the NaN sentinel.

## Boundary invariants

- Only `host.py` / `snapshot.py` / `parity.py` may import SQLite-backed code
  (`tests/jitpipe/test_import_hygiene.py`).
- Nothing outside `cflibs/jitpipe/` imports `cflibs.jitpipe` (ADR-0004 D1).
- Physics-only constraint inherited: ruff TID251 bans sklearn/torch/etc.
