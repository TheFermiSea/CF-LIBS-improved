"""Per-path U(T) VALUE GATE — the partition-function contract guard.

This is the Wave-5 gate of the 2026-06-03 partition-provider unification
(epic CF-LIBS-improved-8dn7, composition-pipeline diagnosis § 2.1 / § 2.6).

The original defect (PF-1..PF-4) was that ``U(T)`` is computed at ~9 sites and
the JAX manifold / Bayesian forward models silently consumed a *stored
polynomial* that was 1.7–2.6× too low at LIBS temperatures, while the only
nominal validation gate exercised the (correct) direct-sum path and never the
broken artifact.  Unit tests on the factory alone are not enough: they prove the
factory is right, not that EVERY consumer ADAPTER actually evaluates the
factory's coefficients+bounds.

This module closes that gap.  For the workhorse species
(Fe/Cr/Ti/Ni/Cu/Al/Ca/Mg/Si, stages I+II) at 8000/10000/12000 K it asserts that
``U(T)`` from *every* consumer path agrees with the DIRECT-SUM reference
(``Σ gᵢ exp(-Eᵢ/kT)`` over the DB's own ``energy_levels``) within tolerance:

* **CPU scalar provider** — ``partition_function_for(...).at(T)`` (the default
  ``invert`` / ``analyze`` / iterative / closed-form path).  This adapter
  evaluates the EXACT direct sum, so it must match the reference to
  floating-point precision (``CPU_RTOL``).  This is the Invariant-#2 regression
  guard: the CPU path was already direct-sum-first and must stay bit-for-bit.

* **Manifold / JAX batched adapter** — ``polynomial_partition_function_jax``
  evaluated over the per-species arrays baked into ``AtomicDatabase.snapshot``.
  Direct-sum-preferred is a BUILD-TIME choice here (``vmap`` needs static
  fixed-shape arrays), so this adapter evaluates the direct-sum-FIT polynomial,
  not the exact sum; it agrees with the reference only within the fit tolerance
  (``FIT_RTOL``).

* **Bayesian adapter** — ``AtomicDataArrays`` built by the Bayesian atomic
  loader, evaluated through the now-shared guarded
  ``cflibs.inversion.solve.bayesian.atomic.partition_function`` (the same shared
  evaluator the manifold uses, after the unguarded duplicate was deleted).  Same
  direct-sum-fit polynomial, plus float32 storage, so also gated at
  ``FIT_RTOL``.

The gate FAILS if any path silently diverges — which is precisely what would
have caught the original defect: the broken polynomial was 40–60 % below the
direct sum at 10000 K, far outside ``FIT_RTOL``.

Tolerance rationale (LIBS band, workhorse species with near-complete levels):
the regenerated direct-sum fit tracks the direct sum to ≤ ~7 % across
8000–12000 K (worst case Mg I @ 12000 K ≈ 6.9 %); ``FIT_RTOL = 0.10`` clears
that with margin while still rejecting the order-of-magnitude original defect.
At the hot edge (≥ ~12000 K) a few light-metal / Cr II species are bounded by
NIST ``energy_levels`` completeness rather than by the fit (diagnosis open
Q #3); that ceiling is a property of the level table common to ALL paths, so it
does not break path-to-path agreement and is documented here rather than
loosened away.

Runs against the production DB (the fit is keyed on physical ``(element,
sp_num)`` tuples; a synthetic fixture would tautologically pass).  Bounded and
fast: a single shared snapshot + cached per-species fits, no MCMC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cflibs.atomic.database import AtomicDatabase
from cflibs.plasma.partition import (
    direct_sum_partition_function,
    get_levels_for_species,
)

DB_PATH = Path(__file__).resolve().parent.parent / "ASD_da" / "libs_production.db"

# Workhorse species (iron-group + light metals), both LIBS-relevant stages.
# These all carry near-complete NIST energy_levels, so the direct-sum fit is the
# achievable target and the gate can be tight.
WORKHORSE_ELEMENTS = ["Fe", "Cr", "Ti", "Ni", "Cu", "Al", "Ca", "Mg", "Si"]
STAGES = (1, 2)
SPECIES = [(el, st) for el in WORKHORSE_ELEMENTS for st in STAGES]

# ps-LIBS band where every adapter must agree with the direct-sum reference.
LIBS_TEMPS = (8000.0, 10000.0, 12000.0)

# CPU scalar adapter evaluates the EXACT direct sum -> floating-point parity.
CPU_RTOL = 1e-6
# JAX / manifold / Bayesian adapters evaluate the direct-sum FIT polynomial.
# Worst observed workhorse deviation across 8000-12000 K is ~6.9 % (Mg I @
# 12000 K); 0.10 clears that with margin yet still rejects the original
# 40-60 %-low defect.  (Bayesian additionally rounds to float32 -> ~1e-6 noise,
# negligible against this bar.)
FIT_RTOL = 0.10


# ---------------------------------------------------------------------------
# Shared, module-scoped state (bounded/fast: build the snapshot + fits ONCE).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def db():
    if not DB_PATH.exists():
        pytest.skip(f"production DB not present at {DB_PATH}")
    return AtomicDatabase(str(DB_PATH))


@pytest.fixture(scope="module")
def reference(db):
    """Direct-sum reference U(T) = Σ gᵢ exp(-Eᵢ/kT) per species per LIBS temp.

    This is the single source of truth the gate trusts (diagnosis § 2.1): the
    DB's own ``energy_levels`` summed directly, no polynomial, no IPD (isolated
    atom, matching the validator's Barklem & Collet convention).

    Returns ``{(element, stage): {T: U_directsum}}``.
    """
    ref: dict[tuple[str, int], dict[float, float]] = {}
    for element, stage in SPECIES:
        levels = get_levels_for_species(db, element, stage)
        assert levels is not None, (
            f"{element} {stage}: workhorse species has no energy_levels; this gate "
            "assumes the direct-sum reference is available for every workhorse species"
        )
        g_arr, e_arr, ip_ev = levels
        ref[(element, stage)] = {
            T: direct_sum_partition_function(T, g_arr, e_arr, ip_ev, n_e=None) for T in LIBS_TEMPS
        }
    return ref


@pytest.fixture(scope="module")
def manifold_snapshot(db):
    """The actual ``AtomicSnapshot`` the manifold / JAX forward models consume.

    Built ONCE (the per-species direct-sum fits are cached, Invariant #5).
    Exercising the real snapshot — not a re-call of ``partition_spec_for`` —
    is what makes this a genuine guard: it would catch a snapshot builder that
    silently diverged from the factory.
    """
    pytest.importorskip("jax")
    return db.snapshot(elements=WORKHORSE_ELEMENTS, wavelength_range=(200.0, 900.0))


@pytest.fixture(scope="module")
def bayesian_arrays(db):  # noqa: ARG001 — db kept for fixture ordering/skip parity
    """The ``AtomicDataArrays`` the Bayesian NUTS forward model consumes.

    Built ONCE from the Bayesian atomic loader, which sources its partition
    coefficients+bounds from the SAME factory (``partition_spec_for``).
    """
    pytest.importorskip("jax")
    from cflibs.inversion.solve.bayesian.atomic import _query_atomic_data

    df, coeffs, ips, t_min, t_max, g0 = _query_atomic_data(
        str(DB_PATH), WORKHORSE_ELEMENTS, (200.0, 900.0)
    )
    el_idx = {el: i for i, el in enumerate(WORKHORSE_ELEMENTS)}
    return el_idx, coeffs, t_min, t_max, g0


# ---------------------------------------------------------------------------
# (1) CPU scalar adapter == direct sum  (Invariant #2 regression guard)
# ---------------------------------------------------------------------------


@pytest.mark.physics
@pytest.mark.requires_db
@pytest.mark.parametrize("element,stage", SPECIES)
def test_cpu_scalar_path_matches_direct_sum(db, reference, element, stage):
    """CPU ``.at(T)`` reproduces the direct-sum reference to fp precision.

    The CPU scalar provider IS the exact direct sum (clamped + g0-floored), so
    this is the tightest gate — it would FAIL if the provider ever quietly
    switched to the polynomial fit for a species that has levels.
    """
    provider = db.partition_function_for(element, stage)
    assert provider is not None, f"{element} {stage}: factory returned no provider"
    for T in LIBS_TEMPS:
        u_cpu = float(provider.at(T))
        u_ref = reference[(element, stage)][T]
        rel = abs(u_cpu - u_ref) / u_ref
        assert rel <= CPU_RTOL, (
            f"CPU scalar path diverged for {element} {stage} @ {T:.0f} K: "
            f"U_cpu={u_cpu:.6f} vs direct-sum={u_ref:.6f} ({rel:+.2%} > {CPU_RTOL:.0e}). "
            "The default invert/analyze/iterative U(T) must stay bit-for-bit "
            "direct-sum (Invariant #2)."
        )


# ---------------------------------------------------------------------------
# (2) Manifold / JAX batched adapter == direct sum (within fit tolerance)
# ---------------------------------------------------------------------------


@pytest.mark.physics
@pytest.mark.requires_jax
@pytest.mark.requires_db
@pytest.mark.parametrize("element,stage", SPECIES)
def test_manifold_jax_path_matches_direct_sum(manifold_snapshot, reference, element, stage):
    """Snapshot polynomial U(T) (the manifold/JAX adapter) tracks the direct sum.

    Evaluates the ONE shared guarded ``polynomial_partition_function_jax`` over
    the per-species coefficients+bounds baked into the real ``AtomicSnapshot``.
    This is the path that originally shipped 40-60 % low and was never gated.
    """
    from cflibs.plasma.partition import polynomial_partition_function_jax

    snap = manifold_snapshot
    species_to_idx = {key: i for i, key in enumerate(snap.species)}
    assert (
        element,
        stage,
    ) in species_to_idx, f"{element} {stage} missing from snapshot species axis {snap.species!r}"
    i = species_to_idx[(element, stage)]
    coeffs = np.asarray(snap.partition_coeffs)[i]
    t_min = float(np.asarray(snap.partition_t_min)[i])
    t_max = float(np.asarray(snap.partition_t_max)[i])
    g0 = float(np.asarray(snap.partition_g0)[i])

    for T in LIBS_TEMPS:
        u_jax = float(polynomial_partition_function_jax(T, coeffs, t_min=t_min, t_max=t_max, g0=g0))
        u_ref = reference[(element, stage)][T]
        rel = abs(u_jax - u_ref) / u_ref
        assert rel <= FIT_RTOL, (
            f"Manifold/JAX snapshot path diverged for {element} {stage} @ {T:.0f} K: "
            f"U_snapshot={u_jax:.4f} vs direct-sum={u_ref:.4f} ({rel:+.2%} > {FIT_RTOL:.0%}). "
            "The snapshot direct-sum-fit polynomial must track the direct sum at "
            "LIBS T — this is the gate that catches the PF-1..PF-4 defect class."
        )


# ---------------------------------------------------------------------------
# (3) Bayesian adapter == direct sum (within fit tolerance)
# ---------------------------------------------------------------------------


@pytest.mark.physics
@pytest.mark.requires_jax
@pytest.mark.requires_db
@pytest.mark.parametrize("element,stage", SPECIES)
def test_bayesian_path_matches_direct_sum(bayesian_arrays, reference, element, stage):
    """Bayesian ``AtomicDataArrays`` U(T) routed through the shared guarded evaluator.

    Guards the site that was the LIVE unguarded duplicate of the JAX poly
    evaluator (no clamp, no g0 floor, no [t_min,t_max]) consumed inside the NUTS
    jit trace.  After the unification it must agree with the direct sum like the
    manifold path.
    """
    from cflibs.inversion.solve.bayesian.atomic import partition_function

    el_idx, coeffs, t_min, t_max, g0 = bayesian_arrays
    ei = el_idx[element]
    si = stage - 1
    c = coeffs[ei, si]
    for T in LIBS_TEMPS:
        u_bayes = float(
            partition_function(
                T, c, t_min=float(t_min[ei, si]), t_max=float(t_max[ei, si]), g0=float(g0[ei, si])
            )
        )
        u_ref = reference[(element, stage)][T]
        rel = abs(u_bayes - u_ref) / u_ref
        assert rel <= FIT_RTOL, (
            f"Bayesian path diverged for {element} {stage} @ {T:.0f} K: "
            f"U_bayes={u_bayes:.4f} vs direct-sum={u_ref:.4f} ({rel:+.2%} > {FIT_RTOL:.0%}). "
            "The Bayesian U(T) must route through the shared guarded evaluator "
            "and match the direct sum (the deleted-duplicate guard)."
        )


# ---------------------------------------------------------------------------
# (4) Cross-path agreement: manifold and Bayesian adapters evaluate the SAME
#     factory coefficients, so they must agree with each other to fp precision
#     (modulo Bayesian float32 storage), independent of the reference.
# ---------------------------------------------------------------------------


@pytest.mark.physics
@pytest.mark.requires_jax
@pytest.mark.requires_db
@pytest.mark.parametrize("element,stage", SPECIES)
def test_manifold_and_bayesian_paths_agree(manifold_snapshot, bayesian_arrays, element, stage):
    """The two JAX adapters share one factory source => same U(T) within float32.

    A divergence here means one of the two snapshot builders stopped sourcing
    coefficients from ``partition_spec_for`` (the single seam).
    """
    from cflibs.inversion.solve.bayesian.atomic import partition_function
    from cflibs.plasma.partition import polynomial_partition_function_jax

    snap = manifold_snapshot
    species_to_idx = {key: i for i, key in enumerate(snap.species)}
    i = species_to_idx[(element, stage)]
    m_coeffs = np.asarray(snap.partition_coeffs)[i]
    m_tmin = float(np.asarray(snap.partition_t_min)[i])
    m_tmax = float(np.asarray(snap.partition_t_max)[i])
    m_g0 = float(np.asarray(snap.partition_g0)[i])

    el_idx, b_coeffs, b_tmin, b_tmax, b_g0 = bayesian_arrays
    ei, si = el_idx[element], stage - 1

    for T in LIBS_TEMPS:
        u_manifold = float(
            polynomial_partition_function_jax(T, m_coeffs, t_min=m_tmin, t_max=m_tmax, g0=m_g0)
        )
        u_bayes = float(
            partition_function(
                T,
                b_coeffs[ei, si],
                t_min=float(b_tmin[ei, si]),
                t_max=float(b_tmax[ei, si]),
                g0=float(b_g0[ei, si]),
            )
        )
        rel = abs(u_manifold - u_bayes) / max(u_manifold, 1e-30)
        # 1e-3 absorbs the Bayesian float32 coefficient storage; the two paths
        # are otherwise the identical guarded polynomial over identical coeffs.
        assert rel <= 1e-3, (
            f"Manifold vs Bayesian adapter disagree for {element} {stage} @ {T:.0f} K: "
            f"manifold={u_manifold:.6f} vs bayesian={u_bayes:.6f} ({rel:+.2%}). "
            "Both must source coefficients from the single partition_spec_for factory."
        )
