"""Unit tests for the single partition-function factory + its two adapters.

Covers the Wave-1 unification (composition-pipeline diagnosis 2026-06-03 § 2.1,
epic CF-LIBS-improved-8dn7): ``AtomicDatabase.partition_function_for`` /
``partition_spec_for`` is THE one source of U(T) coefficients + bounds + g0,
deriving them via the DIRECT-SUM fit over energy levels when the species has
tabulated levels and falling back to the stored polynomial otherwise.  Both the
CPU scalar adapter (``PolynomialPartitionFunctionProvider.at``) and the JAX
batched adapter (the static ``AtomicSnapshot`` arrays) are built from the SAME
spec, so they provably agree.

These tests run against the production DB (the fit is keyed on physical
``(element, sp_num)`` tuples; a fixture DB would tautologically pass).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cflibs.atomic.database import AtomicDatabase

KB_EV = 8.617333262145e-5

DB_PATH = Path(__file__).resolve().parent.parent / "ASD_da" / "libs_production.db"

# Workhorse iron-group species with complete energy_levels (direct-sum-fit path).
WITH_LEVELS = [("Fe", 1), ("Cr", 1), ("Ti", 1), ("Ni", 1), ("Cu", 1)]

# ps-LIBS band where the direct-sum fit must track the direct sum.
LIBS_TEMPS = (8000.0, 10000.0, 12000.0)


@pytest.fixture(scope="module")
def db():
    if not DB_PATH.exists():
        pytest.skip(f"production DB not present at {DB_PATH}")
    return AtomicDatabase(str(DB_PATH))


def _direct_sum(db: AtomicDatabase, element: str, stage: int, T_K: float) -> float:
    """Full U(T) = Σ g_i exp(-E_i/kT) for E_i < IP over the DB's energy_levels."""
    levels = db.get_energy_levels(element, stage)
    g = np.array([lev.g for lev in levels], dtype=np.float64)
    e = np.array([lev.energy_ev for lev in levels], dtype=np.float64)
    ip = db.get_ionization_potential(element, stage)
    mask = e < float(ip)
    if not mask.any():
        return 1.0
    return max(float(np.sum(g[mask] * np.exp(-e[mask] / (KB_EV * T_K)))), 1.0)


# ---------------------------------------------------------------------------
# (a) Species WITH levels -> direct-sum fit, reproducing the direct sum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("element,stage", WITH_LEVELS)
def test_factory_prefers_direct_sum_fit_for_species_with_levels(db, element, stage):
    """Factory derives direct-sum-fit coeffs (not the stored poly) when levels exist."""
    spec = db.partition_spec_for(element, stage)
    assert spec is not None
    assert (
        spec.from_direct_sum is True
    ), f"{element} {stage}: expected direct-sum fit, got source={spec.source!r}"
    assert spec.source == "direct_sum_fit"


@pytest.mark.parametrize("element,stage", WITH_LEVELS)
def test_provider_reproduces_direct_sum_within_tolerance(db, element, stage):
    """CPU scalar adapter reproduces the direct sum within ~10 % across 8000-12000 K.

    Direct summation over energy_levels is a strict lower bound on the true U, so
    a correct fit must be within tolerance AND must not undershoot it.
    """
    provider = db.partition_function_for(element, stage)
    assert provider is not None
    for T in LIBS_TEMPS:
        u_fit = float(provider.at(T))
        u_direct = _direct_sum(db, element, stage, T)
        rel = (u_fit - u_direct) / u_direct
        assert rel >= -0.02, (
            f"{element} {stage} @ {T:.0f} K: U_fit={u_fit:.3f} BELOW direct-sum "
            f"floor {u_direct:.3f} (physically impossible)"
        )
        assert abs(rel) < 0.10, (
            f"{element} {stage} @ {T:.0f} K: U_fit={u_fit:.3f}, "
            f"U_direct={u_direct:.3f}, rel={rel:+.2%} (>10%)"
        )


# ---------------------------------------------------------------------------
# (b) Level-less species -> stored polynomial fallback
# ---------------------------------------------------------------------------


def test_factory_falls_back_to_stored_poly_when_no_levels(db):
    """Species WITHOUT energy_levels use the stored polynomial (never fabricated)."""
    # Nb I/II have a stored partition_functions row but zero energy_levels.
    assert db.get_energy_levels("Nb", 1) == []
    spec = db.partition_spec_for("Nb", 1)
    assert spec is not None
    assert spec.from_direct_sum is False
    assert spec.source != "direct_sum_fit"
    stored = db.get_partition_coefficients("Nb", 1)
    assert stored is not None
    assert np.allclose(spec.coefficients, stored.coefficients, atol=1e-9)


def test_factory_returns_none_for_unknown_species(db):
    """No levels AND no stored row -> None (caller decides the default)."""
    assert db.partition_spec_for("Zz", 9) is None
    assert db.partition_function_for("Zz", 9) is None


# ---------------------------------------------------------------------------
# Bounds + g0 floor are applied by the adapter
# ---------------------------------------------------------------------------


def test_g0_floor_applied(db):
    """Below the clamped window the provider floors U at the ground-state weight g0."""
    provider = db.partition_function_for("Fe", 1)
    g0 = provider.g0
    assert g0 == pytest.approx(9.0)  # Fe I ground-state degeneracy (a 5D4) in the DB
    # At a tiny temperature (clamped to t_min) U cannot drop below g0.
    u_cold = float(provider.at(50.0))
    assert u_cold >= g0 - 1e-9


def test_bounds_clamp_high_temperature(db):
    """Above t_max the provider clamps T so U == U(t_max) (no runaway extrapolation)."""
    provider = db.partition_function_for("Fe", 1)
    _, t_max = provider.valid_range()
    u_at_max = float(provider.at(t_max))
    u_above = float(provider.at(t_max * 5.0))
    assert u_above == pytest.approx(u_at_max, rel=1e-9)


# ---------------------------------------------------------------------------
# Both adapters consume the SAME spec (CPU scalar == JAX snapshot arrays)
# ---------------------------------------------------------------------------


def test_snapshot_bakes_same_spec_as_cpu_adapter(db):
    """JAX batched adapter arrays match the CPU adapter's spec for the same species."""
    snap = db.snapshot(elements=["Fe"], wavelength_range=(200.0, 900.0))
    species = list(snap.species)
    spec = db.partition_spec_for("Fe", 1)
    idx = species.index(("Fe", 1))
    assert np.allclose(np.asarray(snap.partition_coeffs[idx]), spec.coefficients, atol=1e-5)
    assert float(np.asarray(snap.partition_g0[idx])) == pytest.approx(spec.g0)
    assert float(np.asarray(snap.partition_t_min[idx])) == pytest.approx(spec.t_min)
    assert float(np.asarray(snap.partition_t_max[idx])) == pytest.approx(spec.t_max)
