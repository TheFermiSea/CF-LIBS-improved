"""Unit tests for the Stewart-Pyatt IPD model (``cflibs.plasma.ipd``).

These tests assert the *physical properties* named in the acceptance criteria,
using synthetic inputs only (no real atomic database):

1. ``Δχ`` increases monotonically with ``n_e``.
2. ``Δχ`` reduces to the Debye-Hückel form in the weak-coupling limit.
3. The Stewart-Pyatt model is selectable in the Saha balance and turning it on
   shifts the ion/neutral ratio in the physically correct direction (more
   ionization) relative to no IPD.
4. Defaults are unchanged: the solver still uses Debye-Hückel when no
   ``ipd_model`` is passed.

References: Stewart & Pyatt (1966) ApJ 144, 1203; Crowley (2014) HEDP 13, 84.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pytest

from cflibs.plasma.ipd import (
    StewartPyattIPD,
    debye_length_cm,
    ion_sphere_radius_cm,
    make_ipd_model,
    stewart_pyatt_lowering,
)
from cflibs.plasma.partition import ionization_potential_depression
from cflibs.plasma.saha_boltzmann import DebyeHuckelIPD, SahaBoltzmannSolver

# ---------------------------------------------------------------------------
# Property 1: monotonic increase with n_e
# ---------------------------------------------------------------------------


def test_delta_chi_monotonic_increasing_in_ne():
    """Δχ_SP strictly increases with electron density at fixed temperature."""
    T_K = 12000.0
    n_e_grid = np.logspace(13.0, 22.0, 60)
    deltas = [stewart_pyatt_lowering(ne, T_K) for ne in n_e_grid]
    diffs = np.diff(deltas)
    assert np.all(diffs > 0.0), "Δχ_SP must increase monotonically with n_e"
    # Endpoints: strictly positive and meaningfully larger at high density.
    assert deltas[0] > 0.0
    assert deltas[-1] > deltas[0]


def test_delta_chi_nonnegative_and_zero_for_bad_inputs():
    """Δχ is non-negative; non-physical inputs return exactly 0."""
    assert stewart_pyatt_lowering(1e17, 12000.0) > 0.0
    assert stewart_pyatt_lowering(0.0, 12000.0) == 0.0
    assert stewart_pyatt_lowering(1e17, 0.0) == 0.0
    assert stewart_pyatt_lowering(-1.0, 12000.0) == 0.0


# ---------------------------------------------------------------------------
# Property 2: weak-coupling limit reduces to Debye-Hückel
# ---------------------------------------------------------------------------


def test_weak_coupling_limit_reduces_to_debye_huckel():
    """As λ_D ≫ R_0 (low density / high T), Δχ_SP → the canonical DH Δχ.

    The package's canonical Debye-Hückel IPD is
    ``ionization_potential_depression`` (Z=1 for the neutral→I prefactor).
    """
    T_K = 15000.0
    # Low densities push λ_D / R_0 large -> deep weak-coupling regime.
    for n_e in (1e12, 1e13, 1e14):
        x = debye_length_cm(n_e, T_K) / ion_sphere_radius_cm(n_e)
        assert x > 3.0, "expected weak-coupling regime (λ_D/R_0 >> 1)"
        sp = stewart_pyatt_lowering(n_e, T_K, z_net=0)
        dh = ionization_potential_depression(n_e, T_K, Z=1)
        # Converges to DH; the residual is O((R_0/λ_D)^3) and shrinks with n_e.
        assert sp == pytest.approx(dh, rel=2e-2), f"n_e={n_e:.0e}: SP {sp} should approach DH {dh}"

    # The convergence must TIGHTEN as the coupling weakens further.
    err_lo = abs(stewart_pyatt_lowering(1e12, T_K) - ionization_potential_depression(1e12, T_K))
    err_hi = abs(stewart_pyatt_lowering(1e15, T_K) - ionization_potential_depression(1e15, T_K))
    assert err_lo < err_hi, "SP→DH residual should shrink in the weak-coupling direction"


def test_softens_debye_huckel_at_high_density():
    """SP lies below the bare DH value everywhere (softens the DH divergence)."""
    T_K = 12000.0
    for n_e in np.logspace(15.0, 22.0, 30):
        sp = stewart_pyatt_lowering(n_e, T_K)
        dh = ionization_potential_depression(n_e, T_K, Z=1)
        assert sp <= dh + 1e-12, f"n_e={n_e:.0e}: SP {sp} must not exceed DH {dh}"


def test_strong_coupling_limit_approaches_ion_sphere():
    """As λ_D ≪ R_0 (high density), Δχ_SP → the ion-sphere value 3(z+1)e²/2R_0.

    Built from the same CGS constants the module uses, so this is an internal
    self-consistency check of the strong-coupling asymptote.
    """
    from cflibs.plasma.ipd import _E_ESU, _ERG_TO_EV

    T_K = 12000.0
    n_e = 1e23  # very high density -> deep ion-sphere regime
    R0 = ion_sphere_radius_cm(n_e)
    x = debye_length_cm(n_e, T_K) / R0
    assert x < 0.3, "expected strong-coupling regime (λ_D/R_0 << 1)"
    ion_sphere = 1.5 * 1.0 * _E_ESU**2 / R0 * _ERG_TO_EV  # z_net=0 -> (z+1)=1
    sp = stewart_pyatt_lowering(n_e, T_K, z_net=0)
    assert sp == pytest.approx(ion_sphere, rel=5e-2)


# ---------------------------------------------------------------------------
# Selectability + factory
# ---------------------------------------------------------------------------


def test_make_ipd_model_selectable():
    """The factory selects SP, DH, or 'none' (default) by name."""
    assert isinstance(make_ipd_model("stewart_pyatt"), StewartPyattIPD)
    assert isinstance(make_ipd_model("debye_huckel"), DebyeHuckelIPD)
    assert make_ipd_model("none") is None
    assert make_ipd_model(None) is None
    with pytest.raises(ValueError):
        make_ipd_model("not_a_model")


def test_protocol_signature():
    """StewartPyattIPD satisfies the IPDModel protocol call signature."""
    model = StewartPyattIPD()
    val = model.calculate_lowering(1e17, 12000.0)
    assert isinstance(val, float) and val > 0.0


# ---------------------------------------------------------------------------
# Property 3: turning IPD on shifts ion/neutral ratio toward MORE ionization
# ---------------------------------------------------------------------------


class _ZeroIPD:
    """An IPD model that returns no lowering (Δχ = 0) — the IPD-off baseline."""

    def calculate_lowering(self, n_e_cm3: float, T_K: float) -> float:  # noqa: D401
        return 0.0


class _FakeEnergyLevel:
    def __init__(self, energy_ev: float, g: float) -> None:
        self.energy_ev = energy_ev
        self.g = g


class _FakeAtomicDB:
    """Minimal synthetic AtomicDataSource for a one-element two-stage Saha test.

    Provides just enough surface for ``SahaBoltzmannSolver.solve_ionization_balance``:
    ``get_ionization_potential`` and ``get_energy_levels``.  No real database,
    no ``partition_function_for`` (so the solver uses its direct-sum fallback
    over these levels), keeping the test self-contained and fast.
    """

    db_path = "fake://ipd-test"

    def __init__(self, ip_I: float) -> None:
        self._ip = {1: ip_I}  # only first IP defined -> two-stage balance
        # A couple of low-lying levels per stage so U is well-defined.
        self._levels: Dict[int, List[_FakeEnergyLevel]] = {
            1: [_FakeEnergyLevel(0.0, 2.0), _FakeEnergyLevel(1.0, 4.0)],
            2: [_FakeEnergyLevel(0.0, 1.0), _FakeEnergyLevel(2.0, 6.0)],
        }

    def get_ionization_potential(self, element: str, stage: int) -> Optional[float]:
        return self._ip.get(stage)

    def get_energy_levels(self, element: str, stage: int) -> List[_FakeEnergyLevel]:
        return self._levels.get(stage, [])


def _ion_neutral_ratio(ipd_model, ip_I: float, T_e_eV: float, n_e: float) -> float:
    db = _FakeAtomicDB(ip_I=ip_I)
    solver = SahaBoltzmannSolver(db, ipd_model=ipd_model)
    stages = solver.solve_ionization_balance("X", T_e_eV, n_e, total_density_cm3=1.0)
    n_I = stages.get(1, 0.0)
    n_II = stages.get(2, 0.0)
    assert n_I > 0.0
    return n_II / n_I


def test_ipd_on_increases_ionization():
    """Turning IPD on (Δχ>0) lowers the effective IP -> MORE ionization.

    The Saha ratio n_II/n_I ∝ exp(-(IP - Δχ)/kT); a positive Δχ from the
    Stewart-Pyatt model must therefore *increase* n_II/n_I relative to the
    Δχ=0 (IPD-off) baseline. This is the physically-correct direction.
    """
    ip_I = 8.0  # eV
    T_e_eV = 1.0
    n_e = 1e18  # cm^-3, where Δχ is appreciable

    ratio_off = _ion_neutral_ratio(_ZeroIPD(), ip_I, T_e_eV, n_e)
    ratio_sp = _ion_neutral_ratio(StewartPyattIPD(), ip_I, T_e_eV, n_e)

    assert ratio_sp > ratio_off, "Stewart-Pyatt IPD must increase the ion/neutral ratio"

    # Quantitative cross-check: the shift equals exp(+Δχ/kT) on the Saha
    # exponent (partition ratio and prefactor are common to both runs).
    delta_chi = StewartPyattIPD().calculate_lowering(n_e, T_e_eV / 8.617333262e-5)
    expected_factor = math.exp(delta_chi / T_e_eV)
    assert (ratio_sp / ratio_off) == pytest.approx(expected_factor, rel=1e-6)


def test_default_solver_unchanged_is_debye_huckel():
    """Default solver (no ipd_model) still uses Debye-Hückel — defaults unchanged."""
    db = _FakeAtomicDB(ip_I=8.0)
    default_solver = SahaBoltzmannSolver(db)
    assert isinstance(default_solver.ipd_model, DebyeHuckelIPD)

    # And the default path is bit-for-bit identical to an explicit DH model.
    explicit_dh = SahaBoltzmannSolver(db, ipd_model=DebyeHuckelIPD())
    r_default = default_solver.solve_ionization_balance("X", 1.0, 1e18, 1.0)
    r_dh = explicit_dh.solve_ionization_balance("X", 1.0, 1e18, 1.0)
    assert r_default.keys() == r_dh.keys()
    for stage in r_default:
        assert r_default[stage] == pytest.approx(r_dh[stage], rel=1e-12)
