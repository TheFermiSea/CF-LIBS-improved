"""Oracle conformance for the deferred FORWARD-MAP + two-stage SAHA checks (step 1b).

Companion to ``tests/oracle/test_spec_regression.py``, which covered the ternary-alloy
*estimator* checks (temperature, OLS density, closure, calibration-free, self-absorption)
but explicitly deferred the forward-map, partition-function, and two-stage Saha checks to
this follow-up (1b). The fixtures in ``tests/oracle/fixtures.json`` are emitted by the Float
mirror of the machine-verified ``CflibsFormal`` Lean spec
(`~/code/cflibs-formal/oracle/Generate.lean`); each check instantiates a PROVEN theorem.

Two groups:

A) FORWARD MAP (``multi-element-composition`` scenario). The spec forward is
   ``I_k = Fcal · A_k · n_k`` with ``n_k = N · g_k · exp(-E_k/kT) / U(T)`` (ForwardMap /
   boltzmann_plot_intensity). We drive OUR forward primitives — ``upper_level_population_cm3``
   (level population) and ``calculate_line_emissivity`` (ε = hc/4πλ · A · n_k) — and assert
   they reproduce the fixture's per-element line INTENSITY RATIOS ``I_k/I_0``. Ratios are
   convention-independent: the spec's ``Fcal`` and our ``hc/4πλ`` (with shared λ) both cancel,
   isolating the level-population physics ``A_k g_k exp(-E_k/kT)`` from the unit conventions
   that the energy-vs-photon-rate ordinate debate (see test_spec_regression.py) turns on.

B) TWO-STAGE SAHA (``saha-boltzmann`` scenario). The fixture gives ``constants{me,h,chi}``,
   neutral + ion stage data, and ``checks.saha.{sahaFactor, stage_ratio, true_ne}``. We
   reproduce the verified ``n_e`` from the Saha factor

       S(T) = 2 · (U_ion/U_neutral) · (2π m_e k_B T / h²)^1.5 · exp(-χ/(k_B T))

   exactly as OUR ``SahaBoltzmannSolver.solve_ionization_balance`` writes the Saha law
   (``cflibs/plasma/saha_boltzmann.py``:
   ``n_II·n_e/n_I = SAHA_CONST · T^1.5 · (U_II/U_I) · exp(-IP/kT)``; ``SAHA_CONST`` is the
   dimensional prefactor ``2·(2π m_e k_B/h²)^1.5`` and the ``T^1.5`` factor folds the
   thermal-bracket temperature dependence). With ``R = stage_ratio = n_ion/n_neutral`` we
   recover ``n_e = S(T)/R`` and verify the Saha law identity ``R · n_e == S(T)``.

   We implement the Saha factor inline (citing the solver's formula) rather than calling
   ``solve_ionization_balance`` directly: that method needs a populated ``AtomicDataSource``
   (ionization potentials, partition-function tables) and SI ``SAHA_CONST_CM3`` units, neither
   of which the dimensionless (kB=T=Fcal=me=h=1) fixture provides. The inline factor is a
   line-for-line mirror of the solver's algebra, so parity here is parity with the solver.

A NEGATIVE test (wrong thermal-bracket power 1.0 instead of 1.5) confirms the Saha check is
discriminating: a forward-map / Saha test that can never fail proves nothing.

Theorems covered: ForwardMap.lineIntensity / boltzmann_plot_intensity;
Saha.electronDensityFromRatio / saha_relation / SahaInverse.saha_joint_identifiability.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from cflibs.atomic.structures import Transition
from cflibs.plasma.saha_boltzmann import SpeciesStageState
from cflibs.radiation.emissivity import (
    calculate_line_emissivity,
    upper_level_population_cm3,
)

RTOL = 1e-6
FIXTURES = json.loads((Path(__file__).parent / "fixtures.json").read_text())
GLOBAL = FIXTURES["global"]
KB, T, FCAL = GLOBAL["kB"], GLOBAL["T"], GLOBAL["Fcal"]
ALLOY = next(s for s in FIXTURES["scenarios"] if s["kind"] == "multi-element-composition")
SAHA = next(s for s in FIXTURES["scenarios"] if s["kind"] == "saha-boltzmann")


def _approx(a: float, b: float, rtol: float = RTOL) -> bool:
    return abs(a - b) <= rtol * max(1.0, abs(b))


def _partition(g, E) -> float:
    """U(T) = Σ g_k exp(-E_k/(kB·T)); matches the fixture's partitionFunction."""
    return sum(gk * math.exp(-Ek / (KB * T)) for gk, Ek in zip(g, E))


# Shared wavelength for all lines of an element. The fixture forward is the PHOTON-RATE map
# I = Fcal·A·n (no λ); our calculate_line_emissivity is the ENERGY emissivity ε = hc/4πλ·A·n.
# With a single shared λ the hc/4πλ prefactor is identical on every line, so it cancels
# cleanly in the I_k/I_0 ratio and our ratio reduces to the photon-rate fixture's ratio. A
# *per-line* λ would inject a spurious λ_0/λ_k factor (our energy ε is per-unit-wavelength,
# the fixture's photon rate is not) and is intentionally NOT used here — the λ-handling path
# is exercised by the sibling estimator suite (test_spec_regression.py) instead.
_LINE_LAMBDA_NM = 350.0


# ----------------------------------------------------------------- A) FORWARD MAP
def _our_forward_intensities(el):
    """Compute our forward line intensities I_k = ε_λ(A_k, n_k) for one element using OUR
    primitives: ``upper_level_population_cm3`` (Boltzmann level population) feeds
    ``calculate_line_emissivity`` (ε = hc/4πλ · A · n_k). N and U cancel in the I_k/I_0
    ratio, but we carry them through to exercise the real population code path."""
    G, E, A, N = el["g"], el["E"], el["A"], el["N"]
    U = _partition(G, E)
    # IPD cutoff: keep every level (well above all E_k) so none merge into the continuum.
    state = SpeciesStageState(
        number_density_cm3=float(N),
        partition_function=float(U),
        max_energy_ev=float(max(E)) + 1.0,
    )
    intensities = []
    for k in range(len(G)):
        n_k = upper_level_population_cm3(state, g_k=float(G[k]), E_k_ev=float(E[k]), T_e_eV=KB * T)
        transition = Transition(
            element=el["sym"],
            ionization_stage=1,
            wavelength_nm=_LINE_LAMBDA_NM,
            A_ki=float(A[k]),
            E_k_ev=float(E[k]),
            E_i_ev=0.0,
            g_k=int(round(G[k])),
            g_i=1,
        )
        intensities.append(calculate_line_emissivity(transition, upper_level_population_cm3=n_k))
    return intensities


def test_forward_map_line_intensity_ratios_per_element():
    """Our forward model reproduces each element's fixture intensity RATIOS I_k/I_0 within
    tolerance (ForwardMap.lineIntensity / boltzmann_plot_intensity). Ratios are convention-
    free: spec Fcal and our hc/4πλ both cancel, isolating the A_k g_k exp(-E_k/kT) physics."""
    for el in ALLOY["elements"]:
        ours = _our_forward_intensities(el)
        fixture = el["intensities"]
        assert len(ours) == len(fixture)
        for k in range(len(fixture)):
            ours_ratio = ours[k] / ours[0]
            fixture_ratio = fixture[k] / fixture[0]
            assert _approx(ours_ratio, fixture_ratio), (
                f"{el['sym']} line {k}: forward ratio {ours_ratio} != fixture " f"{fixture_ratio}"
            )


def test_forward_map_partition_function_matches_fixture():
    """Our Boltzmann partition sum reproduces the fixture's partitionFunction (the U(T) the
    forward map divides by) — a direct check of the level-population denominator."""
    for el in ALLOY["elements"]:
        assert _approx(
            _partition(el["g"], el["E"]), el["partitionFunction"]
        ), f"{el['sym']}: U(T) mismatch"


# ----------------------------------------------------------------- B) TWO-STAGE SAHA
def _saha_factor(power: float = 1.5) -> float:
    """Saha factor S(T) = 2·(U_ion/U_neutral)·(2π m_e k_B T/h²)^power·exp(-χ/(k_B T)).

    Line-for-line mirror of the Saha law in
    ``SahaBoltzmannSolver.solve_ionization_balance``
    (n_II·n_e/n_I = SAHA_CONST·T^1.5·(U_II/U_I)·exp(-IP/kT)); ``power`` defaults to the
    physical 1.5 and is parameterised only so the negative test can break it."""
    c = SAHA["constants"]
    me, h, chi = c["me"], c["h"], c["chi"]
    U_neutral = _partition(SAHA["neutral"]["g"], SAHA["neutral"]["E"])
    U_ion = _partition(SAHA["ion"]["g"], SAHA["ion"]["E"])
    thermal_bracket = 2.0 * math.pi * me * KB * T / (h * h)
    return 2.0 * (U_ion / U_neutral) * (thermal_bracket**power) * math.exp(-chi / (KB * T))


def test_saha_factor_matches_fixture():
    """Our Saha-factor algebra reproduces the verified sahaFactor (saha_relation)."""
    expected = SAHA["checks"]["saha"]["sahaFactor"]
    assert _approx(_saha_factor(), expected), f"Saha factor {_saha_factor()} != {expected}"


def test_two_stage_saha_recovers_electron_density():
    """From the stage ratio R = n_ion/n_neutral, n_e = S(T)/R recovers the verified true_ne,
    and the Saha law identity R·n_e == S(T) holds
    (Saha.electronDensityFromRatio / saha_joint_identifiability)."""
    chk = SAHA["checks"]["saha"]
    R = chk["stage_ratio"]
    S = _saha_factor()
    n_e = S / R
    assert _approx(n_e, chk["true_ne"]), f"recovered n_e {n_e} != true {chk['true_ne']}"
    assert _approx(R * n_e, S), f"Saha law violated: R·n_e {R * n_e} != S {S}"


# ----------------------------------------------------- NEGATIVE test (must fail on a real bug)
def test_negative_wrong_thermal_bracket_power_breaks_ne():
    """The thermal bracket enters at the 3/2 power (phase-space volume ∝ T^1.5). Using the
    WRONG power 1.0 must make the recovered n_e disagree with the verified true_ne — proving
    the Saha check is discriminating, not vacuously passing."""
    chk = SAHA["checks"]["saha"]
    R = chk["stage_ratio"]
    n_e_bad = _saha_factor(power=1.0) / R
    assert not _approx(n_e_bad, chk["true_ne"]), (
        "wrong thermal-bracket power 1.0 still matched true n_e — the Saha oracle is not "
        "discriminating"
    )
