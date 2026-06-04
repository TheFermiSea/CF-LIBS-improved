"""
Regression tests for the τ update inside
``SelfAbsorptionCorrector._apply_recursive_correction``.

These tests pin down the Wave-1 / B4 fix from
``docs/architecture/2026-05-27-physics-audit.md``: the recursive
self-absorption correction must recompute τ from the *plasma state*
each iteration (Bulajic, Corsi, Cristoforetti, Legnaioli, Palleschi,
Salvetti, Tognoni 2002 — *Spectrochim. Acta B* 57 339,
doi:10.1016/S0584-8547(01)00398-6, §3), and **never** from the observed
intensity via ``τ ← τ · I_corr/I_obs``. The latter was a dimensionally
wrong feedback hack that diverged monotonically in the optically-thick
regime, bounded only by ``max_iterations``.

Each test below targets a different aspect of the fix:

1. ``test_tau_matches_estimate_optical_depth``: the final τ returned by
   ``_apply_recursive_correction`` is *exactly* the value
   ``_estimate_optical_depth`` would give for the same plasma state —
   not the seed, not a scaled version.
2. ``test_tau_does_not_match_buggy_scaling``: under conditions where the
   bug would have produced a clearly different value, the fix returns
   the physics-correct τ instead.
3. ``test_converges_within_max_iterations``: with frozen plasma state τ
   is a constant of motion → the loop converges in ≤ ``max_iterations``
   (in fact in 1 iteration).
4. ``test_tau_invariant_under_intensity_rescaling``: scaling I_obs by
   10× must NOT change τ. The bug scaled τ by ~10× through the
   ``I_new/I_obs`` factor — the fix is intensity-invariant because τ
   depends only on plasma state + atomic data.
"""

from __future__ import annotations

import pytest

from cflibs.inversion.physics.boltzmann import LineObservation
from cflibs.inversion.physics.self_absorption import (
    SelfAbsorptionCorrector,
    _escape_factor,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def plasma_state():
    """LIBS-typical plasma state with major Fe content (drives τ > 1)."""
    return {
        "temperature_K": 10_000.0,
        "concentrations": {"Fe": 0.5},
        "total_n_cm3": 1.0e18,
        "partition_funcs": {"Fe": 25.0},
        "E_i_ev": 0.0,
    }


@pytest.fixture
def thick_line():
    """Fe I resonance-like line that lands in the optically-thick regime."""
    return LineObservation(
        wavelength_nm=400.0,
        intensity=1000.0,
        intensity_uncertainty=20.0,
        element="Fe",
        ionization_stage=1,
        E_k_ev=3.0,
        g_k=9,
        A_ki=1.0e8,
    )


@pytest.fixture
def corrector():
    return SelfAbsorptionCorrector(
        optical_depth_threshold=0.1,
        mask_threshold=1000.0,  # disable masking so we always hit the recursive path
        max_iterations=5,
        convergence_tolerance=1.0e-3,
        plasma_length_cm=0.1,
    )


# -----------------------------------------------------------------------------
# (a) Final τ matches the physics estimate
# -----------------------------------------------------------------------------


def test_tau_matches_estimate_optical_depth(corrector, thick_line, plasma_state):
    """
    Final τ returned by ``_apply_recursive_correction`` must equal the
    value ``_estimate_optical_depth`` produces for the same plasma state.

    Within a single call the inputs to ``_estimate_optical_depth`` do
    not change between iterations, so τ is a constant of motion and the
    loop should converge immediately to that constant.
    """
    tau_physics = corrector._estimate_optical_depth(
        thick_line,
        plasma_state["temperature_K"],
        plasma_state["concentrations"],
        plasma_state["total_n_cm3"],
        plasma_state["partition_funcs"],
        plasma_state["E_i_ev"],
    )
    assert tau_physics > 0.1, (
        "Test setup error: expected the resonance line to be optically "
        f"thick (τ > 0.1), got τ = {tau_physics:.3e}."
    )

    result = corrector._apply_recursive_correction(
        thick_line,
        tau_physics,
        plasma_state["temperature_K"],
        plasma_state["concentrations"],
        plasma_state["total_n_cm3"],
        plasma_state["partition_funcs"],
        plasma_state["E_i_ev"],
    )

    assert result.optical_depth == pytest.approx(tau_physics, rel=1.0e-9), (
        "_apply_recursive_correction must return the τ predicted by "
        "_estimate_optical_depth for the current plasma state "
        "(Bulajic 2002 §3)."
    )


# -----------------------------------------------------------------------------
# (b) Final τ is NOT the buggy I_new/I_obs scaling
# -----------------------------------------------------------------------------


def test_tau_does_not_match_buggy_scaling(corrector, thick_line, plasma_state):
    """
    Under the historical bug ``tau ← tau · (I_new / I_obs)`` ≡
    ``tau ← tau / f(τ)``, which for an optically-thick line with
    ``max_iterations = 5`` would scale τ by ``(1/f(τ))^k`` for some k>0
    — a divergent geometric series.

    We compute that buggy projection explicitly and assert the actual τ
    returned by the fixed implementation is *not* it.
    """
    tau_seed = corrector._estimate_optical_depth(
        thick_line,
        plasma_state["temperature_K"],
        plasma_state["concentrations"],
        plasma_state["total_n_cm3"],
        plasma_state["partition_funcs"],
        plasma_state["E_i_ev"],
    )
    assert tau_seed > 1.0, (
        "Test setup error: bug only diverges visibly in the optically "
        f"thick regime (τ > 1), got τ = {tau_seed:.3e}."
    )

    # Simulate the bug for at least 2 iterations to get a value that is
    # provably distinct from tau_seed.
    tau_buggy = tau_seed
    for _ in range(2):
        f_tau = _escape_factor(tau_buggy)
        I_new = thick_line.intensity / f_tau
        tau_buggy = tau_buggy * (I_new / thick_line.intensity)

    # Sanity check on the simulation — the bug must produce a τ that
    # differs from the seed by far more than the convergence tolerance.
    assert tau_buggy > tau_seed * 1.5, (
        "Bug-simulation sanity check failed: expected the buggy update "
        f"to amplify τ noticeably, got τ_buggy = {tau_buggy:.3e} vs "
        f"τ_seed = {tau_seed:.3e}."
    )

    result = corrector._apply_recursive_correction(
        thick_line,
        tau_seed,
        plasma_state["temperature_K"],
        plasma_state["concentrations"],
        plasma_state["total_n_cm3"],
        plasma_state["partition_funcs"],
        plasma_state["E_i_ev"],
    )

    # The fixed τ must equal tau_seed (the physics value) — emphatically
    # not the runaway tau_buggy value.
    assert result.optical_depth == pytest.approx(tau_seed, rel=1.0e-9)
    assert abs(result.optical_depth - tau_buggy) > 0.1 * tau_seed, (
        "Final τ matches the buggy I_new/I_obs scaling — the τ-update " "regression has reappeared."
    )


# -----------------------------------------------------------------------------
# (c) Convergence within max_iterations
# -----------------------------------------------------------------------------


def test_converges_within_max_iterations(corrector, thick_line, plasma_state):
    """With frozen plasma state τ is constant → converge in 1 iteration."""
    tau_seed = corrector._estimate_optical_depth(
        thick_line,
        plasma_state["temperature_K"],
        plasma_state["concentrations"],
        plasma_state["total_n_cm3"],
        plasma_state["partition_funcs"],
        plasma_state["E_i_ev"],
    )
    result = corrector._apply_recursive_correction(
        thick_line,
        tau_seed,
        plasma_state["temperature_K"],
        plasma_state["concentrations"],
        plasma_state["total_n_cm3"],
        plasma_state["partition_funcs"],
        plasma_state["E_i_ev"],
    )
    assert result.iterations <= corrector.max_iterations
    # Stronger claim: with constant inputs the loop should converge on
    # the very first iteration.
    assert result.iterations == 1


# -----------------------------------------------------------------------------
# (d) τ is intensity-invariant — the buggy scaling would NOT be
# -----------------------------------------------------------------------------


def test_tau_invariant_under_intensity_rescaling(corrector, thick_line, plasma_state):
    """
    Rescaling I_obs by 10× must not change τ. The bug scaled τ through
    the ``I_new / I_obs`` factor — but I_new ∝ I_obs, so the bug had a
    different vulnerability: the *progression* depended on the absolute
    intensity through ``f(τ)`` only. The cleanest invariance check is
    the simpler one: the physics-defined τ is a property of the plasma
    state and the atomic data — it cannot depend on the observed
    intensity at all.
    """
    tau_baseline = corrector._estimate_optical_depth(
        thick_line,
        plasma_state["temperature_K"],
        plasma_state["concentrations"],
        plasma_state["total_n_cm3"],
        plasma_state["partition_funcs"],
        plasma_state["E_i_ev"],
    )
    result_baseline = corrector._apply_recursive_correction(
        thick_line,
        tau_baseline,
        plasma_state["temperature_K"],
        plasma_state["concentrations"],
        plasma_state["total_n_cm3"],
        plasma_state["partition_funcs"],
        plasma_state["E_i_ev"],
    )

    # Rebuild the same line with 10× the intensity (e.g., a stronger
    # collection-geometry coupling). Plasma state, atomic data, and
    # path length are unchanged → physics τ must be identical.
    thick_line_loud = LineObservation(
        wavelength_nm=thick_line.wavelength_nm,
        intensity=thick_line.intensity * 10.0,
        intensity_uncertainty=thick_line.intensity_uncertainty * 10.0,
        element=thick_line.element,
        ionization_stage=thick_line.ionization_stage,
        E_k_ev=thick_line.E_k_ev,
        g_k=thick_line.g_k,
        A_ki=thick_line.A_ki,
    )

    result_loud = corrector._apply_recursive_correction(
        thick_line_loud,
        tau_baseline,
        plasma_state["temperature_K"],
        plasma_state["concentrations"],
        plasma_state["total_n_cm3"],
        plasma_state["partition_funcs"],
        plasma_state["E_i_ev"],
    )

    assert result_loud.optical_depth == pytest.approx(result_baseline.optical_depth, rel=1.0e-9), (
        "Final τ depends on observed intensity — that is the bug. τ "
        "is a property of the plasma column (Bulajic 2002 §3); it "
        "must be invariant under intensity rescaling."
    )


# -----------------------------------------------------------------------------
# (e) Correction-factor cap bounds the intensity boost without changing τ
# -----------------------------------------------------------------------------


def test_correction_tau_cap_bounds_correction_factor(thick_line, plasma_state):
    """With ``correction_tau_cap`` set, the *correction factor* is clamped.

    The escape-factor boost ``I/f(τ) ≈ τ`` grows without bound as ``τ → ∞`` and
    the Doppler-core curve-of-growth model loses validity beyond τ ~ 5-10
    (El Sherbini 2005). A configured cap must bound the intensity boost to
    ``≈ τ_cap`` while still *reporting* the true (uncapped) optical depth — this
    is what keeps the iterative-solver wiring stable when the absorbing column
    density is only known to an order of magnitude.
    """
    cap = 10.0
    capped = SelfAbsorptionCorrector(
        optical_depth_threshold=0.1,
        mask_threshold=1.0e9,
        max_iterations=5,
        convergence_tolerance=1.0e-3,
        plasma_length_cm=0.1,
        correction_tau_cap=cap,
    )
    tau = capped._estimate_optical_depth(
        thick_line,
        plasma_state["temperature_K"],
        plasma_state["concentrations"],
        plasma_state["total_n_cm3"],
        plasma_state["partition_funcs"],
        plasma_state["E_i_ev"],
    )
    assert tau > cap, "Test setup error: need τ above the cap to exercise it."

    result = capped._apply_recursive_correction(
        thick_line,
        tau,
        plasma_state["temperature_K"],
        plasma_state["concentrations"],
        plasma_state["total_n_cm3"],
        plasma_state["partition_funcs"],
        plasma_state["E_i_ev"],
    )

    # Reported τ is the TRUE plasma value, not the cap.
    assert result.optical_depth == pytest.approx(tau, rel=1.0e-9)
    # But the applied correction factor is bounded by the capped escape factor.
    expected_factor = _escape_factor(cap)
    assert result.corrected_intensity == pytest.approx(
        thick_line.intensity / expected_factor, rel=1.0e-9
    )
    # And it is far below the unbounded ``I/f(τ_true)`` the uncapped path gives.
    unbounded = thick_line.intensity / _escape_factor(tau)
    assert result.corrected_intensity < 0.5 * unbounded


def test_no_cap_preserves_uncapped_correction(thick_line, plasma_state):
    """``correction_tau_cap=None`` (default) leaves the historical behaviour."""
    uncapped = SelfAbsorptionCorrector(
        optical_depth_threshold=0.1,
        mask_threshold=1.0e9,
        max_iterations=5,
        convergence_tolerance=1.0e-3,
        plasma_length_cm=0.1,
        correction_tau_cap=None,
    )
    tau = uncapped._estimate_optical_depth(
        thick_line,
        plasma_state["temperature_K"],
        plasma_state["concentrations"],
        plasma_state["total_n_cm3"],
        plasma_state["partition_funcs"],
        plasma_state["E_i_ev"],
    )
    result = uncapped._apply_recursive_correction(
        thick_line,
        tau,
        plasma_state["temperature_K"],
        plasma_state["concentrations"],
        plasma_state["total_n_cm3"],
        plasma_state["partition_funcs"],
        plasma_state["E_i_ev"],
    )
    assert result.corrected_intensity == pytest.approx(
        thick_line.intensity / _escape_factor(tau), rel=1.0e-9
    )
