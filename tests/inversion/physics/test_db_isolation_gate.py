"""Tests for the opt-in DB-isolation gate (salvage of WIP 47133ac).

Covers:
* ``compute_db_isolation_weights`` down-weights a line blended by an unresolved
  atomic-DB neighbor while leaving isolated lines at weight 1.0,
* the element-agnostic cross-element blend and the sparse-element floor,
* the disabled/degenerate fast-exits, and
* the solver wiring (``IterativeCFLIBSSolver._apply_db_isolation_gate``): a strict
  no-op object pass-through when the gate is OFF, and sigma inflation by
  ``1/sqrt(w)`` on blended lines when ON (default path stays byte-identical).

Fast: no real DB, no XLA, no full solve.
"""

from __future__ import annotations

import math
from typing import List, Optional

import pytest

from cflibs.atomic.structures import Transition
from cflibs.inversion.physics.boltzmann import LineObservation
from cflibs.inversion.physics.line_selection import compute_db_isolation_weights


class FakeDB:
    """Minimal duck-typed atomic DB returning canned transitions by window."""

    def __init__(self, transitions: List[Transition]):
        self._t = transitions

    def get_transitions(
        self,
        element: str,
        ionization_stage: Optional[int] = None,
        wavelength_min: Optional[float] = None,
        wavelength_max: Optional[float] = None,
        min_relative_intensity: Optional[float] = None,
    ) -> List[Transition]:
        out = []
        for t in self._t:
            if t.element != element:
                continue
            if ionization_stage is not None and t.ionization_stage != ionization_stage:
                continue
            if wavelength_min is not None and t.wavelength_nm < wavelength_min:
                continue
            if wavelength_max is not None and t.wavelength_nm > wavelength_max:
                continue
            out.append(t)
        return out


def _tr(element: str, wl: float, e_k: float = 4.0, a_ki: float = 1e8, g_k: int = 5) -> Transition:
    return Transition(
        element=element,
        ionization_stage=1,
        wavelength_nm=wl,
        A_ki=a_ki,
        E_k_ev=e_k,
        E_i_ev=1.0,
        g_k=g_k,
        g_i=3,
        is_resonance=False,
        aki_uncertainty=0.05,
    )


def _obs(element: str, wl: float, e_k: float = 4.0, a_ki: float = 1e8, g_k: int = 5) -> LineObservation:
    return LineObservation(
        wavelength_nm=wl,
        intensity=1000.0,
        intensity_uncertainty=10.0,
        element=element,
        ionization_stage=1,
        E_k_ev=e_k,
        g_k=g_k,
        A_ki=a_ki,
        aki_uncertainty=0.05,
    )


# --- compute_db_isolation_weights ------------------------------------------------


def test_blended_line_downweighted_isolated_lines_untouched():
    """A dense element line with a 2x DB neighbor in-window is down-weighted;
    its isolated siblings (element not protected, count > floor) stay at 1.0."""
    # Fe has 4 lines -> count 4 > min_lines_per_element (3): not protected.
    obs = [
        _obs("Fe", 500.0),  # blended by an Fe DB neighbor at 500.06
        _obs("Fe", 510.0),  # isolated
        _obs("Fe", 520.0),  # isolated
        _obs("Fe", 530.0),  # isolated
    ]
    db = FakeDB([_tr("Fe", 500.06, a_ki=2e8)])  # 2x candidate emissivity, same E_k/g

    weights = compute_db_isolation_weights(
        obs, db, fwhm_nm=0.1, window_n_fwhm=1.5, blend_fraction=0.15
    )

    # ratio = (2e8*5) / (1e8*5) = 2.0 -> w = 0.15 / 2.0 = 0.075 (no floor: count 4 > 3)
    assert weights[id(obs[0])] == pytest.approx(0.075, rel=1e-9)
    for o in obs[1:]:
        assert weights[id(o)] == 1.0


def test_cross_element_blend_is_element_agnostic():
    """A Cu candidate blended by an Fe DB neighbor is down-weighted: the gate
    triggers on local spectral density, not on element identity."""
    obs = [
        _obs("Cu", 400.0),
        _obs("Cu", 410.0),
        _obs("Cu", 420.0),
        _obs("Cu", 430.0),  # 4 Cu lines -> not protected
    ]
    db = FakeDB([_tr("Fe", 400.05, a_ki=1e8)])  # equal emissivity Fe neighbor -> ratio 1.0
    weights = compute_db_isolation_weights(
        obs, db, elements=["Cu", "Fe"], fwhm_nm=0.1, window_n_fwhm=1.5, blend_fraction=0.15
    )
    # ratio == 1.0 -> w = 0.15
    assert weights[id(obs[0])] == pytest.approx(0.15, rel=1e-9)
    for o in obs[1:]:
        assert weights[id(o)] == 1.0


def test_sparse_element_protected_by_floor():
    """A blended line of a sparse element (count <= floor) is clamped to >= 0.5."""
    obs = [_obs("Cu", 500.0)]  # single Cu line -> count 1 <= floor 3 -> protected
    db = FakeDB([_tr("Fe", 500.06, a_ki=2e8)])
    weights = compute_db_isolation_weights(
        obs,
        db,
        elements=["Cu", "Fe"],
        fwhm_nm=0.1,
        window_n_fwhm=1.5,
        blend_fraction=0.15,
        min_lines_per_element=3,
    )
    # Unprotected w would be 0.075; floor lifts it to 0.5.
    assert weights[id(obs[0])] == pytest.approx(0.5, rel=1e-9)


def test_self_neighbor_not_counted_as_contaminant():
    """A DB line coincident with the candidate (same element, ~same wl) is the
    candidate itself and must not down-weight it."""
    obs = [_obs("Fe", 500.0), _obs("Fe", 510.0), _obs("Fe", 520.0), _obs("Fe", 530.0)]
    db = FakeDB([_tr("Fe", 500.0, a_ki=1e8)])  # exact self -> skipped
    weights = compute_db_isolation_weights(obs, db, fwhm_nm=0.1, window_n_fwhm=1.5)
    assert all(weights[id(o)] == 1.0 for o in obs)


def test_neighbor_below_blend_fraction_keeps_weight_one():
    """A faint neighbor (ratio <= blend_fraction) does not trigger the gate."""
    obs = [_obs("Fe", 500.0), _obs("Fe", 510.0), _obs("Fe", 520.0), _obs("Fe", 530.0)]
    db = FakeDB([_tr("Fe", 500.06, a_ki=1e7)])  # 0.1x candidate -> ratio 0.1 <= 0.15
    weights = compute_db_isolation_weights(
        obs, db, fwhm_nm=0.1, window_n_fwhm=1.5, blend_fraction=0.15
    )
    assert all(weights[id(o)] == 1.0 for o in obs)


def test_neighbor_outside_window_ignored():
    """A strong neighbor outside +/- window_n_fwhm*fwhm is not a contaminant."""
    obs = [_obs("Fe", 500.0), _obs("Fe", 510.0), _obs("Fe", 520.0), _obs("Fe", 530.0)]
    db = FakeDB([_tr("Fe", 500.5, a_ki=1e10)])  # 0.5 nm away, window is +/-0.15 nm
    weights = compute_db_isolation_weights(
        obs, db, fwhm_nm=0.1, window_n_fwhm=1.5, blend_fraction=0.15
    )
    assert all(weights[id(o)] == 1.0 for o in obs)


def test_disabled_and_degenerate_fast_exits():
    obs = [_obs("Fe", 500.0)]
    db = FakeDB([_tr("Fe", 500.06, a_ki=2e8)])
    # fwhm <= 0 disables the gate
    assert compute_db_isolation_weights(obs, db, fwhm_nm=0.0) == {id(obs[0]): 1.0}
    # None db -> all 1.0
    assert compute_db_isolation_weights(obs, None, fwhm_nm=0.1) == {id(obs[0]): 1.0}
    # empty observations -> empty mapping
    assert compute_db_isolation_weights([], db, fwhm_nm=0.1) == {}


def test_abundance_scaling_demotes_trace_contaminant():
    """A trace-abundance contaminant cannot blend a major candidate when a
    composition prior is supplied (abundance-scaled emissivity)."""
    obs = [_obs("Fe", 500.0), _obs("Fe", 510.0), _obs("Fe", 520.0), _obs("Fe", 530.0)]
    db = FakeDB([_tr("Cu", 500.06, a_ki=2e8)])  # 2x per-atom, but trace abundance
    # Without a prior, the trace Cu neighbor blends (ratio 2.0 -> down-weight).
    w_noprior = compute_db_isolation_weights(
        obs, db, elements=["Fe", "Cu"], fwhm_nm=0.1, window_n_fwhm=1.5
    )
    assert w_noprior[id(obs[0])] < 1.0
    # With a strong-Fe / trace-Cu prior, the abundance ratio kills the blend.
    w_prior = compute_db_isolation_weights(
        obs,
        db,
        elements=["Fe", "Cu"],
        fwhm_nm=0.1,
        window_n_fwhm=1.5,
        concentrations={"Fe": 0.99, "Cu": 0.01},
    )
    # contaminant scaled by 0.01, candidate by 0.99 -> ratio ~ 2*0.01/0.99 << 0.15
    assert w_prior[id(obs[0])] == 1.0


# --- solver wiring: IterativeCFLIBSSolver._apply_db_isolation_gate ---------------


def _make_solver(gate: bool, db):
    from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver

    return IterativeCFLIBSSolver(db, db_isolation_gate=gate)


def test_gate_off_is_strict_object_passthrough():
    """gate OFF: the same mapping object is returned (byte-identical default)."""
    db = FakeDB([_tr("Fe", 500.06, a_ki=2e8)])
    solver = _make_solver(gate=False, db=db)
    obs_by_element = {"Fe": [_obs("Fe", 500.0), _obs("Fe", 510.0)]}
    out = solver._apply_db_isolation_gate(obs_by_element, T_K=11000.0, n_e=1e17, concentrations={})
    assert out is obs_by_element  # identity: no copy, no mutation


def test_gate_on_inflates_sigma_of_blended_line_only():
    """gate ON: a blended line's intensity_uncertainty is inflated by 1/sqrt(w);
    isolated lines pass through unchanged (same object)."""
    db = FakeDB([_tr("Fe", 500.06, a_ki=2e8)])
    solver = _make_solver(gate=True, db=db)
    blended = _obs("Fe", 500.0)
    iso = [_obs("Fe", 510.0), _obs("Fe", 520.0), _obs("Fe", 530.0)]
    obs_by_element = {"Fe": [blended, *iso]}

    out = solver._apply_db_isolation_gate(
        obs_by_element, T_K=11000.0, n_e=1e17, concentrations={}
    )

    assert out is not obs_by_element  # a new mapping when active
    gated = out["Fe"]
    # The blended line is a new object with inflated sigma (w = 0.075).
    new_blended = gated[0]
    assert new_blended is not blended
    expected_sigma = 10.0 / math.sqrt(0.075)
    assert new_blended.intensity_uncertainty == pytest.approx(expected_sigma, rel=1e-9)
    # Intensity / atomic data untouched -> only the fit weight changes.
    assert new_blended.intensity == blended.intensity
    assert new_blended.A_ki == blended.A_ki
    # Isolated lines are passed through as the SAME objects.
    for original, returned in zip(iso, gated[1:]):
        assert returned is original

    # The inflation reduces the fit weight (1/sigma_y^2) by exactly w.
    w_fit_ratio = (blended.y_uncertainty / new_blended.y_uncertainty) ** 2
    assert w_fit_ratio == pytest.approx(0.075, rel=1e-9)


def test_gate_on_enabled_flag_forces_host_path():
    """Enabling the gate must force the Python loop (host-only); the solver
    exposes the flag the dispatcher reads."""
    db = FakeDB([])
    solver = _make_solver(gate=True, db=db)
    assert solver.db_isolation_gate is True
    # default construction keeps it off
    off = _make_solver(gate=False, db=db)
    assert off.db_isolation_gate is False


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
