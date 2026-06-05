"""Per-element Boltzmann-weight dynamic-range cap (rank-1 presence fix).

The pooled common-slope fit weights lines by inverse variance w = 1/sigma_y^2.
Under the Poisson intensity model w ~ I, so an element's fitted intercept q_s is
dominated by its single brightest line. The cap clips each element's weights to
``K x median(valid weights)`` so a single artificially-bright line can no longer
hijack the intercept.

These tests are backend-agnostic by construction:

* The CPU host path builds weights+intercepts in ``_fit_common_boltzmann_plane``.
* The JAX ``lax`` path builds weights in ``_build_padded_arrays_from_obs`` and
  feeds the same WLS kernel.

Both consume the *same* helper ``_cap_boltzmann_weights`` on the same
post-valid-mask weights, so capped weights are bit-for-bit identical across
backends — we assert that directly rather than depending on a JAX install.
"""

from __future__ import annotations

import numpy as np
import pytest

from cflibs.inversion.common import LineObservation
from cflibs.inversion.solve.iterative import (
    _build_padded_arrays_from_obs,
    _cap_boltzmann_weights,
)


def _line(
    wl: float,
    E_k: float,
    intensity: float,
    rel_unc: float,
    element: str,
    stage: int = 1,
    g_k: float = 5.0,
    A_ki: float = 1.0e8,
) -> LineObservation:
    """A LineObservation whose fit-space weight is controlled via rel_unc.

    ``y_uncertainty == intensity_uncertainty / intensity == rel_unc`` so the
    inverse-variance weight is ``1 / rel_unc**2``; a smaller ``rel_unc`` => a
    larger weight (mimics a bright, low-noise line).
    """
    return LineObservation(
        wavelength_nm=wl,
        intensity=intensity,
        intensity_uncertainty=rel_unc * intensity,
        element=element,
        ionization_stage=stage,
        E_k_ev=E_k,
        g_k=g_k,
        A_ki=A_ki,
    )


def test_cap_helper_clips_to_k_times_median_and_preserves_low_weights():
    w = np.array([1.0, 1.0, 1.0, 1.0, 1000.0])  # median == 1.0
    capped = _cap_boltzmann_weights(w, cap=5.0)
    # The 1000x outlier is clipped to 5*median; the in-regime weights untouched.
    assert capped[-1] == pytest.approx(5.0)
    assert np.allclose(capped[:4], 1.0)


def test_cap_helper_disabled_is_identity():
    w = np.array([1.0, 1000.0])
    assert np.array_equal(_cap_boltzmann_weights(w, cap=0.0), w)
    assert np.array_equal(_cap_boltzmann_weights(w, cap=-1.0), w)


def test_cap_helper_noop_when_already_in_range():
    w = np.array([1.0, 2.0, 3.0, 4.0])  # max 4 < 5*median(2.5)=12.5
    assert np.array_equal(_cap_boltzmann_weights(w, cap=5.0), w)


def _two_element_obs() -> dict[str, list[LineObservation]]:
    """Two elements with identical TRUE Boltzmann lines (same x, same honest y).

    Element A additionally gets ONE artificially bright, low-noise line that
    sits ABOVE its honest Boltzmann line (a self-absorption / saturation-style
    anomaly) and carries ~1e4x the weight of the honest lines — the
    single-bright-line pathology the cap is designed to neutralize. The line's
    huge inverse-variance weight makes A's *weighted* intercept track that one
    point, inflating q_A above the (physically equal) q_B.
    """
    # Shared honest lines: a clean negative-slope Boltzmann set, moderate weight
    # (rel_unc 0.1 => w = 100).
    base = [(2.0, 1.0e3), (3.0, 5.0e2), (4.0, 2.5e2), (5.0, 1.0e2)]
    a = [_line(400.0 + i, E, inten, rel_unc=0.1, element="A") for i, (E, inten) in enumerate(base)]
    b = [_line(500.0 + i, E, inten, rel_unc=0.1, element="B") for i, (E, inten) in enumerate(base)]
    # A's hijack line: at the low-E_k (high-intercept) end, ~30x brighter than the
    # honest line there and with tiny rel_unc => weight ~ (1/1e-3)^2 = 1e6, i.e.
    # ~1e4x the honest weights. It lies ABOVE A's true line, lifting q_A.
    a.append(_line(409.0, E_k=2.0, intensity=3.0e4, rel_unc=1.0e-3, element="A"))
    return {"A": a, "B": b}


def _row_weights(obs_map, weight_cap):
    elements, _x, _y, w, _stage, mask = _build_padded_arrays_from_obs(
        obs_map, weight_cap=weight_cap
    )
    out = {}
    for i, el in enumerate(elements):
        out[el] = np.asarray(w[i][mask[i]], dtype=float)
    return out


def test_padded_builder_uncapped_has_huge_dynamic_range_capped_bounds_it():
    obs_map = _two_element_obs()

    uncapped = _row_weights(obs_map, weight_cap=0.0)
    capped = _row_weights(obs_map, weight_cap=5.0)

    # Uncapped: A's hijack line dominates — max/median weight ratio is enormous.
    a_unc = uncapped["A"]
    assert a_unc.max() / np.median(a_unc) > 100.0

    # Capped: A's max weight is exactly 5x its own median; B (no outlier)
    # is unchanged.
    a_cap = capped["A"]
    assert a_cap.max() == pytest.approx(5.0 * np.median(a_cap))
    assert np.allclose(capped["B"], uncapped["B"])


def _fit_gap(cap: float) -> float:
    """A-minus-B intercept gap from the production numpy host fit at a given cap."""
    from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver

    obs_map = _two_element_obs()
    # No DB needed: _fit_common_boltzmann_plane only touches obs y/x/weights.
    solver = IterativeCFLIBSSolver.__new__(IterativeCFLIBSSolver)
    solver.aki_uncertainty_weighting = False
    solver.boltzmann_weight_cap = cap
    fit = solver._fit_common_boltzmann_plane(obs_map)
    assert fit is not None
    return fit.intercepts["A"] - fit.intercepts["B"]


def test_cap_corrects_intercept_misranking_cpu_host_path():
    """Uncapped fit mis-ranks A's intercept above B; the cap shrinks the gap.

    Exercises the production numpy host fit (``_fit_common_boltzmann_plane``),
    the same code the analyze path runs. A and B have physically identical
    honest lines, so a faithful fit must give equal intercepts; A's single
    artificially-bright high-weight line inflates q_A in the uncapped WLS.
    """
    gap_unc = _fit_gap(0.0)
    gap_cap = _fit_gap(5.0)

    # Uncapped: A's hijack line inflates its intercept materially above B's.
    assert gap_unc > 0.5, f"expected uncapped A>B intercept inflation, got {gap_unc:.3f}"
    # The cap reduces the single-bright-line dominance, shrinking the gap toward
    # the physical zero. (It bounds, not deletes, the line, so the gap need not
    # vanish for a 5-point element.)
    assert gap_cap < gap_unc, (
        f"cap should shrink A-B intercept gap: uncapped={gap_unc:.3f} " f"capped={gap_cap:.3f}"
    )


def test_cap_effect_is_monotone_in_k():
    """Tighter caps (smaller K) shrink the spurious intercept gap more.

    Direction-safety check: the correction is monotone in the cap, so K is a
    well-behaved single knob and there is no pathological non-monotonicity.
    """
    gaps = {k: _fit_gap(k) for k in (0.0, 8.0, 5.0, 3.0)}
    # Disabled (0.0) is the largest gap; tightening monotonically reduces it.
    assert gaps[0.0] >= gaps[8.0] >= gaps[5.0] >= gaps[3.0], gaps
