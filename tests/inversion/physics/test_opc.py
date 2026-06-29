"""Unit tests for the robust, conditioning-gated OPC core (known-matrix mode).

These tests are fully self-contained (synthetic composition-recovery closures, no
atomic DB) and assert the four load-bearing properties of the promoted lever-L3b
algorithm:

1. the per-element geometric mean with the clamp-saturated / degenerate filter;
2. the in-sample-only conditioning gate (converged / not-degenerate / RMSEP /
   non-matrix-fraction);
3. the structural-honesty property — :func:`apply_opc` takes only observations +
   a calibration and never reads a recovered composition; and
4. a tiny synthetic calibrate -> apply round-trip.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from cflibs.inversion.physics.opc import (
    F_MAX,
    F_MIN,
    OPCCalibration,
    Standard,
    StandardRecovery,
    _geomean_F,
    apply_opc,
    assess_conditioning,
    calibrate_opc,
    choose_optimal_temperature,
)

# ---------------------------------------------------------------------------
# Synthetic standard builders (no DB)
# ---------------------------------------------------------------------------


def _biased_standard(
    name: str,
    certified: dict[str, float],
    bias: dict[str, float],
    *,
    converged: bool = True,
    degenerate: bool = False,
    t_optimum: float | None = None,
):
    """Standard whose uncorrected recovery applies a per-element multiplicative ``bias``.

    If ``t_optimum`` is given, an extra T-dependent skew is added to ``Cr`` that
    vanishes at ``t_optimum`` (so :func:`choose_optimal_temperature` is exercised).
    """

    def recover(T_K: float) -> StandardRecovery:
        raw = {e: certified.get(e, 0.0) * bias.get(e, 1.0) for e in certified}
        if t_optimum is not None and "Cr" in raw:
            raw["Cr"] += 40.0 * abs(T_K - t_optimum) / 10000.0
        return StandardRecovery(composition=raw, converged=converged, degenerate=degenerate)

    return Standard(name=name, certified=certified, recover=recover)


# ---------------------------------------------------------------------------
# 1. geometric mean + clamp / degenerate filter
# ---------------------------------------------------------------------------


def test_geomean_basic():
    # geomean(2, 8) = sqrt(16) = 4 (multiplicative average, not arithmetic 5).
    out = _geomean_F([{"Fe": 2.0}, {"Fe": 8.0}])
    assert out["Fe"] == pytest.approx(4.0, rel=1e-9)


def test_geomean_filters_clamp_saturated_values():
    # A clamp-saturated factor (degenerate standard) must be dropped before the mean.
    out = _geomean_F([{"Mo": F_MAX}, {"Mo": 2.0}, {"Mo": 3.0}])
    assert out["Mo"] == pytest.approx(np.sqrt(6.0), rel=1e-9)

    # F_MIN saturation is filtered the same way.
    out_min = _geomean_F([{"Mo": F_MIN}, {"Mo": 4.0}])
    assert out_min["Mo"] == pytest.approx(4.0, rel=1e-9)


def test_geomean_all_degenerate_yields_unity():
    # Every standard saturated for this element -> no information -> F = 1 (no correction).
    out = _geomean_F([{"X": F_MAX}, {"X": F_MIN}])
    assert out["X"] == 1.0


def test_geomean_drops_nonfinite_and_nonpositive():
    out = _geomean_F([{"X": -1.0}, {"X": float("nan")}, {"X": 9.0}])
    assert out["X"] == pytest.approx(9.0, rel=1e-9)


def test_geomean_result_within_band():
    # A huge unsaturated value gets clamped into the band after averaging.
    out = _geomean_F([{"Y": 50.0}, {"Y": 80.0}])
    assert F_MIN <= out["Y"] <= F_MAX


# ---------------------------------------------------------------------------
# 2. conditioning gate (in-sample only)
# ---------------------------------------------------------------------------


def test_conditioning_passes_well_conditioned_standard():
    certified = {"Fe": 90.0, "Cr": 7.0, "Ni": 3.0}
    std = _biased_standard("good", certified, bias={"Fe": 1.0, "Cr": 1.0, "Ni": 1.0})
    a = assess_conditioning(std)
    assert a.passed
    assert a.matrix_element == "Fe"
    assert a.in_sample_rmsep == pytest.approx(0.0, abs=1e-9)


def test_conditioning_drops_keystone_collapsed_standard():
    # Cr (a trace minor) recovered as the dominant fraction -> nonmatrix gate fails.
    certified = {"Fe": 90.0, "Cr": 7.0, "Ni": 3.0}
    std = _biased_standard("collapsed", certified, bias={"Fe": 0.02, "Cr": 30.0})
    a = assess_conditioning(std)
    assert not a.passed
    assert a.nonmatrix_max_pct > 60.0


def test_conditioning_drops_unconverged_and_degenerate():
    certified = {"Fe": 90.0, "Cr": 10.0}
    not_conv = _biased_standard("nc", certified, bias={}, converged=False)
    degen = _biased_standard("dg", certified, bias={}, degenerate=True)
    assert not assess_conditioning(not_conv).passed
    assert not assess_conditioning(degen).passed


# ---------------------------------------------------------------------------
# choose_optimal_temperature
# ---------------------------------------------------------------------------


def test_choose_optimal_temperature_finds_minimum():
    certified = {"Fe": 80.0, "Cr": 13.0, "Ni": 7.0}
    std = _biased_standard(
        "topt", certified, bias={"Fe": 1.0, "Cr": 1.0, "Ni": 1.0}, t_optimum=10000.0
    )
    grid = (7000.0, 8000.0, 9000.0, 10000.0, 11000.0, 12000.0)
    assert choose_optimal_temperature(std, grid) == 10000.0


# ---------------------------------------------------------------------------
# 3. structural honesty
# ---------------------------------------------------------------------------


class _Obs:
    """Minimal mutable line-observation stand-in (element + intensity + uncertainty)."""

    def __init__(self, element: str, intensity: float, unc: float = 1.0):
        self.element = element
        self.intensity = intensity
        self.intensity_uncertainty = unc


def test_apply_opc_signature_has_no_composition():
    params = list(inspect.signature(apply_opc).parameters)
    assert params == ["observations", "calibration"]
    banned = ("composition", "comp", "truth", "recovered", "concentration")
    assert not any(any(b in p.lower() for b in banned) for p in params)


def test_apply_opc_rescales_in_place_and_only_reads_F():
    cal = OPCCalibration(
        robust_T_K=9000.0,
        F={"Fe": 2.0, "Cr": 0.5},
        selected_standards=["s"],
        conditioning_rule="test",
    )
    obs = [_Obs("Fe", 10.0, 1.0), _Obs("Cr", 4.0, 2.0), _Obs("Ni", 5.0, 1.0)]
    assert apply_opc(obs, cal) is None  # in place, returns None
    assert obs[0].intensity == pytest.approx(20.0)  # Fe * 2.0
    assert obs[0].intensity_uncertainty == pytest.approx(2.0)
    assert obs[1].intensity == pytest.approx(2.0)  # Cr * 0.5
    assert obs[2].intensity == pytest.approx(5.0)  # Ni absent from F -> unchanged


# ---------------------------------------------------------------------------
# 4. tiny synthetic calibrate -> apply round-trip
# ---------------------------------------------------------------------------


def test_calibrate_round_trip_recovers_inverse_bias_ordering():
    certified = {"Fe": 80.0, "Cr": 13.0, "Ni": 7.0}
    # Cr under-recovered (bias < 1) -> F_Cr should be large; Ni over-recovered
    # (bias > 1) -> F_Ni should be small; Fe ~ unbiased anchor in between.
    bias = {"Fe": 1.0, "Cr": 0.5, "Ni": 2.0}
    stds = [_biased_standard(f"std{i}", certified, bias) for i in range(3)]
    cal = calibrate_opc(stds)

    assert isinstance(cal, OPCCalibration)
    assert len(cal.selected_standards) == 3  # all well-conditioned -> all selected
    assert min(DEFAULT_GRID := (7000.0, 12000.0)) <= cal.robust_T_K <= max(DEFAULT_GRID)
    # OPC factor recovers the inverse-bias ordering.
    assert cal.F["Cr"] > cal.F["Fe"] > cal.F["Ni"]

    # Apply to a fresh unknown carrying the same matrix bias: Cr boosted, Ni damped.
    obs = [_Obs("Fe", 10.0), _Obs("Cr", 10.0), _Obs("Ni", 10.0)]
    apply_opc(obs, cal)
    assert obs[1].intensity > obs[0].intensity > obs[2].intensity


def test_calibrate_drops_degenerate_standard_from_selection():
    certified = {"Fe": 90.0, "Cr": 7.0, "Ni": 3.0}
    good = [
        _biased_standard(f"good{i}", certified, {"Fe": 1.0, "Cr": 1.0, "Ni": 1.0}) for i in range(2)
    ]
    bad = _biased_standard("bad", certified, {"Fe": 0.02, "Cr": 30.0})
    cal = calibrate_opc(good + [bad])
    assert "bad" not in cal.selected_standards
    assert set(cal.selected_standards) == {"good0", "good1"}


def test_calibrate_requires_at_least_one_standard():
    with pytest.raises(ValueError):
        calibrate_opc([])
