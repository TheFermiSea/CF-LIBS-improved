"""Validate the C-sigma composition solver's partition-function correction.

The bare ``fit_csigma`` omits the species partition function from the cross-section, so its
``relative_concentrations`` are ``C_s / U_s(T)`` — NOT a composition (wrong by up to the
inter-element U ratio, ~2-50x). This is exactly the per-element-``U_s`` corruption the
cflibs-formal oracle (``Closure.composition_sum_one`` / the README's per-element-U_s negative
test) is built to catch. ``solve_csigma_composition`` restores ``U_s(T)`` from the DB.

These tests forward-generate PHYSICALLY-correct C-sigma intensities (the dimensionless
kB=T=1 oracle fixture cannot drive C-sigma's physical Planck radiance), with a KNOWN
composition and KNOWN partition functions, then verify:
  (1) the RAW fit is wrong by exactly the U ratio (so we never mistake it for a composition),
  (2) applying U_s recovers the true composition.
"""

from __future__ import annotations

import math

import pytest

from cflibs.inversion.common.data_structures import LineObservation
from cflibs.inversion.physics.csigma import (
    _log_sigma_rel,
    _planck_radiance,
    cog_function,
    fit_csigma,
)

T_K = 10000.0
SCALE = 1e-3  # thin lines -> clean recovery


def _forward_lines(
    true_conc: dict[str, float], partition: dict[str, float]
) -> list[LineObservation]:
    """Build C-sigma lines whose optical depth carries the physical C_s/U_s the bare fit omits."""
    lines: list[LineObservation] = []
    for el in true_conc:
        for e_k, lam in [(3.5, 320.0), (4.2, 400.0), (5.0, 480.0), (5.8, 560.0)]:
            probe = LineObservation(
                wavelength_nm=lam,
                intensity=1.0,
                intensity_uncertainty=1e-6,
                element=el,
                ionization_stage=1,
                E_k_ev=e_k,
                g_k=4,
                A_ki=5e7,
            )
            sigma_rel = 10.0 ** _log_sigma_rel(probe, T_K)
            tau0 = SCALE * (true_conc[el] / partition[el]) * sigma_rel  # physical /U_s
            inten = _planck_radiance(lam, T_K) * cog_function(tau0)
            lines.append(
                LineObservation(
                    wavelength_nm=lam,
                    intensity=inten,
                    intensity_uncertainty=1e-9,
                    element=el,
                    ionization_stage=1,
                    E_k_ev=e_k,
                    g_k=4,
                    A_ki=5e7,
                )
            )
    return lines


def _norm(d: dict[str, float]) -> dict[str, float]:
    tot = sum(d.values())
    return {k: v / tot for k, v in d.items()}


def test_raw_csigma_is_wrong_by_the_partition_ratio():
    """RAW fit_csigma.relative_concentrations = C_s/U_s, NOT the composition: equal true
    concentrations with U=2 vs U=20 come out ~10:1. Asserts the bug is real (so the corrected
    solver is necessary, not optional)."""
    true_conc = {"A": 0.5, "B": 0.5}
    partition = {"A": 2.0, "B": 20.0}
    raw = fit_csigma(_forward_lines(true_conc, partition)).relative_concentrations
    ratio = raw["A"] / raw["B"]
    assert math.isclose(ratio, partition["B"] / partition["A"], rel_tol=0.05), (
        f"raw C-sigma ratio {ratio:.2f} should be U_B/U_A={partition['B'] / partition['A']:.1f} "
        "(the omitted partition function)"
    )
    # and it must be FAR from the true (equal) composition
    assert abs(raw["A"] - 0.5) > 0.3, "raw C-sigma must NOT already equal the true composition"


def test_partition_correction_recovers_composition():
    """Applying U_s(T) to the raw fit recovers the true composition (the solve_csigma_composition
    correction), for several known compositions/partition ratios."""
    partition = {"A": 2.0, "B": 20.0, "C": 8.0}
    for true_conc in (
        {"A": 0.5, "B": 0.5},
        {"A": 0.7, "B": 0.3},
        {"A": 0.2, "B": 0.3, "C": 0.5},
    ):
        sub_part = {k: partition[k] for k in true_conc}
        raw = fit_csigma(_forward_lines(true_conc, sub_part)).relative_concentrations
        corrected = _norm({el: raw[el] * sub_part[el] for el in raw})
        for el in true_conc:
            assert math.isclose(
                corrected[el], true_conc[el], abs_tol=0.02
            ), f"{true_conc}: corrected[{el}]={corrected[el]:.3f} != true {true_conc[el]}"


@pytest.mark.requires_db
def test_solve_csigma_composition_runs_on_real_db():
    """The DB-backed solver returns a valid number-fraction composition on a real spectrum."""
    pytest.importorskip("numpy")
    from cflibs.atomic.database import AtomicDatabase
    from cflibs.benchmark.scoreboard import (
        _sample_indices,
        ensure_default_datasets,
        iter_datasets,
    )
    from cflibs.inversion.physics.csigma import solve_csigma_composition
    from cflibs.inversion.pipeline import build_pipeline_config, run_pipeline
    from cflibs.inversion.physics.line_selection import LineSelector

    import os

    if not os.path.exists("ASD_da/libs_production.db"):
        pytest.skip("NIST DB not present")
    ensure_default_datasets()
    db = AtomicDatabase("ASD_da/libs_production.db")
    entry = next(iter_datasets(names=["supercam_labcal"]))
    items = list(entry.adapter_factory())
    sid, wl, inten, truth = items[_sample_indices(len(items), 1, 7)[0]]

    captured: dict = {}
    orig = LineSelector.select

    def spy(self, observations, **kw):
        res = orig(self, observations, **kw)
        captured["obs"] = res.selected_lines
        return res

    LineSelector.select = spy
    try:
        run_pipeline(wl, inten, db, build_pipeline_config(list(truth.composition_wt.keys())))
    finally:
        LineSelector.select = orig

    out = solve_csigma_composition(captured.get("obs", []), db)
    if out is None:
        pytest.skip("C-sigma found no stage with >=3 lines for this spectrum")
    composition, t_k = out
    assert composition and math.isclose(sum(composition.values()), 1.0, abs_tol=1e-6)
    assert 3000.0 <= t_k <= 30000.0
