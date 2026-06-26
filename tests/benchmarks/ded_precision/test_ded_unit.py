"""DB-free unit tests for the DED precision benchmark scaffold (DED-PLAN step 3).

Covers noise reproducibility/shape, plasma-jitter sampling, composition-series
integrity, and number<->mass round-tripping. No atomic DB -> safe for CI.
"""

import numpy as np
import pytest

from tests.benchmarks.ded_precision.alloy_definitions import (
    ALLOY_COMPOSITIONS,
    COMPOSITION_SERIES,
    elements_of,
    make_series,
)
from tests.benchmarks.ded_precision.noise_model import DEDNoiseParams, apply_ded_noise

pytestmark = pytest.mark.unit


def _clean():
    x = np.zeros(400)
    for c in (80, 200, 330):
        x += np.exp(-0.5 * ((np.arange(400) - c) / 3.0) ** 2)
    return x


def test_noise_reproducible_same_seed():
    p = DEDNoiseParams()
    a = apply_ded_noise(_clean(), p, np.random.default_rng(7))
    b = apply_ded_noise(_clean(), p, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_noise_differs_different_seed():
    p = DEDNoiseParams()
    a = apply_ded_noise(_clean(), p, np.random.default_rng(1))
    b = apply_ded_noise(_clean(), p, np.random.default_rng(2))
    assert not np.array_equal(a, b)


def test_noise_shape_and_nonneg():
    x = _clean()
    y = apply_ded_noise(x, DEDNoiseParams(), np.random.default_rng(0))
    assert y.shape == x.shape
    assert np.all(y >= 0.0)


def test_zero_noise_params_recovers_clean_plus_no_baseline():
    x = _clean()
    p = DEDNoiseParams(
        sigma_shot=0.0,
        peak_photons=1e18,
        readout_frac=0.0,
        baseline_b0_frac=0.0,
        baseline_b1_frac=0.0,
    )
    y = apply_ded_noise(x, p, np.random.default_rng(0))
    assert np.allclose(y, x, rtol=1e-3, atol=1e-6)


def test_plasma_jitter_sampling_reproducible_and_bounded():
    p = DEDNoiseParams()
    t1, ne1 = p.sample_plasma_jitter(11000.0, 1e17, np.random.default_rng(3))
    t2, ne2 = p.sample_plasma_jitter(11000.0, 1e17, np.random.default_rng(3))
    assert (t1, ne1) == (t2, ne2)
    assert 9000 < t1 < 13000
    assert 1e16 < ne1 < 1e18


def test_peak_snr_matches_photon_budget():
    assert DEDNoiseParams(peak_photons=1e4).peak_snr == pytest.approx(100.0)


def test_nominal_compositions_sum_to_100():
    for alloy, comp in ALLOY_COMPOSITIONS.items():
        assert sum(comp.values()) == pytest.approx(100.0), alloy


def test_composition_series_sums_to_100_and_varies_target():
    for alloy, axes in COMPOSITION_SERIES.items():
        for element, series in axes.items():
            vals = [c[element] for c in series]
            assert vals == sorted(vals) and len(set(vals)) == len(vals)
            for comp in series:
                assert set(comp) == set(elements_of(alloy))
                assert sum(comp.values()) == pytest.approx(100.0)


def test_make_series_preserves_other_element_ratios():
    base = {"Ti": 90.0, "Al": 6.0, "V": 4.0}
    series = make_series(base, "Al", np.array([4.0, 8.0]))
    for comp in series:
        # Ti:V ratio is preserved (both scaled by the same factor)
        assert comp["Ti"] / comp["V"] == pytest.approx(base["Ti"] / base["V"])


def test_number_to_mass_roundtrip():
    from cflibs.atomic.masses import STANDARD_ATOMIC_MASSES as M
    from cflibs.inversion.pipeline import _number_to_mass_fractions

    mass = {e: w / 100.0 for e, w in ALLOY_COMPOSITIONS["Ti-6Al-4V"].items()}
    number = {e: (mass[e] / M[e]) for e in mass}
    s = sum(number.values())
    number = {e: v / s for e, v in number.items()}
    back = _number_to_mass_fractions(number)
    for e in mass:
        assert back[e] == pytest.approx(mass[e], abs=1e-9)
