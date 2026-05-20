"""Tighter-tolerance reconstruction tests for PCA pipeline.

Followup to test-coverage audit (2026-05-20): the round-trip
reconstruction test in tests/test_pca.py used rtol=0.15 which is too
loose to catch real regressions in the inverse-transform path.
"""

import numpy as np
import pytest

from cflibs.inversion.pca import PCAPipeline


@pytest.fixture
def low_rank_spectra():
    rng = np.random.default_rng(2026)
    n_samples = 40
    n_wavelengths = 150
    wl = np.linspace(400.0, 700.0, n_wavelengths)

    pc1 = np.exp(-((wl - 500) ** 2) / (2 * 40**2))
    pc2 = np.exp(-((wl - 600) ** 2) / (2 * 15**2))
    pc3 = (wl - 400) / 300

    s1 = rng.normal(10, 3, n_samples)
    s2 = rng.normal(5, 1, n_samples)
    s3 = rng.normal(2, 0.5, n_samples)

    return np.outer(s1, pc1) + np.outer(s2, pc2) + np.outer(s3, pc3)


def test_reconstruction_tight_when_components_match_rank(low_rank_spectra):
    pca = PCAPipeline(n_components=5, use_jax=False)
    result = pca.fit(low_rank_spectra)
    scores = result.transform(low_rank_spectra)
    reconstructed = result.inverse_transform(scores)
    np.testing.assert_allclose(
        reconstructed, low_rank_spectra, rtol=1e-8, atol=1e-8
    )


def test_full_rank_reconstruction_machine_precision(low_rank_spectra):
    n_samples, n_features = low_rank_spectra.shape
    max_components = min(n_samples, n_features)
    pca = PCAPipeline(n_components=max_components, use_jax=False)
    result = pca.fit(low_rank_spectra)
    scores = result.transform(low_rank_spectra)
    reconstructed = result.inverse_transform(scores)
    np.testing.assert_allclose(
        reconstructed, low_rank_spectra, rtol=1e-10, atol=1e-10
    )


def test_rmse_below_signal_scale_for_high_component_count():
    rng = np.random.default_rng(42)
    n_samples, n_features = 60, 200
    wl = np.linspace(400, 800, n_features)
    base = np.exp(-((wl - 550) ** 2) / (2 * 50**2))
    scores = rng.normal(10, 3, n_samples)
    data = np.outer(scores, base) + rng.normal(0, 0.05, (n_samples, n_features))
    signal_scale = float(np.ptp(data))

    pca = PCAPipeline(n_components=30, use_jax=False)
    result = pca.fit(data)
    _, rmse = result.reconstruction_error(data)
    assert rmse < 0.01 * signal_scale
