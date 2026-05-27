"""Coverage backfill for cflibs.inversion.solve.spectral_refiner.

Note: tests/test_spectral_refiner.py already exists with its own class
set; this module is named differently so it does not clobber those
symbols. Followup to test-coverage audit (2026-05-20).
"""

from dataclasses import dataclass

import numpy as np

from cflibs.inversion.solve.spectral_refiner import (
    RefinementResult,
    SpectralRefiner,
)


@dataclass
class _FakeBasisLibrary:
    wavelength: np.ndarray
    elements: list

    def __init__(self, n_pixels=64, elements=None):
        self.wavelength = np.linspace(400.0, 500.0, n_pixels)
        self.elements = elements or ["Fe", "Ca", "Mg"]
        self._n_pixels = n_pixels

    def _gaussian(self, center, fwhm=2.0, amp=1.0):
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        return amp * np.exp(-((self.wavelength - center) ** 2) / (2 * sigma**2))

    def get_basis_matrix_interp(self, T_K, ne_cm3):
        T_factor = T_K / 10000.0
        centers = [420.0, 440.0, 460.0]
        basis = np.zeros((len(self.elements), self._n_pixels))
        for i, c in enumerate(centers[: len(self.elements)]):
            basis[i] = self._gaussian(c, fwhm=2.0, amp=T_factor)
        return basis


class TestRefinementResultDataclassCoverage:
    def test_construct(self):
        r = RefinementResult(
            T_K=8000.0,
            ne_cm3=1e17,
            concentrations={"Fe": 0.5, "Ca": 0.5},
            residual_norm=0.1,
            n_iterations=5,
            converged=True,
            chi_squared=10.0,
            chi_squared_reduced=0.5,
        )
        # NOSONAR — these are setter/getter round-trip checks on values literally
        # passed to RefinementResult() above; exact equality is the correct test.
        assert r.T_K == 8000.0  # noqa: PLR2004
        assert r.concentrations["Fe"] == 0.5  # noqa: PLR2004
        assert r.converged is True


class TestSpectralRefinerEmptyPathsCoverage:
    def test_no_detected_elements(self):
        lib = _FakeBasisLibrary()
        refiner = SpectralRefiner(lib)
        observed = np.ones(64)
        result = refiner.refine(
            wavelength=lib.wavelength,
            observed=observed,
            detected_elements=[],
            T_init_K=9000.0,
            ne_init_cm3=1e17,
            concentrations_init={},
        )
        assert result.concentrations == {}
        assert result.n_iterations == 0
        assert result.converged is True
        assert (
            result.T_K == 9000.0
        )  # NOSONAR — T_init_K was set literally above; exact eq is correct

    def test_detected_elements_not_in_library(self):
        lib = _FakeBasisLibrary(elements=["Fe", "Ca"])
        refiner = SpectralRefiner(lib)
        observed = np.ones(64)
        result = refiner.refine(
            wavelength=lib.wavelength,
            observed=observed,
            detected_elements=["Zr", "Hf"],
            T_init_K=9000.0,
            ne_init_cm3=1e17,
            concentrations_init={"Zr": 0.5, "Hf": 0.5},
        )
        assert result.n_iterations == 0
        assert "Zr" in result.concentrations
        assert "Hf" in result.concentrations


class TestSpectralRefinerCoreCoverage:
    def test_refine_runs_and_returns_result(self):
        lib = _FakeBasisLibrary()
        refiner = SpectralRefiner(lib, max_iterations=5)
        basis = lib.get_basis_matrix_interp(8000.0, 1e17)
        observed = 0.6 * basis[0] + 0.4 * basis[1]

        result = refiner.refine(
            wavelength=lib.wavelength,
            observed=observed,
            detected_elements=["Fe", "Ca"],
            T_init_K=8000.0,
            ne_init_cm3=1e17,
            concentrations_init={"Fe": 0.5, "Ca": 0.5},
        )
        assert isinstance(result, RefinementResult)
        assert set(result.concentrations.keys()) == {"Fe", "Ca"}
        assert 3000.0 <= result.T_K <= 30000.0
        assert 1e14 <= result.ne_cm3 <= 1e19
        for c in result.concentrations.values():
            assert 0.0 <= c <= 1.0

    def test_chi_squared_reduced_finite(self):
        lib = _FakeBasisLibrary()
        refiner = SpectralRefiner(lib, max_iterations=3)
        basis = lib.get_basis_matrix_interp(8000.0, 1e17)
        observed = basis[0] * 0.5

        result = refiner.refine(
            wavelength=lib.wavelength,
            observed=observed,
            detected_elements=["Fe"],
            T_init_K=8000.0,
            ne_init_cm3=1e17,
            concentrations_init={"Fe": 0.5},
        )
        assert np.isfinite(result.chi_squared)
        assert np.isfinite(result.chi_squared_reduced)
        assert result.chi_squared_reduced >= 0.0

    def test_observed_resampled_when_wavelength_differs(self):
        lib = _FakeBasisLibrary(n_pixels=64)
        refiner = SpectralRefiner(lib, max_iterations=2)
        other_wl = np.linspace(400.0, 500.0, 32)
        basis = lib.get_basis_matrix_interp(8000.0, 1e17)
        observed_lib = basis[0]
        observed = np.interp(other_wl, lib.wavelength, observed_lib)

        result = refiner.refine(
            wavelength=other_wl,
            observed=observed,
            detected_elements=["Fe"],
            T_init_K=8000.0,
            ne_init_cm3=1e17,
            concentrations_init={"Fe": 0.5},
        )
        assert isinstance(result, RefinementResult)
        assert "Fe" in result.concentrations

    def test_noise_with_matching_length(self):
        lib = _FakeBasisLibrary()
        refiner = SpectralRefiner(lib, max_iterations=3)
        basis = lib.get_basis_matrix_interp(8000.0, 1e17)
        observed = 0.5 * basis[0]
        noise = np.full_like(observed, 0.01)

        result = refiner.refine(
            wavelength=lib.wavelength,
            observed=observed,
            detected_elements=["Fe"],
            T_init_K=8000.0,
            ne_init_cm3=1e17,
            concentrations_init={"Fe": 0.5},
            noise=noise,
        )
        assert np.isfinite(result.chi_squared)

    def test_noise_resampled_when_length_differs(self):
        lib = _FakeBasisLibrary(n_pixels=64)
        refiner = SpectralRefiner(lib, max_iterations=2)
        basis = lib.get_basis_matrix_interp(8000.0, 1e17)
        observed = 0.5 * basis[0]
        noise = np.full(32, 0.01)
        other_wl = np.linspace(400.0, 500.0, 32)

        result = refiner.refine(
            wavelength=other_wl,
            observed=np.interp(other_wl, lib.wavelength, observed),
            detected_elements=["Fe"],
            T_init_K=8000.0,
            ne_init_cm3=1e17,
            concentrations_init={"Fe": 0.5},
            noise=noise,
        )
        assert isinstance(result, RefinementResult)

    def test_initial_values_clipped_to_bounds(self):
        lib = _FakeBasisLibrary()
        refiner = SpectralRefiner(lib, max_iterations=1)
        basis = lib.get_basis_matrix_interp(8000.0, 1e17)
        observed = basis[0]

        result = refiner.refine(
            wavelength=lib.wavelength,
            observed=observed,
            detected_elements=["Fe"],
            T_init_K=1e6,
            ne_init_cm3=1e25,
            concentrations_init={"Fe": 5.0},
        )
        assert 3000.0 <= result.T_K <= 30000.0
        assert 1e14 <= result.ne_cm3 <= 1e19
        assert 0.0 <= result.concentrations["Fe"] <= 1.0

    def test_missing_concentration_defaults_to_uniform(self):
        lib = _FakeBasisLibrary()
        refiner = SpectralRefiner(lib, max_iterations=2)
        basis = lib.get_basis_matrix_interp(8000.0, 1e17)
        observed = 0.5 * basis[0] + 0.5 * basis[1]

        result = refiner.refine(
            wavelength=lib.wavelength,
            observed=observed,
            detected_elements=["Fe", "Ca"],
            T_init_K=8000.0,
            ne_init_cm3=1e17,
            concentrations_init={},
        )
        assert set(result.concentrations.keys()) == {"Fe", "Ca"}

    def test_residual_norm_nonnegative(self):
        lib = _FakeBasisLibrary()
        refiner = SpectralRefiner(lib, max_iterations=3)
        basis = lib.get_basis_matrix_interp(8000.0, 1e17)
        observed = 0.5 * basis[0]

        result = refiner.refine(
            wavelength=lib.wavelength,
            observed=observed,
            detected_elements=["Fe"],
            T_init_K=8000.0,
            ne_init_cm3=1e17,
            concentrations_init={"Fe": 0.5},
        )
        assert result.residual_norm >= 0.0
