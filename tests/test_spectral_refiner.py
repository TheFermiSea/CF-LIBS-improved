"""
Tests for SpectralRefiner — gradient-based refinement of NNLS results.
"""

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from cflibs.inversion.solve.spectral_refiner import RefinementResult, SpectralRefiner  # noqa: E402

pytestmark = [pytest.mark.requires_db]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_basis_library(request):
    """Generate a small basis library for testing (module-scoped for speed)."""
    import tempfile
    from pathlib import Path

    from cflibs.manifold.basis_library import (
        BasisLibrary,
        BasisLibraryConfig,
        BasisLibraryGenerator,
    )

    db_path = "ASD_da/libs_production.db"
    if not Path(db_path).exists():
        pytest.skip("Database not found")

    tmp_dir = tempfile.mkdtemp()
    cfg = BasisLibraryConfig(
        db_path=db_path,
        output_path=str(Path(tmp_dir) / "test_basis.h5"),
        wavelength_range=(370.0, 380.0),
        pixels=256,
        temperature_range=(6000.0, 10000.0),
        temperature_steps=3,
        density_range=(1e16, 1e17),
        density_steps=2,
    )
    gen = BasisLibraryGenerator(cfg)
    gen.generate()

    lib = BasisLibrary(cfg.output_path)
    yield lib
    lib.close()


@pytest.fixture
def refiner(small_basis_library):
    """SpectralRefiner with default settings."""
    return SpectralRefiner(basis_library=small_basis_library, max_iterations=20)


def _make_synthetic(basis_library, elements_fracs, T_K=8000.0, ne=5e16, noise_level=0.001):
    """Create a synthetic spectrum as weighted sum of basis spectra + noise."""
    wl = basis_library.wavelength
    spectrum = np.zeros(len(wl))
    for el, frac in elements_fracs.items():
        try:
            el_spec = basis_library.get_element_spectrum(el, T_K, ne)
            spectrum += frac * el_spec
        except KeyError:
            pass
    rng = np.random.RandomState(42)
    spectrum += noise_level * rng.normal(size=len(spectrum))
    spectrum = np.maximum(spectrum, 0.0)
    return wl, spectrum


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRefinementResult:
    def test_dataclass_creation(self):
        """RefinementResult can be instantiated with expected fields."""
        r = RefinementResult(
            T_K=8000.0,
            ne_cm3=1e17,
            concentrations={"Fe": 0.7, "Cr": 0.3},
            residual_norm=0.01,
            n_iterations=5,
            converged=True,
            chi_squared=10.0,
            chi_squared_reduced=1.2,
        )
        assert r.T_K == 8000.0
        assert r.ne_cm3 == 1e17
        assert r.concentrations == {"Fe": 0.7, "Cr": 0.3}
        assert r.converged is True
        assert r.chi_squared_reduced == 1.2

    def test_dataclass_fields_are_accessible(self):
        r = RefinementResult(
            T_K=6000.0,
            ne_cm3=5e16,
            concentrations={"Ti": 1.0},
            residual_norm=0.005,
            n_iterations=10,
            converged=False,
            chi_squared=50.0,
            chi_squared_reduced=5.0,
        )
        assert r.n_iterations == 10
        assert r.residual_norm == 0.005
        assert r.converged is False


class TestSpectralRefinerSingleElement:
    def test_refine_fe_recovers_temperature(self, refiner, small_basis_library):
        """Refine with single element (Fe) recovers input T within 20%."""
        true_T = 8000.0
        true_ne = 5e16
        wl, spec = _make_synthetic(
            small_basis_library, {"Fe": 1.0}, T_K=true_T, ne=true_ne, noise_level=0.0005
        )

        # Perturb initial conditions away from truth
        T_init = 7000.0
        ne_init = 3e16

        result = refiner.refine(
            wavelength=wl,
            observed=spec,
            detected_elements=["Fe"],
            T_init_K=T_init,
            ne_init_cm3=ne_init,
            concentrations_init={"Fe": 1.0},
        )

        assert isinstance(result, RefinementResult)
        assert result.T_K > 0
        # Temperature should be within 20% of truth
        assert (
            abs(result.T_K - true_T) / true_T < 0.20
        ), f"T recovered={result.T_K:.0f} K, expected ~{true_T:.0f} K"

    def test_refine_converges_for_clean_data(self, small_basis_library):
        """Refine converges (converged=True) for clean synthetic data."""
        # Use a higher iteration budget so L-BFGS-B can reach convergence
        high_iter_refiner = SpectralRefiner(basis_library=small_basis_library, max_iterations=100)
        wl, spec = _make_synthetic(
            small_basis_library, {"Fe": 1.0}, T_K=8000.0, ne=5e16, noise_level=0.0001
        )

        result = high_iter_refiner.refine(
            wavelength=wl,
            observed=spec,
            detected_elements=["Fe"],
            T_init_K=8000.0,
            ne_init_cm3=5e16,
            concentrations_init={"Fe": 1.0},
        )

        assert result.converged is True


class TestSpectralRefinerTwoElements:
    def test_refine_two_elements_relative_concentrations(self, refiner, small_basis_library):
        """Refine with two elements recovers relative concentrations."""
        true_T = 8000.0
        true_ne = 5e16
        wl, spec = _make_synthetic(
            small_basis_library,
            {"Fe": 0.7, "Cr": 0.3},
            T_K=true_T,
            ne=true_ne,
            noise_level=0.0005,
        )

        result = refiner.refine(
            wavelength=wl,
            observed=spec,
            detected_elements=["Fe", "Cr"],
            T_init_K=7500.0,
            ne_init_cm3=4e16,
            concentrations_init={"Fe": 0.5, "Cr": 0.5},
        )

        assert isinstance(result, RefinementResult)
        # Both elements should be present in result
        assert "Fe" in result.concentrations
        assert "Cr" in result.concentrations
        # Fe should have higher concentration than Cr
        if result.concentrations["Fe"] > 0 and result.concentrations["Cr"] > 0:
            assert result.concentrations["Fe"] > result.concentrations["Cr"], (
                f"Expected Fe > Cr, got Fe={result.concentrations['Fe']:.4f}, "
                f"Cr={result.concentrations['Cr']:.4f}"
            )


class TestSpectralRefinerEdgeCases:
    def test_empty_element_list(self, refiner, small_basis_library):
        """Refine handles empty element list gracefully."""
        wl = small_basis_library.wavelength
        obs = np.ones(len(wl)) * 0.01

        result = refiner.refine(
            wavelength=wl,
            observed=obs,
            detected_elements=[],
            T_init_K=8000.0,
            ne_init_cm3=1e17,
            concentrations_init={},
        )

        assert isinstance(result, RefinementResult)
        assert result.concentrations == {}
        assert result.converged is True
        assert result.n_iterations == 0

    def test_single_pixel(self, refiner, small_basis_library):
        """Refine handles single-pixel spectrum without error."""
        wl = np.array([375.0])
        obs = np.array([0.5])

        result = refiner.refine(
            wavelength=wl,
            observed=obs,
            detected_elements=["Fe"],
            T_init_K=8000.0,
            ne_init_cm3=5e16,
            concentrations_init={"Fe": 1.0},
        )

        assert isinstance(result, RefinementResult)

    def test_unknown_element_ignored(self, refiner, small_basis_library):
        """Elements not in the basis library are silently skipped."""
        wl, spec = _make_synthetic(small_basis_library, {"Fe": 1.0}, noise_level=0.001)

        result = refiner.refine(
            wavelength=wl,
            observed=spec,
            detected_elements=["Fe", "Unobtanium"],
            T_init_K=8000.0,
            ne_init_cm3=5e16,
            concentrations_init={"Fe": 0.9, "Unobtanium": 0.1},
        )

        assert "Fe" in result.concentrations
        assert "Unobtanium" not in result.concentrations

    def test_all_unknown_elements(self, refiner, small_basis_library):
        """If all elements are unknown, return gracefully."""
        wl = small_basis_library.wavelength
        obs = np.ones(len(wl)) * 0.01

        result = refiner.refine(
            wavelength=wl,
            observed=obs,
            detected_elements=["Unobtanium"],
            T_init_K=8000.0,
            ne_init_cm3=1e17,
            concentrations_init={"Unobtanium": 1.0},
        )

        assert isinstance(result, RefinementResult)
        assert result.n_iterations == 0


class TestSpectralRefinerFitQuality:
    def test_chi_squared_reduced_reasonable(self, refiner, small_basis_library):
        """chi_squared_reduced < 10 for synthetic data with known noise."""
        true_T = 8000.0
        true_ne = 5e16
        noise_level = 0.001
        wl, spec = _make_synthetic(
            small_basis_library, {"Fe": 1.0}, T_K=true_T, ne=true_ne, noise_level=noise_level
        )

        # Provide a reasonable noise estimate
        noise = np.full(len(wl), noise_level)

        result = refiner.refine(
            wavelength=wl,
            observed=spec,
            detected_elements=["Fe"],
            T_init_K=8000.0,
            ne_init_cm3=5e16,
            concentrations_init={"Fe": 1.0},
            noise=noise,
        )

        assert (
            result.chi_squared_reduced < 10.0
        ), f"chi2_red={result.chi_squared_reduced:.2f}, expected < 10"

    def test_residual_norm_decreases(self, refiner, small_basis_library):
        """Residual norm after refinement should be no worse than initial."""
        wl, spec = _make_synthetic(
            small_basis_library, {"Fe": 1.0}, T_K=8000.0, ne=5e16, noise_level=0.001
        )
        lib_wl = small_basis_library.wavelength
        obs = np.interp(lib_wl, wl, spec)

        # Compute initial residual
        basis = small_basis_library.get_basis_matrix_interp(7000.0, 3e16)
        lib_els = small_basis_library.elements
        fe_idx = lib_els.index("Fe")
        initial_model = 1.0 * basis[fe_idx, :]
        initial_resid = float(np.linalg.norm(obs - initial_model))

        result = refiner.refine(
            wavelength=wl,
            observed=spec,
            detected_elements=["Fe"],
            T_init_K=7000.0,
            ne_init_cm3=3e16,
            concentrations_init={"Fe": 1.0},
        )

        assert (
            result.residual_norm <= initial_resid * 1.01
        ), f"Residual norm increased: {result.residual_norm:.6f} > {initial_resid:.6f}"
