"""Closure-constraint tests for :class:`SpectralRefiner` (audit Family 4, bug a).

These tests use a lightweight fake basis library (no atomic DB / h5py
required) so they run fast and watchdog-safe.  They encode the validation
gate that the refiner must enforce ``sum(C_s) = 1`` and expose a
``closure_residual`` diagnostic, with composition decoupled from the global
intensity amplitude.

The oracle composition (Fe=0.7, Ni=0.3) is independent of the refiner: the
synthetic spectrum is built directly as ``amplitude * (C @ basis)`` from
distinguishable single-element basis shapes, so the expected fractions are
not derived from the (previously unconstrained) optimizer.
"""

import numpy as np
import pytest

from cflibs.inversion.solve.spectral_refiner import (
    CLOSURE_TOLERANCE,
    SpectralRefiner,
)


class _FakeBasisLibrary:
    """Two distinguishable single-element basis spectra (Fe, Ni)."""

    def __init__(self, n_pixels: int = 128):
        self.wavelength = np.linspace(370.0, 390.0, n_pixels)
        self.elements = ["Fe", "Ni"]
        self._n_pixels = n_pixels

    def _gaussian(self, center: float, fwhm: float = 1.5, amp: float = 1.0) -> np.ndarray:
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        return amp * np.exp(-((self.wavelength - center) ** 2) / (2 * sigma**2))

    def get_basis_matrix_interp(self, T_K: float, ne_cm3: float) -> np.ndarray:
        # Two well-separated lines per element so the design is identifiable.
        fe = self._gaussian(374.0) + 0.6 * self._gaussian(382.0)
        ni = self._gaussian(378.0) + 0.5 * self._gaussian(386.0)
        return np.vstack([fe, ni])


def _make_spectrum(lib: _FakeBasisLibrary, conc: dict, amplitude: float) -> np.ndarray:
    basis = lib.get_basis_matrix_interp(8000.0, 5e16)
    c = np.array([conc[el] for el in lib.elements])
    return amplitude * (c @ basis)


def test_refiner_enforces_sum_to_one_and_exposes_closure_residual():
    """Validation gate (a): Fe=0.7, Ni=0.3 → sum(C)=1±1e-6, residual exposed."""
    lib = _FakeBasisLibrary()
    refiner = SpectralRefiner(lib, max_iterations=200)
    true_conc = {"Fe": 0.7, "Ni": 0.3}
    observed = _make_spectrum(lib, true_conc, amplitude=3.0)

    result = refiner.refine(
        wavelength=lib.wavelength,
        observed=observed,
        detected_elements=["Fe", "Ni"],
        T_init_K=8000.0,
        ne_init_cm3=5e16,
        concentrations_init={"Fe": 0.5, "Ni": 0.5},
    )

    # Closure: concentrations sum to 1 within tolerance, and the diagnostic
    # is exposed and consistent.
    total = sum(result.concentrations.values())
    assert abs(total - 1.0) <= CLOSURE_TOLERANCE
    assert hasattr(result, "closure_residual")
    assert result.closure_residual <= CLOSURE_TOLERANCE
    np.testing.assert_allclose(result.closure_residual, abs(total - 1.0), atol=1e-12)

    # Composition is recovered (independent oracle), not just normalized noise.
    assert abs(result.concentrations["Fe"] - 0.7) < 0.05
    assert abs(result.concentrations["Ni"] - 0.3) < 0.05


def test_refiner_decouples_composition_from_amplitude():
    """Validation gate (a): scaling all intensities by 1e3 leaves composition
    (and closure) invariant; the amplitude absorbs the intensity scale."""
    lib = _FakeBasisLibrary()
    refiner = SpectralRefiner(lib, max_iterations=200)
    true_conc = {"Fe": 0.7, "Ni": 0.3}

    obs_lo = _make_spectrum(lib, true_conc, amplitude=2.0)
    obs_hi = obs_lo * 1.0e3

    res_lo = refiner.refine(
        wavelength=lib.wavelength,
        observed=obs_lo,
        detected_elements=["Fe", "Ni"],
        T_init_K=8000.0,
        ne_init_cm3=5e16,
        concentrations_init={"Fe": 0.5, "Ni": 0.5},
    )
    res_hi = refiner.refine(
        wavelength=lib.wavelength,
        observed=obs_hi,
        detected_elements=["Fe", "Ni"],
        T_init_K=8000.0,
        ne_init_cm3=5e16,
        concentrations_init={"Fe": 0.5, "Ni": 0.5},
    )

    # Composition unchanged under a global intensity rescale.
    for el in ("Fe", "Ni"):
        assert abs(res_lo.concentrations[el] - res_hi.concentrations[el]) < 1e-3
    # Both satisfy closure.
    assert abs(sum(res_lo.concentrations.values()) - 1.0) <= CLOSURE_TOLERANCE
    assert abs(sum(res_hi.concentrations.values()) - 1.0) <= CLOSURE_TOLERANCE
    # The amplitude (not the composition) carries the 1e3 factor.
    assert abs(res_hi.amplitude / res_lo.amplitude - 1.0e3) / 1.0e3 < 0.05


def test_refiner_single_element_trivial_closure():
    lib = _FakeBasisLibrary()
    refiner = SpectralRefiner(lib, max_iterations=50)
    observed = _make_spectrum(lib, {"Fe": 1.0, "Ni": 0.0}, amplitude=1.5)

    result = refiner.refine(
        wavelength=lib.wavelength,
        observed=observed,
        detected_elements=["Fe"],
        T_init_K=8000.0,
        ne_init_cm3=5e16,
        concentrations_init={"Fe": 1.0},
    )

    assert result.concentrations["Fe"] == pytest.approx(1.0)
    assert result.closure_residual <= CLOSURE_TOLERANCE
