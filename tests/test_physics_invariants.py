"""Physics invariant and unit sanity tests for CF-LIBS forward model."""

import numpy as np
import pytest
from cflibs.plasma.state import SingleZoneLTEPlasma
from cflibs.instrument.model import InstrumentModel
from cflibs.radiation import SpectrumModel
from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolver
from cflibs.core.constants import EV_TO_K

pytestmark = [pytest.mark.requires_db, pytest.mark.physics]


@pytest.fixture
def solver(atomic_db):
    return SahaBoltzmannSolver(atomic_db)


def _make_spectrum_model(atomic_db, T_K, n_e, element="Fe", **kw):
    """Helper to build a SpectrumModel for a single-element plasma."""
    plasma = SingleZoneLTEPlasma(T_e=T_K, n_e=n_e, species={element: 1e16})
    instrument = InstrumentModel(resolution_fwhm_nm=0.05)
    return SpectrumModel(
        plasma=plasma,
        atomic_db=atomic_db,
        instrument=instrument,
        lambda_min=kw.get("lambda_min", 370.0),
        lambda_max=kw.get("lambda_max", 376.0),
        delta_lambda=kw.get("delta_lambda", 0.01),
    )


class TestNonNegativeIntensity:
    """All synthetic spectra must have non-negative intensity."""

    @pytest.mark.parametrize(
        "T_K, n_e",
        [
            (8000.0, 1e16),
            (10000.0, 1e17),
            (15000.0, 1e17),
            (20000.0, 1e18),
        ],
        ids=["8kK-1e16", "10kK-1e17", "15kK-1e17", "20kK-1e18"],
    )
    def test_spectrum_nonnegative(self, atomic_db, T_K, n_e):
        model = _make_spectrum_model(atomic_db, T_K, n_e)
        _, intensity = model.compute_spectrum()
        assert np.all(
            np.isfinite(intensity)
        ), f"Non-finite values in spectrum at T={T_K} K, n_e={n_e:.0e}"
        assert np.all(
            intensity >= 0
        ), f"Negative intensity at T={T_K} K, n_e={n_e:.0e}: min={intensity.min():.4e}"


class TestBoltzmannLineRatio:
    """Line ratio I(E_high)/I(E_low) must increase with temperature."""

    def test_line_ratio_increases_with_temperature(self, atomic_db):
        T_low, T_high, n_e = 8000.0, 15000.0, 1e17
        model_low = _make_spectrum_model(atomic_db, T_low, n_e)
        model_high = _make_spectrum_model(atomic_db, T_high, n_e)
        _, I_low = model_low.compute_spectrum()
        _, I_high = model_high.compute_spectrum()
        wl = model_low.wavelength

        def peak_at(spectrum, center, window=0.05):
            mask = np.abs(wl - center) <= window
            return spectrum[mask].max() if mask.any() else 0.0

        ratio_lowT = peak_at(I_low, 371.99) / max(peak_at(I_low, 374.95), 1e-30)
        ratio_highT = peak_at(I_high, 371.99) / max(peak_at(I_high, 374.95), 1e-30)
        if peak_at(I_low, 374.95) < 1e-30:
            pytest.skip("Line intensities too weak for ratio test")
        assert ratio_highT > ratio_lowT, (
            f"Boltzmann violation: ratio@{T_low}K={ratio_lowT:.4f}, "
            f"ratio@{T_high}K={ratio_highT:.4f}"
        )


class TestPathLengthScaling:
    """With self-absorption, I(2L) < 2 * I(L) for optically thick lines, but approaches 2*I(L) for thin lines."""

    def test_path_length_self_absorption(self, atomic_db):
        plasma = SingleZoneLTEPlasma(T_e=10000.0, n_e=1e17, species={"Fe": 1e16})
        instrument = InstrumentModel(resolution_fwhm_nm=0.05)
        common = dict(
            plasma=plasma,
            atomic_db=atomic_db,
            instrument=instrument,
            lambda_min=370.0,
            lambda_max=376.0,
            delta_lambda=0.01,
        )
        _, I1 = SpectrumModel(**common, path_length_m=0.01).compute_spectrum()
        _, I2 = SpectrumModel(**common, path_length_m=0.02).compute_spectrum()

        # Intensity must be monotonically increasing with length
        assert np.all(I2 >= I1)
        # Intensity must be bounded by optically thin limit (2 * I1)
        # We add small tolerance for floating point issues on zero regions
        assert np.all(I2 <= 2.0 * I1 + 1e-12)
        # Check that self-absorption is actually occurring on strong lines
        # where I2 is noticeably less than 2*I1
        strong_lines = I1 > 0.1 * np.max(I1)
        assert np.any(I2[strong_lines] < 1.95 * I1[strong_lines]), "Self-absorption not observed"


class TestPartitionFunction:
    """Partition functions must increase monotonically with temperature."""

    @pytest.mark.parametrize("element,stage", [("Fe", 1), ("H", 1)])
    def test_increases_with_T(self, solver, element, stage):
        temps = [0.5, 0.8, 1.0, 1.5, 2.0]
        U = [solver.calculate_partition_function(element, stage, T) for T in temps]
        for i in range(len(U) - 1):
            assert U[i + 1] >= U[i], (
                f"U({element} {stage}) decreased: "
                f"U({temps[i]:.1f} eV)={U[i]:.4f} > U({temps[i+1]:.1f} eV)={U[i+1]:.4f}"
            )

    @pytest.mark.parametrize("element,stage", [("Fe", 1), ("H", 1)])
    def test_positive(self, solver, element, stage):
        for T_eV in [0.3, 0.8, 1.5, 3.0]:
            U = solver.calculate_partition_function(element, stage, T_eV)
            assert U > 0, f"U({element} {stage}, {T_eV} eV) = {U}"


class TestIonizationBalance:
    """Saha equation invariants."""

    def test_fractions_sum_to_one(self, solver):
        total = sum(solver.get_ionization_fractions("Fe", 1.0, 1e17).values())
        assert abs(total - 1.0) < 1e-10, f"Sum = {total:.12f}"

    def test_fractions_in_unit_interval(self, solver):
        for stage, f in solver.get_ionization_fractions("Fe", 1.0, 1e17).items():
            assert 0.0 <= f <= 1.0, f"Stage {stage}: {f}"

    def test_higher_T_more_ionization(self, solver):
        f_low = solver.get_ionization_fractions("Fe", 0.5, 1e17)
        f_high = solver.get_ionization_fractions("Fe", 2.0, 1e17)
        assert f_high.get(1, 0) < f_low.get(1, 0), (
            f"Neutral fraction should decrease: "
            f"f@0.5eV={f_low.get(1,0):.4f}, f@2.0eV={f_high.get(1,0):.4f}"
        )

    def test_higher_ne_suppresses_ionization(self, solver):
        f_lo = solver.get_ionization_fractions("Fe", 1.0, 1e16)
        f_hi = solver.get_ionization_fractions("Fe", 1.0, 1e18)
        assert f_hi.get(1, 0) >= f_lo.get(
            1, 0
        ), "Le Chatelier: neutral fraction should increase with n_e"


class TestUnitConsistency:
    """Verify physical units are consistent through the pipeline."""

    def test_temperature_roundtrip(self):
        T_K = 11604.5
        assert abs(T_K - (T_K / EV_TO_K) * EV_TO_K) < 1e-6

    def test_eV_to_K_constant(self):
        assert abs(EV_TO_K - 11604.5) < 1.0, f"EV_TO_K={EV_TO_K}"

    def test_wavelength_grid(self, atomic_db):
        model = _make_spectrum_model(atomic_db, 10000.0, 1e17)
        wl = model.wavelength
        assert np.all(np.diff(wl) > 0), "Must be monotonic"
        assert wl[0] >= 370.0 - 0.01
        assert wl[-1] <= 376.0 + 0.01
        np.testing.assert_allclose(np.diff(wl), 0.01, rtol=1e-10)


# end of tests
