"""
Tests for radiation calculations.
"""

import pytest
import numpy as np
from cflibs.radiation.profiles import (
    BroadeningMode,
    gaussian_profile,
    doppler_width,
    apply_gaussian_broadening,
    apply_gaussian_broadening_per_line,
    resolving_power_sigma,
)
from cflibs.radiation.emissivity import calculate_line_emissivity, calculate_spectrum_emissivity


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def test_gaussian_profile():
    """Test Gaussian profile calculation."""
    wavelength = np.linspace(370, 375, 100)
    center = 372.0
    sigma = 0.1
    amplitude = 1.0

    profile = gaussian_profile(wavelength, center, sigma, amplitude)

    assert len(profile) == len(wavelength)
    assert np.all(profile >= 0)

    # Peak should be at center
    peak_idx = np.argmax(profile)
    assert abs(wavelength[peak_idx] - center) < 0.1

    # Profile should be normalized (approximately)
    integral = _trapezoid(profile, wavelength)
    assert abs(integral - amplitude) < 0.1


def test_gaussian_profile_single_value():
    """Test Gaussian profile with single wavelength value."""
    profile = gaussian_profile(372.0, 372.0, 0.1, 1.0)
    assert isinstance(profile, (float, np.floating))
    assert profile > 0


def test_doppler_width():
    """Test Doppler width calculation."""
    width = doppler_width(wavelength_nm=500.0, T_eV=0.86, mass_amu=56.0)  # ~10000 K  # Iron

    assert width > 0
    # Typical Doppler width for Fe at 10000 K should be ~0.01-0.1 nm
    assert 0.001 < width < 1.0


def test_doppler_width_temperature_scaling():
    """Test that Doppler width scales with temperature."""
    w1 = doppler_width(500.0, 0.5, 56.0)
    w2 = doppler_width(500.0, 1.0, 56.0)

    # Higher temperature should give larger width
    assert w2 > w1


def test_apply_gaussian_broadening():
    """Test applying Gaussian broadening to lines."""
    wavelength = np.linspace(370, 375, 100)
    line_wavelengths = np.array([371.99, 373.49])
    line_intensities = np.array([1000.0, 500.0])
    sigma = 0.05

    spectrum = apply_gaussian_broadening(wavelength, line_wavelengths, line_intensities, sigma)

    assert len(spectrum) == len(wavelength)
    assert np.all(spectrum >= 0)
    assert np.any(spectrum > 0)  # Should have some signal


def test_calculate_line_emissivity(sample_transition):
    """Test calculating line emissivity."""
    population = 1e10  # cm^-3

    emissivity = calculate_line_emissivity(sample_transition, population)

    assert emissivity > 0
    # Emissivity should scale with population
    emissivity2 = calculate_line_emissivity(sample_transition, 2 * population)
    assert emissivity2 == pytest.approx(2 * emissivity, rel=0.01)


def test_calculate_line_emissivity_custom_wavelength(sample_transition):
    """Test calculating emissivity with custom wavelength."""
    population = 1e10
    custom_wl = 372.0

    emissivity = calculate_line_emissivity(sample_transition, population, wavelength_nm=custom_wl)

    assert emissivity > 0


def test_calculate_spectrum_emissivity(atomic_db, sample_plasma):
    """Test calculating spectrum emissivity."""
    from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolver

    # Solve for per-species Saha states
    solver = SahaBoltzmannSolver(atomic_db)
    species_states = solver.solve_species_states(sample_plasma)

    # Get transitions
    transitions = atomic_db.get_transitions("Fe", wavelength_min=370.0, wavelength_max=375.0)

    if len(transitions) == 0:
        pytest.skip("No transitions in test database")

    wavelength = np.linspace(370, 375, 100)
    sigma = 0.05

    emissivity = calculate_spectrum_emissivity(
        transitions, species_states, wavelength, sigma, sample_plasma.T_e_eV
    )

    assert len(emissivity) == len(wavelength)
    assert np.all(emissivity >= 0)
    assert np.any(emissivity > 0)


@pytest.mark.requires_jax
def test_calculate_spectrum_emissivity_jax(atomic_db, sample_plasma):
    """Test calculating spectrum emissivity with JAX path."""
    try:
        import jax  # noqa: F401
    except ImportError:
        pytest.skip("JAX not installed")

    from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolver

    solver = SahaBoltzmannSolver(atomic_db)
    species_states = solver.solve_species_states(sample_plasma)
    transitions = atomic_db.get_transitions("Fe", wavelength_min=370.0, wavelength_max=375.0)

    if len(transitions) == 0:
        pytest.skip("No transitions in test database")

    wavelength = np.linspace(370, 375, 100)
    sigma = 0.05

    emissivity_np = calculate_spectrum_emissivity(
        transitions, species_states, wavelength, sigma, sample_plasma.T_e_eV, use_jax=False
    )
    emissivity_jax = calculate_spectrum_emissivity(
        transitions, species_states, wavelength, sigma, sample_plasma.T_e_eV, use_jax=True
    )

    assert len(emissivity_jax) == len(wavelength)
    assert np.all(emissivity_jax >= 0)
    # JAX uses float32 by default, NumPy uses float64 - compare peak regions
    # with relaxed tolerance; float32 has ~7 decimal digits precision
    # so rtol=1e-2 (1%) is appropriate for numerical algorithm comparisons
    peak_mask = emissivity_np > 1.0  # Focus on significant values
    if np.any(peak_mask):
        assert np.allclose(
            emissivity_jax[peak_mask], emissivity_np[peak_mask], rtol=1e-2, atol=1e-2
        )
    else:
        # Fallback for degenerate cases
        assert np.allclose(emissivity_jax, emissivity_np, rtol=1e-2, atol=1e-2)


def test_calculate_spectrum_emissivity_no_populations(atomic_db):
    """Test spectrum emissivity with no matching species states."""
    transitions = atomic_db.get_transitions("Fe", wavelength_min=370, wavelength_max=375)
    species_states = {}  # Empty species states
    wavelength = np.linspace(370, 375, 100)

    emissivity = calculate_spectrum_emissivity(transitions, species_states, wavelength, 0.05, 0.86)

    # Should return zeros
    assert np.all(emissivity == 0)


# --- BroadeningMode Enum Tests ---


def test_broadening_mode_enum_values():
    """Test BroadeningMode enum has expected values."""
    assert BroadeningMode.LEGACY == "legacy"
    assert BroadeningMode.NIST_PARITY == "nist_parity"
    assert BroadeningMode.PHYSICAL_DOPPLER == "physical_doppler"


# --- resolving_power_sigma Tests ---


def test_resolving_power_sigma_basic():
    """Test resolving_power_sigma at known values."""
    # R=1000, lambda=500 nm => FWHM=0.5 nm, sigma=0.5/2.355
    sigma = resolving_power_sigma(500.0, 1000.0)
    expected = 500.0 / 1000.0 / 2.355
    assert sigma == pytest.approx(expected, rel=1e-6)


def test_resolving_power_sigma_wavelength_proportional():
    """Test that sigma scales linearly with wavelength at fixed R."""
    s1 = resolving_power_sigma(200.0, 1000.0)
    s2 = resolving_power_sigma(400.0, 1000.0)
    assert s2 == pytest.approx(2.0 * s1, rel=1e-6)


def test_resolving_power_sigma_inversely_proportional_to_R():
    """Test that sigma scales inversely with resolving power."""
    s1 = resolving_power_sigma(500.0, 500.0)
    s2 = resolving_power_sigma(500.0, 1000.0)
    assert s1 == pytest.approx(2.0 * s2, rel=1e-6)


# --- apply_gaussian_broadening_per_line Tests ---


def test_apply_gaussian_broadening_per_line_basic():
    """Test per-line broadening produces spectrum."""
    wavelength = np.linspace(370, 375, 500)
    line_wavelengths = np.array([371.99, 373.49])
    line_intensities = np.array([1000.0, 500.0])
    sigmas = np.array([0.05, 0.10])

    spectrum = apply_gaussian_broadening_per_line(
        wavelength, line_wavelengths, line_intensities, sigmas
    )

    assert len(spectrum) == len(wavelength)
    assert np.all(spectrum >= 0)
    assert np.any(spectrum > 0)


def test_apply_gaussian_broadening_per_line_wider_sigma_gives_lower_peak():
    """Test that wider sigma gives lower peak height (area preserved)."""
    wavelength = np.linspace(370, 376, 1000)
    line_wl = np.array([373.0])
    line_int = np.array([1000.0])

    spec_narrow = apply_gaussian_broadening_per_line(
        wavelength, line_wl, line_int, np.array([0.02])
    )
    spec_wide = apply_gaussian_broadening_per_line(wavelength, line_wl, line_int, np.array([0.10]))

    assert spec_narrow.max() > spec_wide.max()


def test_apply_gaussian_broadening_per_line_matches_scalar():
    """Test per-line with uniform sigmas matches scalar broadening."""
    wavelength = np.linspace(370, 375, 500)
    line_wavelengths = np.array([371.99, 373.49])
    line_intensities = np.array([1000.0, 500.0])
    sigma = 0.05

    spec_scalar = apply_gaussian_broadening(wavelength, line_wavelengths, line_intensities, sigma)
    spec_per_line = apply_gaussian_broadening_per_line(
        wavelength, line_wavelengths, line_intensities, np.full(2, sigma)
    )

    np.testing.assert_allclose(spec_scalar, spec_per_line, rtol=1e-10)


# --- Per-line sigma emissivity Tests ---


def test_calculate_spectrum_emissivity_per_line_sigma(atomic_db, sample_plasma):
    """Test spectrum emissivity with per-line sigma array."""
    from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolver
    from cflibs.radiation.emissivity import upper_level_population_cm3

    solver = SahaBoltzmannSolver(atomic_db)
    species_states = solver.solve_species_states(sample_plasma)
    transitions = atomic_db.get_transitions("Fe", wavelength_min=370.0, wavelength_max=375.0)

    if len(transitions) == 0:
        pytest.skip("No transitions in test database")

    wavelength = np.linspace(370, 375, 100)

    # Count how many transitions will have non-zero populations
    n_with_pop = 0
    for t in transitions:
        state = species_states.get((t.element, t.ionization_stage))
        if state is None:
            continue
        if upper_level_population_cm3(state, t.g_k, t.E_k_ev, sample_plasma.T_e_eV) > 0:
            n_with_pop += 1

    if n_with_pop == 0:
        pytest.skip("No transitions with populations")

    sigmas = np.full(n_with_pop, 0.05)
    emissivity = calculate_spectrum_emissivity(
        transitions, species_states, wavelength, sigmas, sample_plasma.T_e_eV
    )

    assert len(emissivity) == len(wavelength)
    assert np.all(emissivity >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
