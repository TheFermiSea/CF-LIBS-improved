"""Tests for BIC-based model selection (backward elimination)."""

import numpy as np

from cflibs.inversion.model_selection import (
    ModelSelectionResult,
    bic_prune_elements,
    boltzmann_consistency_filter,
    _compute_bic,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FWHM_TO_SIGMA = 1.0 / 2.3548200450309493


def _make_element_basis(wavelength: np.ndarray, line_centers: list, fwhm: float = 0.1):
    """Create a synthetic element basis spectrum from Gaussian peaks."""
    sigma = fwhm * FWHM_TO_SIGMA
    spectrum = np.zeros_like(wavelength)
    for center, amplitude in line_centers:
        spectrum += amplitude * np.exp(-0.5 * ((wavelength - center) / sigma) ** 2)
    area = np.sum(spectrum)
    if area > 0:
        spectrum /= area
    return spectrum


def _make_synthetic_problem(
    n_pixels: int = 500,
    wl_range: tuple = (300.0, 500.0),
    true_elements: dict = None,
    spurious_elements: dict = None,
    noise_sigma: float = 0.001,
    seed: int = 42,
):
    """Build a synthetic NNLS problem with known true and spurious elements.

    Parameters
    ----------
    true_elements : dict
        Mapping element name -> (line_centers, true_coefficient).
        line_centers is a list of (center_nm, amplitude) tuples.
    spurious_elements : dict
        Same format but these elements are NOT in the observed spectrum.

    Returns
    -------
    dict with keys: wavelength, observed, basis_matrix, element_list,
    coefficients, noise_variance
    """
    rng = np.random.default_rng(seed)
    wavelength = np.linspace(*wl_range, n_pixels)

    if true_elements is None:
        true_elements = {
            "Fe": ([(350.0, 1.0), (370.0, 0.5), (400.0, 0.8)], 5.0),
        }
    if spurious_elements is None:
        spurious_elements = {}

    all_elements = {}
    all_elements.update(true_elements)
    all_elements.update(spurious_elements)

    element_list = list(all_elements.keys())

    # Build basis matrix (n_elements + continuum, n_pixels)
    # Add one continuum column (constant offset)
    basis_rows = []
    true_coeffs = []
    for el in element_list:
        line_centers, coeff = all_elements[el]
        basis = _make_element_basis(wavelength, line_centers)
        basis_rows.append(basis)
        if el in true_elements:
            true_coeffs.append(coeff)
        else:
            true_coeffs.append(0.0)

    # Continuum: flat baseline
    continuum = np.ones(n_pixels) / n_pixels
    basis_rows.append(continuum)
    true_coeffs.append(0.1)  # small continuum contribution

    basis_matrix = np.array(basis_rows)  # (n_components, n_pixels)

    # Synthesize observed spectrum from true elements only
    true_coeffs_arr = np.array(true_coeffs)
    observed = basis_matrix.T @ true_coeffs_arr
    noise = rng.normal(0, noise_sigma, n_pixels)
    observed += noise
    observed = np.maximum(observed, 0.0)

    # For the initial NNLS solve, provide coefficients that include a
    # small spurious contribution for spurious elements
    init_coeffs = true_coeffs_arr.copy()
    for i, el in enumerate(element_list):
        if el in spurious_elements:
            # Give a small but nonzero initial coefficient
            init_coeffs[i] = 0.01

    return {
        "wavelength": wavelength,
        "observed": observed,
        "basis_matrix": basis_matrix,
        "element_list": element_list,
        "coefficients": init_coeffs,
        "noise_variance": noise_sigma**2,
    }


# ---------------------------------------------------------------------------
# Tests: bic_prune_elements
# ---------------------------------------------------------------------------


class TestBicPruneElements:
    """Tests for the BIC backward elimination algorithm."""

    def test_removes_spurious_element(self):
        """1 true element + 1 spurious -> removes spurious."""
        problem = _make_synthetic_problem(
            true_elements={
                "Fe": ([(350.0, 1.0), (370.0, 0.5), (400.0, 0.8)], 5.0),
            },
            spurious_elements={
                "Na": ([(320.0, 0.3), (460.0, 0.2)], 0.0),
            },
            noise_sigma=0.001,
        )

        result = bic_prune_elements(
            observed=problem["observed"],
            basis_matrix=problem["basis_matrix"],
            element_list=problem["element_list"],
            element_coefficients=problem["coefficients"],
            noise_variance=problem["noise_variance"],
        )

        assert isinstance(result, ModelSelectionResult)
        assert "Fe" in result.selected_elements
        # Na should be removed (zero coefficient -> initially inactive)
        assert "Na" not in result.selected_elements

    def test_keeps_both_true_elements(self):
        """2 true elements -> keeps both."""
        problem = _make_synthetic_problem(
            true_elements={
                "Fe": ([(350.0, 1.0), (370.0, 0.5)], 5.0),
                "Cu": ([(420.0, 0.8), (450.0, 0.6)], 3.0),
            },
            spurious_elements={},
            noise_sigma=0.001,
        )

        result = bic_prune_elements(
            observed=problem["observed"],
            basis_matrix=problem["basis_matrix"],
            element_list=problem["element_list"],
            element_coefficients=problem["coefficients"],
            noise_variance=problem["noise_variance"],
        )

        assert "Fe" in result.selected_elements
        assert "Cu" in result.selected_elements

    def test_all_zero_coefficients(self):
        """All zero coefficients -> returns empty."""
        n_pixels = 200
        basis_matrix = np.random.default_rng(0).random((3, n_pixels))
        observed = np.ones(n_pixels) * 0.5
        element_list = ["Fe", "Cu"]
        coefficients = np.zeros(3)

        result = bic_prune_elements(
            observed=observed,
            basis_matrix=basis_matrix,
            element_list=element_list,
            element_coefficients=coefficients,
            noise_variance=0.01,
        )

        assert result.selected_elements == []
        assert set(result.removed_elements) == {"Fe", "Cu"}

    def test_single_element_kept(self):
        """Single element with strong signal -> keeps it."""
        problem = _make_synthetic_problem(
            true_elements={
                "Fe": ([(350.0, 1.0), (370.0, 0.5), (400.0, 0.8)], 5.0),
            },
            spurious_elements={},
            noise_sigma=0.001,
        )

        result = bic_prune_elements(
            observed=problem["observed"],
            basis_matrix=problem["basis_matrix"],
            element_list=problem["element_list"],
            element_coefficients=problem["coefficients"],
            noise_variance=problem["noise_variance"],
        )

        assert "Fe" in result.selected_elements
        assert len(result.selected_elements) == 1

    def test_bic_decreases_when_removing_noise_element(self):
        """BIC should decrease when removing a noise-only element."""
        # Create a problem where a spurious element has a small nonzero coeff
        n_pixels = 500
        wl = np.linspace(300.0, 500.0, n_pixels)

        # True element with strong signal
        fe_basis = _make_element_basis(wl, [(350.0, 1.0), (370.0, 0.5)])
        # Spurious element whose lines do NOT overlap with observed
        na_basis = _make_element_basis(wl, [(480.0, 0.3)])
        # Continuum
        cont = np.ones(n_pixels) / n_pixels

        basis_matrix = np.array([fe_basis, na_basis, cont])
        true_observed = 5.0 * fe_basis + 0.1 * cont
        rng = np.random.default_rng(42)
        observed = true_observed + rng.normal(0, 0.0005, n_pixels)
        observed = np.maximum(observed, 0.0)

        # Give Na a small spurious coefficient
        coefficients = np.array([5.0, 0.05, 0.1])

        result = bic_prune_elements(
            observed=observed,
            basis_matrix=basis_matrix,
            element_list=["Fe", "Na"],
            element_coefficients=coefficients,
            noise_variance=0.0005**2,
        )

        assert result.bic_final <= result.bic_initial

    def test_result_dataclass_fields(self):
        """ModelSelectionResult has all expected fields."""
        result = ModelSelectionResult(
            selected_elements=["Fe"],
            removed_elements=["Na"],
            concentrations={"Fe": 1.0},
            bic_final=-100.0,
            bic_initial=-90.0,
            boltzmann_results={"Fe": {"T_K": 10000, "R_squared": 0.95}},
        )

        assert result.selected_elements == ["Fe"]
        assert result.removed_elements == ["Na"]
        assert result.concentrations == {"Fe": 1.0}
        assert result.bic_final == -100.0
        assert result.bic_initial == -90.0
        assert "Fe" in result.boltzmann_results

    def test_concentrations_sum_to_one(self):
        """Returned concentrations should sum to ~1.0."""
        problem = _make_synthetic_problem(
            true_elements={
                "Fe": ([(350.0, 1.0), (370.0, 0.5)], 5.0),
                "Cu": ([(420.0, 0.8), (450.0, 0.6)], 3.0),
            },
            noise_sigma=0.001,
        )

        result = bic_prune_elements(
            observed=problem["observed"],
            basis_matrix=problem["basis_matrix"],
            element_list=problem["element_list"],
            element_coefficients=problem["coefficients"],
            noise_variance=problem["noise_variance"],
        )

        total = sum(result.concentrations.values())
        assert abs(total - 1.0) < 1e-10

    def test_default_boltzmann_results_empty(self):
        """Default boltzmann_results should be an empty dict."""
        result = ModelSelectionResult(
            selected_elements=[],
            removed_elements=[],
            concentrations={},
            bic_final=0.0,
            bic_initial=0.0,
        )
        assert result.boltzmann_results == {}


# ---------------------------------------------------------------------------
# Tests: boltzmann_consistency_filter
# ---------------------------------------------------------------------------


class _MockTransition:
    """Minimal mock transition for testing."""

    def __init__(self, wavelength_nm, A_ki, g_k, E_k_ev):
        self.wavelength_nm = wavelength_nm
        self.A_ki = A_ki
        self.g_k = g_k
        self.E_k_ev = E_k_ev


class TestBoltzmannConsistencyFilter:
    """Tests for the Boltzmann linearity check."""

    def test_linear_boltzmann_passes(self):
        """Synthetic data following Boltzmann distribution should pass."""
        KB_EV = 8.617333262e-5
        T_true = 10000.0  # K
        n_pixels = 1000
        wavelength = np.linspace(300.0, 600.0, n_pixels)

        # Create transitions at well-separated wavelengths with known E_k
        transitions = []
        E_k_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        line_wavelengths = [320.0, 370.0, 420.0, 470.0, 520.0]
        g_k_values = [9, 7, 11, 5, 9]
        A_ki_values = [1e7, 5e6, 2e7, 1e7, 8e6]

        for wl, A, g, E in zip(line_wavelengths, A_ki_values, g_k_values, E_k_values):
            transitions.append(_MockTransition(wl, A, g, E))

        # Construct observed spectrum following Boltzmann distribution
        # I = C * g_k * A_ki / lambda * exp(-E_k / kT)
        observed = np.zeros(n_pixels)
        basis_spectrum = np.zeros(n_pixels)
        sigma = 1.0 * FWHM_TO_SIGMA  # 1 nm FWHM, well-resolved on grid

        for trans in transitions:
            boltz = np.exp(-trans.E_k_ev / (KB_EV * T_true))
            intensity = trans.g_k * trans.A_ki * boltz / trans.wavelength_nm
            peak = intensity * np.exp(-0.5 * ((wavelength - trans.wavelength_nm) / sigma) ** 2)
            observed += peak
            basis_spectrum += np.exp(-0.5 * ((wavelength - trans.wavelength_nm) / sigma) ** 2)

        result = boltzmann_consistency_filter(
            element="Fe",
            wavelength=wavelength,
            observed=observed,
            basis_spectrum=basis_spectrum,
            transitions=transitions,
            T_estimated_K=T_true,
        )

        assert result["passes"]
        assert result["n_lines"] == 5
        assert result["R_squared"] > 0.9
        # Recovered temperature should be close to true
        assert abs(result["T_K"] - T_true) / T_true < 0.1

    def test_random_data_fails(self):
        """Random (non-Boltzmann) data should fail the consistency check."""
        n_pixels = 1000
        wavelength = np.linspace(300.0, 600.0, n_pixels)
        rng = np.random.default_rng(123)

        transitions = []
        line_wavelengths = [320.0, 370.0, 420.0, 470.0, 520.0]
        E_k_values = [1.0, 2.0, 3.0, 4.0, 5.0]

        for wl, E in zip(line_wavelengths, E_k_values):
            transitions.append(_MockTransition(wl, 1e7, 9, E))

        # Random observed spectrum -- no physical relationship
        observed = rng.uniform(0.1, 10.0, n_pixels)

        # Basis spectrum with peaks at the transition wavelengths
        basis_spectrum = np.zeros(n_pixels)
        sigma = 1.0 * FWHM_TO_SIGMA
        for trans in transitions:
            basis_spectrum += np.exp(-0.5 * ((wavelength - trans.wavelength_nm) / sigma) ** 2)

        result = boltzmann_consistency_filter(
            element="Fe",
            wavelength=wavelength,
            observed=observed,
            basis_spectrum=basis_spectrum,
            transitions=transitions,
            T_estimated_K=10000.0,
        )

        # R^2 should be low for random data
        assert result["R_squared"] < 0.5 or result["passes"] is False

    def test_empty_transitions(self):
        """No transitions -> passes=False."""
        wavelength = np.linspace(300.0, 600.0, 500)
        result = boltzmann_consistency_filter(
            element="X",
            wavelength=wavelength,
            observed=np.ones(500),
            basis_spectrum=np.ones(500),
            transitions=[],
            T_estimated_K=10000.0,
        )
        assert result["passes"] is False
        assert result["n_lines"] == 0

    def test_too_few_lines(self):
        """Fewer than 3 usable lines -> passes=False."""
        n_pixels = 500
        wavelength = np.linspace(300.0, 600.0, n_pixels)
        transitions = [
            _MockTransition(350.0, 1e7, 9, 2.0),
            _MockTransition(450.0, 1e7, 9, 3.0),
        ]
        basis_spectrum = np.zeros(n_pixels)
        sigma = 1.0 * FWHM_TO_SIGMA
        for t in transitions:
            basis_spectrum += np.exp(-0.5 * ((wavelength - t.wavelength_nm) / sigma) ** 2)
        observed = basis_spectrum * 100.0

        result = boltzmann_consistency_filter(
            element="Fe",
            wavelength=wavelength,
            observed=observed,
            basis_spectrum=basis_spectrum,
            transitions=transitions,
            T_estimated_K=10000.0,
        )
        assert result["passes"] is False
        assert result["n_lines"] == 2

    def test_zero_basis_spectrum(self):
        """All-zero basis spectrum -> passes=False."""
        wavelength = np.linspace(300.0, 600.0, 500)
        transitions = [_MockTransition(350.0, 1e7, 9, 2.0)]
        result = boltzmann_consistency_filter(
            element="Fe",
            wavelength=wavelength,
            observed=np.ones(500),
            basis_spectrum=np.zeros(500),
            transitions=transitions,
            T_estimated_K=10000.0,
        )
        assert result["passes"] is False


# ---------------------------------------------------------------------------
# Tests: _compute_bic
# ---------------------------------------------------------------------------


class TestComputeBic:
    def test_bic_increases_with_more_params(self):
        """BIC should increase when adding parameters for same RSS."""
        observed = np.ones(100)
        predicted = np.ones(100) * 1.01
        bic_few = _compute_bic(observed, predicted, k=2)
        bic_many = _compute_bic(observed, predicted, k=10)
        assert bic_many > bic_few

    def test_bic_decreases_with_better_fit(self):
        """BIC should decrease with lower RSS for same k."""
        observed = np.ones(100)
        bad_predicted = np.ones(100) * 1.5
        good_predicted = np.ones(100) * 1.01
        bic_bad = _compute_bic(observed, bad_predicted, k=3)
        bic_good = _compute_bic(observed, good_predicted, k=3)
        assert bic_good < bic_bad
