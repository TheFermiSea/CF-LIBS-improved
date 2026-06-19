"""Tests for additive information-criterion model selection (AIC / AICc / SpIC).

These criteria extend the existing BIC backward-elimination prune in
:mod:`cflibs.inversion.identify.model_selection` with the *same* interface,
selectable via :class:`Criterion`.  The math follows:

* AICc = AIC + 2k(k+1)/(n-k-1)        -- Hurvich & Tsai, Biometrika 76 (1989).
* SpIC line-strength-weighted hybrid  -- Webb et al., MNRAS 501 (2021),
  arXiv:2009.08336 ("Getting the model right: an information criterion for
  spectroscopy").

The suite verifies (a) AICc penalizes more than AIC at small n and converges
to it at large n, (b) every criterion recovers the known true component count
on a clean synthetic NNLS problem with one spurious element, and (c) interface
parity with :func:`bic_prune_elements`.
"""

import numpy as np
import pytest

from cflibs.inversion.identify.model_selection import (
    Criterion,
    ModelSelectionResult,
    _compute_aic,
    _compute_aicc,
    _compute_bic,
    _compute_spic,
    bic_prune_elements,
    prune_elements,
)

FWHM_TO_SIGMA = 1.0 / 2.3548200450309493


# ---------------------------------------------------------------------------
# Helpers (mirror tests/test_model_selection.py so the suites stay aligned)
# ---------------------------------------------------------------------------


def _make_element_basis(wavelength, line_centers, fwhm=0.5):
    """Create a synthetic element basis spectrum from Gaussian peaks (area=1)."""
    sigma = fwhm * FWHM_TO_SIGMA
    spectrum = np.zeros_like(wavelength)
    for center, amplitude in line_centers:
        spectrum += amplitude * np.exp(-0.5 * ((wavelength - center) / sigma) ** 2)
    area = np.sum(spectrum)
    if area > 0:
        spectrum /= area
    return spectrum


def _make_problem(seed=7, noise_sigma=1e-4):
    """Build a clean 3-element problem where exactly one element is spurious.

    True spectrum contains Fe + Cu.  Mg is offered as a basis row but is NOT
    present in the observed spectrum (zero true coefficient), so a correct
    selection recovers exactly two components (Fe, Cu) and drops Mg.
    """
    rng = np.random.default_rng(seed)
    wavelength = np.linspace(300.0, 500.0, 600)

    fe = _make_element_basis(wavelength, [(350.0, 1.0), (370.0, 0.6), (400.0, 0.8)])
    cu = _make_element_basis(wavelength, [(420.0, 0.9), (450.0, 0.7)])
    mg = _make_element_basis(wavelength, [(330.0, 1.0), (480.0, 0.5)])

    continuum = np.ones_like(wavelength)

    true_coeffs = {"Fe": 5.0, "Cu": 3.0}
    observed = true_coeffs["Fe"] * fe + true_coeffs["Cu"] * cu
    observed = observed + noise_sigma * rng.standard_normal(observed.shape)

    element_list = ["Fe", "Cu", "Mg"]
    # basis_matrix: element rows first, then a continuum row.
    basis_matrix = np.vstack([fe, cu, mg, continuum])

    # Initial NNLS coefficients (all three elements + continuum). Mg picks up a
    # tiny spurious amplitude from noise; backward elimination must drop it.
    from scipy.optimize import nnls

    coeffs, _ = nnls(basis_matrix.T, observed)
    noise_variance = noise_sigma**2

    return {
        "wavelength": wavelength,
        "observed": observed,
        "basis_matrix": basis_matrix,
        "element_list": element_list,
        "coefficients": coeffs,
        "noise_variance": noise_variance,
    }


# ---------------------------------------------------------------------------
# Pure-criterion math: AICc penalizes more than AIC at small n
# ---------------------------------------------------------------------------


class TestAICcVsAIC:
    def test_aicc_penalizes_more_than_aic_at_small_n(self):
        """At small n the AICc correction term is strictly positive."""
        rng = np.random.default_rng(0)
        n = 12
        k = 4
        observed = rng.standard_normal(n)
        predicted = observed + 0.1 * rng.standard_normal(n)

        aic = _compute_aic(observed, predicted, k)
        aicc = _compute_aicc(observed, predicted, k)

        # AICc = AIC + 2k(k+1)/(n-k-1), correction > 0 when n > k+1.
        correction = 2.0 * k * (k + 1) / (n - k - 1)
        assert correction > 0.0
        assert aicc > aic
        assert aicc == pytest.approx(aic + correction)

    def test_aicc_converges_to_aic_at_large_n(self):
        """The small-sample correction vanishes as n -> infinity."""
        rng = np.random.default_rng(1)
        k = 4
        gaps = []
        for n in (50, 500, 5000):
            observed = rng.standard_normal(n)
            predicted = observed + 0.1 * rng.standard_normal(n)
            gaps.append(
                _compute_aicc(observed, predicted, k) - _compute_aic(observed, predicted, k)
            )
        # Monotonically shrinking gap, approaching zero.
        assert gaps[0] > gaps[1] > gaps[2] > 0.0
        assert gaps[-1] < 0.05

    def test_aicc_rejects_when_n_le_k_plus_1(self):
        """AICc is undefined for n <= k + 1; return +inf so the model is rejected."""
        observed = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.0, 2.0, 3.0])
        assert _compute_aicc(observed, predicted, k=2) == float("inf")
        assert _compute_aicc(observed, predicted, k=3) == float("inf")

    def test_bic_unchanged_by_refactor(self):
        """_compute_bic still equals n*ln(RSS/n) + k*ln(n) after refactor."""
        rng = np.random.default_rng(2)
        n = 200
        observed = rng.standard_normal(n)
        predicted = observed + 0.05 * rng.standard_normal(n)
        rss = float(np.sum((observed - predicted) ** 2))
        expected = n * np.log(rss / n) + 5 * np.log(n)
        assert _compute_bic(observed, predicted, k=5) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# SpIC pure-criterion math
# ---------------------------------------------------------------------------


class TestSpIC:
    def test_spic_f_extremes_match_bracket_terms(self):
        """f=1 keeps only the AICc-like bracket; f=0 keeps only the BIC-like term."""
        rng = np.random.default_rng(3)
        n = 100
        observed = rng.standard_normal(n)
        predicted = observed + 0.05 * rng.standard_normal(n)
        strengths = np.array([20.0, 8.0, 50.0])
        k_a = 1.0
        fit = n * np.log(float(np.sum((observed - predicted) ** 2)) / n)

        r_a = np.maximum(strengths, k_a + 2.0)
        aicc_like = float(np.sum(2.0 * 1.0 * k_a * r_a / (r_a - k_a - 1.0)))
        bic_like = float(np.sum(0.0 * k_a * np.log(r_a)))  # zero by construction
        spic_aicc = _compute_spic(observed, predicted, strengths, f=1.0)
        assert spic_aicc == pytest.approx(fit + aicc_like + bic_like)

        bic_like_only = float(np.sum(1.0 * k_a * np.log(r_a)))
        spic_bic = _compute_spic(observed, predicted, strengths, f=0.0)
        assert spic_bic == pytest.approx(fit + bic_like_only)

    def test_spic_lower_bound_prevents_blowup(self):
        """R_a is floored at k_a + 2 so the first bracket term cannot diverge."""
        observed = np.zeros(50)
        predicted = np.zeros(50)
        # Strengths below the k_a + 1 singularity (k_a = 1 -> singular at R_a=2).
        spic = _compute_spic(observed, predicted, np.array([0.0, 1.5, 2.0]), f=1.0)
        assert np.isfinite(spic)

    def test_spic_strong_line_penalty_approaches_aic_term(self):
        """For strong lines (R_a -> N) the SpIC_A (f=1) penalty -> 2*k_a per comp.

        Webb et al. replace the global N by each component's own line strength
        R_a.  As R_a grows, R_a/(R_a - k_a - 1) -> 1 so the per-component AICc-
        analogue term -> 2*k_a, i.e. the pure-AIC cost, *below* the global
        AICc's inflated per-parameter cost 2N/(N-k-1).  This is the mechanism
        by which SpIC "requires fewer absorption components to achieve a
        similar goodness of fit" than AICc (Webb et al. 2021).
        """
        rng = np.random.default_rng(4)
        n = 4000  # large N -> global AICc per-parameter cost > 2
        observed = rng.standard_normal(n)
        predicted = observed + 0.05 * rng.standard_normal(n)  # finite, identical fit
        k = 3
        # Strong lines: R_a comparable to N.
        strengths = np.array([0.8 * n, 0.9 * n, n], dtype=float)

        fit = n * np.log(float(np.sum((observed - predicted) ** 2)) / n)
        aicc_penalty = _compute_aicc(observed, predicted, k) - fit
        spic_a_penalty = _compute_spic(observed, predicted, strengths, f=1.0) - fit

        # SpIC_A penalty for strong lines sits just above 2*k (= 6) and below
        # the global AICc penalty 2k + 2k(k+1)/(n-k-1).
        assert spic_a_penalty < aicc_penalty
        assert spic_a_penalty > 2.0 * k  # never below the pure-AIC floor

    def test_spic_line_strength_sensitivity(self):
        """SpIC penalty depends on line strength -- the property AICc/BIC lack.

        AICc/BIC "treat all model parameters as being of equal importance"
        (Webb et al. 2021); SpIC does not.  Webb's Fig. 2 shows the AICc
        analogue (f=1) penalizes *weak* lines (small R_a) more than strong
        ones, while the BIC analogue (f=0) does the opposite (the penalty
        grows like ln(R_a)).  Both directions are verified here, confirming
        the penalty genuinely tracks line strength rather than a global N.
        """
        observed = np.zeros(100)
        predicted = np.zeros(100)  # identical (zero) fit term -> compare penalties

        weak_aicc = _compute_spic(observed, predicted, np.array([4.0]), f=1.0)
        strong_aicc = _compute_spic(observed, predicted, np.array([400.0]), f=1.0)
        # AICc analogue: weak lines penalized more (Webb Fig. 2, blue curve).
        assert weak_aicc > strong_aicc

        weak_bic = _compute_spic(observed, predicted, np.array([4.0]), f=0.0)
        strong_bic = _compute_spic(observed, predicted, np.array([400.0]), f=0.0)
        # BIC analogue: strong lines penalized more (Webb Fig. 2, red curve).
        assert strong_bic > weak_bic

    def test_spic_f_out_of_range_raises(self):
        observed = np.zeros(10)
        predicted = np.zeros(10)
        with pytest.raises(ValueError):
            _compute_spic(observed, predicted, np.array([5.0]), f=1.5)


# ---------------------------------------------------------------------------
# Backward-elimination: recover the true component count
# ---------------------------------------------------------------------------


class TestRecoverTrueComponentCount:
    @pytest.mark.parametrize(
        "criterion",
        [Criterion.BIC, Criterion.AIC, Criterion.AICC, Criterion.SPIC],
    )
    def test_drops_spurious_element(self, criterion):
        """On a clean Fe+Cu spectrum every criterion drops the spurious Mg."""
        problem = _make_problem()
        result = prune_elements(
            observed=problem["observed"],
            basis_matrix=problem["basis_matrix"],
            element_list=problem["element_list"],
            element_coefficients=problem["coefficients"],
            noise_variance=problem["noise_variance"],
            criterion=criterion,
        )
        assert isinstance(result, ModelSelectionResult)
        assert result.criterion is criterion
        assert set(result.selected_elements) == {"Fe", "Cu"}
        assert "Mg" in result.removed_elements
        assert len(result.selected_elements) == 2  # true component count

    @pytest.mark.parametrize(
        "criterion",
        [Criterion.BIC, Criterion.AIC, Criterion.AICC, Criterion.SPIC],
    )
    def test_score_does_not_increase(self, criterion):
        """Backward elimination never accepts a worsening step."""
        problem = _make_problem()
        result = prune_elements(
            observed=problem["observed"],
            basis_matrix=problem["basis_matrix"],
            element_list=problem["element_list"],
            element_coefficients=problem["coefficients"],
            noise_variance=problem["noise_variance"],
            criterion=criterion,
        )
        assert result.bic_final <= result.bic_initial + 1e-9

    def test_concentrations_normalised(self):
        """Selected concentrations sum to ~1 regardless of criterion (point est.)."""
        problem = _make_problem()
        for criterion in Criterion:
            result = prune_elements(
                observed=problem["observed"],
                basis_matrix=problem["basis_matrix"],
                element_list=problem["element_list"],
                element_coefficients=problem["coefficients"],
                noise_variance=problem["noise_variance"],
                criterion=criterion,
            )
            assert sum(result.concentrations.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Interface parity with the existing BIC entry point
# ---------------------------------------------------------------------------


class TestInterfaceParity:
    def test_bic_wrapper_matches_generic_bic(self):
        """bic_prune_elements == prune_elements(criterion=BIC) on every field."""
        problem = _make_problem()
        common = dict(
            observed=problem["observed"],
            basis_matrix=problem["basis_matrix"],
            element_list=problem["element_list"],
            element_coefficients=problem["coefficients"],
            noise_variance=problem["noise_variance"],
        )
        legacy = bic_prune_elements(**common)
        generic = prune_elements(**common, criterion=Criterion.BIC)

        assert legacy.selected_elements == generic.selected_elements
        assert legacy.removed_elements == generic.removed_elements
        assert legacy.concentrations == generic.concentrations
        assert legacy.bic_final == pytest.approx(generic.bic_final)
        assert legacy.bic_initial == pytest.approx(generic.bic_initial)
        # Legacy wrapper still tags its result as the BIC criterion.
        assert legacy.criterion is Criterion.BIC

    def test_legacy_signature_unchanged(self):
        """bic_prune_elements accepts exactly its historical keyword set."""
        problem = _make_problem()
        result = bic_prune_elements(
            observed=problem["observed"],
            basis_matrix=problem["basis_matrix"],
            element_list=problem["element_list"],
            element_coefficients=problem["coefficients"],
            noise_variance=problem["noise_variance"],
            use_jax_nnls=False,
            jax_nnls_max_iter=300,
            jax_batch_trials=False,
        )
        assert isinstance(result, ModelSelectionResult)

    def test_all_criteria_share_signature(self):
        """Every criterion is callable through the identical prune_elements API."""
        problem = _make_problem()
        for criterion in Criterion:
            result = prune_elements(
                observed=problem["observed"],
                basis_matrix=problem["basis_matrix"],
                element_list=problem["element_list"],
                element_coefficients=problem["coefficients"],
                noise_variance=problem["noise_variance"],
                criterion=criterion,
            )
            assert isinstance(result, ModelSelectionResult)
            assert result.criterion is criterion
