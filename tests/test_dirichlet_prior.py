"""
Tests for Dirichlet prior distribution in Bayesian CF-LIBS.

Validates:
1. Dirichlet prior parameterization in cflibs/inversion/bayesian.py
2. Domain-specific alpha presets (geological, metallurgical, uninformative)
3. Stick-breaking transform for nested sampling
4. create_concentration_prior() helper (requires NumPyro)
5. Sum-to-one constraint guaranteed by Dirichlet
"""

import numpy as np
import pytest
from scipy import stats

pytestmark = pytest.mark.unit


class TestPriorConfig:
    """Tests for PriorConfig Dirichlet presets."""

    def test_default_alpha_is_uninformative(self):
        from cflibs.inversion.bayesian import PriorConfig

        config = PriorConfig()
        assert config.concentration_alpha == 1.0

    def test_geological_preset(self):
        from cflibs.inversion.bayesian import PriorConfig

        config = PriorConfig.geological()
        # Sparse compositions: alpha < 1
        assert config.concentration_alpha < 1.0
        assert config.T_eV_range == (0.5, 2.0)

    def test_metallurgical_preset(self):
        from cflibs.inversion.bayesian import PriorConfig

        config = PriorConfig.metallurgical()
        # Peaked concentrations: alpha > 1
        assert config.concentration_alpha > 1.0
        # Narrower temperature range for alloys
        assert config.T_eV_range[1] <= 2.0

    def test_uninformative_preset(self):
        from cflibs.inversion.bayesian import PriorConfig

        config = PriorConfig.uninformative()
        assert config.concentration_alpha == 1.0

    def test_presets_accept_overrides(self):
        from cflibs.inversion.bayesian import PriorConfig

        config = PriorConfig.geological(concentration_alpha=0.3)
        assert config.concentration_alpha == 0.3

    def test_presets_return_priorconfig(self):
        from cflibs.inversion.bayesian import PriorConfig

        for factory in [
            PriorConfig.geological,
            PriorConfig.metallurgical,
            PriorConfig.uninformative,
        ]:
            config = factory()
            assert isinstance(config, PriorConfig)


class TestCreateConcentrationPrior:
    """Tests for the create_concentration_prior() helper (requires NumPyro)."""

    @pytest.mark.requires_bayesian
    def test_basic_dirichlet_creation(self):
        pytest.importorskip("numpyro")
        from cflibs.inversion.bayesian import create_concentration_prior

        prior = create_concentration_prior(n_elements=3, alpha=1.0)
        assert hasattr(prior, "sample")

    @pytest.mark.requires_bayesian
    def test_informative_prior_with_known_concentrations(self):
        pytest.importorskip("numpyro")
        from cflibs.inversion.bayesian import create_concentration_prior

        prior = create_concentration_prior(
            n_elements=3, alpha=1.0, known_concentrations={0: 0.7, 1: 0.2}
        )
        assert hasattr(prior, "sample")

    @pytest.mark.requires_bayesian
    def test_sparse_alpha_produces_valid_prior(self):
        pytest.importorskip("numpyro")
        from cflibs.inversion.bayesian import create_concentration_prior

        prior = create_concentration_prior(n_elements=5, alpha=0.5)
        assert hasattr(prior, "sample")


class TestStickBreakingTransform:
    """Tests for the stick-breaking Dirichlet transform used in nested sampling."""

    def test_stick_breaking_produces_valid_simplex(self):
        """Stick-breaking samples should sum to 1 and be non-negative."""
        n_elements = 5
        alpha = 1.0
        rng = np.random.default_rng(42)

        for _ in range(100):
            u = rng.uniform(0, 1, n_elements - 1)
            remaining = 1.0
            conc = np.zeros(n_elements)

            for i in range(n_elements - 1):
                beta_sample = stats.beta.ppf(u[i], alpha, alpha * (n_elements - 1 - i))
                conc[i] = remaining * beta_sample
                remaining -= conc[i]

            conc[-1] = remaining

            # All non-negative
            assert np.all(conc >= 0), f"Negative concentration: {conc}"
            # Sum to 1
            assert abs(np.sum(conc) - 1.0) < 1e-10, f"Sum != 1: {np.sum(conc)}"

    def test_stick_breaking_with_sparse_alpha(self):
        """Sparse alpha (<1) should produce valid concentrations."""
        n_elements = 4
        alpha = 0.3
        rng = np.random.default_rng(42)

        for _ in range(100):
            u = rng.uniform(0, 1, n_elements - 1)
            remaining = 1.0
            conc = np.zeros(n_elements)

            for i in range(n_elements - 1):
                beta_sample = stats.beta.ppf(u[i], alpha, alpha * (n_elements - 1 - i))
                conc[i] = remaining * beta_sample
                remaining -= conc[i]

            conc[-1] = remaining
            assert np.all(conc >= 0)
            assert abs(np.sum(conc) - 1.0) < 1e-10

    def test_stick_breaking_uniform_alpha_is_uniform_on_simplex(self):
        """Dirichlet(1,...,1) should produce uniform samples on the simplex."""
        n_elements = 3
        alpha = 1.0
        n_samples = 10000
        rng = np.random.default_rng(42)

        samples = []
        for _ in range(n_samples):
            u = rng.uniform(0, 1, n_elements - 1)
            remaining = 1.0
            conc = np.zeros(n_elements)

            for i in range(n_elements - 1):
                beta_sample = stats.beta.ppf(u[i], alpha, alpha * (n_elements - 1 - i))
                conc[i] = remaining * beta_sample
                remaining -= conc[i]

            conc[-1] = remaining
            samples.append(conc)

        samples = np.array(samples)

        # All elements should have similar mean (1/n_elements)
        means = samples.mean(axis=0)
        expected_mean = 1.0 / n_elements
        for i, m in enumerate(means):
            assert (
                abs(m - expected_mean) < 0.02
            ), f"Element {i} mean {m:.4f} != expected {expected_mean:.4f}"

    def test_stick_breaking_peaked_alpha_concentrates_mass(self):
        """Dirichlet(alpha>1) should concentrate mass near center."""
        n_elements = 3
        alpha = 10.0
        n_samples = 5000
        rng = np.random.default_rng(42)

        samples = []
        for _ in range(n_samples):
            u = rng.uniform(0, 1, n_elements - 1)
            remaining = 1.0
            conc = np.zeros(n_elements)

            for i in range(n_elements - 1):
                beta_sample = stats.beta.ppf(u[i], alpha, alpha * (n_elements - 1 - i))
                conc[i] = remaining * beta_sample
                remaining -= conc[i]

            conc[-1] = remaining
            samples.append(conc)

        samples = np.array(samples)

        # Standard deviation should be small for peaked alpha
        stds = samples.std(axis=0)
        for i, s in enumerate(stds):
            assert s < 0.15, f"Element {i} std {s:.4f} too large for alpha={alpha}"


class TestDirichletProperties:
    """Test mathematical properties of the Dirichlet parameterization."""

    def test_dirichlet_sum_to_one(self):
        """Dirichlet samples must sum to exactly 1."""
        rng = np.random.default_rng(42)
        for alpha in [0.1, 0.5, 1.0, 2.0, 10.0]:
            for n in [2, 3, 5, 10]:
                samples = rng.dirichlet(np.ones(n) * alpha, size=100)
                sums = samples.sum(axis=1)
                np.testing.assert_allclose(sums, 1.0, atol=1e-10, err_msg=f"alpha={alpha}, n={n}")

    def test_dirichlet_all_positive(self):
        """Dirichlet samples must be non-negative."""
        rng = np.random.default_rng(42)
        for alpha in [0.1, 0.5, 1.0, 5.0]:
            samples = rng.dirichlet(np.ones(5) * alpha, size=1000)
            assert np.all(samples >= 0)

    def test_dirichlet_mean_equals_normalized_alpha(self):
        """E[X_i] = alpha_i / sum(alpha)."""
        rng = np.random.default_rng(42)
        alphas = np.array([2.0, 3.0, 5.0])
        expected_mean = alphas / alphas.sum()

        samples = rng.dirichlet(alphas, size=50000)
        sample_means = samples.mean(axis=0)

        np.testing.assert_allclose(sample_means, expected_mean, atol=0.01)

    def test_asymmetric_alpha_encodes_prior_knowledge(self):
        """Different alpha values should shift concentration means."""
        rng = np.random.default_rng(42)
        # Fe-dominated alloy: alpha_Fe >> alpha_Cu
        alphas = np.array([10.0, 1.0, 1.0])
        samples = rng.dirichlet(alphas, size=10000)

        # First element (Fe) should dominate
        assert samples[:, 0].mean() > 0.7
        # Others should be small
        assert samples[:, 1].mean() < 0.15
        assert samples[:, 2].mean() < 0.15
