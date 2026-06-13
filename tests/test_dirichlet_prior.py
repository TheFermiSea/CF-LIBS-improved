"""
Tests for Dirichlet prior distribution in Bayesian CF-LIBS.

Validates:
1. Domain-specific alpha presets (geological, metallurgical, uninformative)
2. create_concentration_prior() helper (requires NumPyro)
"""

import pytest

pytestmark = pytest.mark.unit


class TestPriorConfig:
    """Tests for PriorConfig Dirichlet presets."""

    def test_default_alpha_is_uninformative(self):
        from cflibs.inversion.solve.bayesian import PriorConfig

        config = PriorConfig()
        assert config.concentration_alpha == 1.0

    def test_geological_preset(self):
        from cflibs.inversion.solve.bayesian import PriorConfig

        config = PriorConfig.geological()
        # Sparse compositions: alpha < 1
        assert config.concentration_alpha < 1.0
        assert config.T_eV_range == (0.5, 2.0)

    def test_metallurgical_preset(self):
        from cflibs.inversion.solve.bayesian import PriorConfig

        config = PriorConfig.metallurgical()
        # Peaked concentrations: alpha > 1
        assert config.concentration_alpha > 1.0
        # Narrower temperature range for alloys
        assert config.T_eV_range[1] <= 2.0

    def test_uninformative_preset(self):
        from cflibs.inversion.solve.bayesian import PriorConfig

        config = PriorConfig.uninformative()
        assert config.concentration_alpha == 1.0

    def test_presets_accept_overrides(self):
        from cflibs.inversion.solve.bayesian import PriorConfig

        config = PriorConfig.geological(concentration_alpha=0.3)
        assert config.concentration_alpha == 0.3

    def test_presets_return_priorconfig(self):
        from cflibs.inversion.solve.bayesian import PriorConfig

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
        from cflibs.inversion.solve.bayesian import create_concentration_prior

        prior = create_concentration_prior(n_elements=3, alpha=1.0)
        assert hasattr(prior, "sample")

    @pytest.mark.requires_bayesian
    def test_informative_prior_with_known_concentrations(self):
        pytest.importorskip("numpyro")
        from cflibs.inversion.solve.bayesian import create_concentration_prior

        prior = create_concentration_prior(
            n_elements=3, alpha=1.0, known_concentrations={0: 0.7, 1: 0.2}
        )
        assert hasattr(prior, "sample")

    @pytest.mark.requires_bayesian
    def test_sparse_alpha_produces_valid_prior(self):
        pytest.importorskip("numpyro")
        from cflibs.inversion.solve.bayesian import create_concentration_prior

        prior = create_concentration_prior(n_elements=5, alpha=0.5)
        assert hasattr(prior, "sample")

