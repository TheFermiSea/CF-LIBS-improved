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


class TestNominalFeedstockPrior:
    """DED weak feedstock prior: Dirichlet centered on nominal, never pinning."""

    def test_nominal_mole_fracs_from_wt(self):
        import numpy as np

        from cflibs.inversion.solve.bayesian import PriorConfig

        x = PriorConfig.nominal_mole_fracs_from_wt(
            {"Ti": 90.0, "Al": 6.0, "V": 4.0}, ("Ti", "Al", "V")
        )
        assert float(np.sum(x)) == pytest.approx(1.0)
        # Ti-6Al-4V wt% -> mole fractions (Ti heavy -> smaller mole fraction)
        np.testing.assert_allclose(x, [0.862, 0.102, 0.036], atol=2e-3)

    def test_from_wt_respects_element_order(self):
        import numpy as np

        from cflibs.inversion.solve.bayesian import PriorConfig

        a = PriorConfig.nominal_mole_fracs_from_wt(
            {"Ti": 90.0, "Al": 6.0, "V": 4.0}, ("Ti", "Al", "V")
        )
        b = PriorConfig.nominal_mole_fracs_from_wt(
            {"Ti": 90.0, "Al": 6.0, "V": 4.0}, ("V", "Al", "Ti")
        )
        np.testing.assert_allclose(a, b[::-1], atol=1e-9)

    def test_alpha_symmetric_by_default(self):
        import numpy as np

        from cflibs.inversion.solve.bayesian import PriorConfig
        from cflibs.inversion.solve.bayesian.models import _concentration_dirichlet_alpha

        alpha = np.asarray(_concentration_dirichlet_alpha(PriorConfig(concentration_alpha=2.0), 3))
        np.testing.assert_allclose(alpha, [2.0, 2.0, 2.0])

    def test_alpha_centered_on_nominal_with_weak_concentration(self):
        import numpy as np

        from cflibs.inversion.solve.bayesian import PriorConfig
        from cflibs.inversion.solve.bayesian.models import _concentration_dirichlet_alpha

        x = PriorConfig.nominal_mole_fracs_from_wt(
            {"Ti": 90.0, "Al": 6.0, "V": 4.0}, ("Ti", "Al", "V")
        )
        pc = PriorConfig(concentration_alpha=60.0, nominal_mole_fracs=x)
        alpha = np.asarray(_concentration_dirichlet_alpha(pc, 3))
        # Dirichlet mean == nominal exactly; total concentration == 60 (weak)
        assert float(alpha.sum()) == pytest.approx(60.0)
        np.testing.assert_allclose(alpha / alpha.sum(), np.asarray(x), atol=1e-6)

    def test_alpha_length_mismatch_raises(self):
        from cflibs.inversion.solve.bayesian import PriorConfig
        from cflibs.inversion.solve.bayesian.models import _concentration_dirichlet_alpha

        pc = PriorConfig(concentration_alpha=60.0, nominal_mole_fracs=[0.5, 0.5])
        with pytest.raises(ValueError, match="nominal_mole_fracs length"):
            _concentration_dirichlet_alpha(pc, 3)

    def test_zero_nominal_element_not_pinned_to_zero(self):
        """A contaminant with zero nominal must not get alpha=0 (degenerate
        Dirichlet that forbids detection); it is floored, present elements are
        unaffected."""
        import numpy as np

        from cflibs.inversion.solve.bayesian import PriorConfig
        from cflibs.inversion.solve.bayesian.models import (
            _ALPHA_FLOOR,
            _concentration_dirichlet_alpha,
        )

        x = np.array([0.86, 0.10, 0.04, 0.0])  # 4th element absent from feedstock
        pc = PriorConfig(concentration_alpha=60.0, nominal_mole_fracs=x)
        alpha = np.asarray(_concentration_dirichlet_alpha(pc, 4))
        assert alpha[3] >= _ALPHA_FLOOR and alpha[3] > 0.0  # detectable, not pinned
        assert alpha[0] == pytest.approx(60.0 * 0.86, rel=0.02)  # present el unaffected
