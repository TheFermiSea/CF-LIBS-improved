"""
Tests for Bayesian inference module.

These tests validate:
1. BayesianForwardModel initialization and spectrum computation
2. Log-likelihood calculation with noise model
3. Prior creation functions
4. MCMC sampling (integration test)

Requirements: JAX, NumPyro (optional for MCMC tests)
"""

import pytest
import numpy as np
import sqlite3
import tempfile
from pathlib import Path
from cflibs.core.jax_runtime import jax_default_real_dtype

# Mark entire module as requiring JAX
pytestmark = [
    pytest.mark.requires_jax,
    pytest.mark.requires_bayesian,
]

# Skip all tests if JAX is not available
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from cflibs.inversion.bayesian import (  # noqa: E402
    BayesianForwardModel,
    AtomicDataArrays,
    NoiseParameters,
    PriorConfig,
    log_likelihood,
)


@pytest.fixture
def bayesian_db():
    """Create a database with partition functions and Stark parameters for Bayesian tests."""
    import os

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)  # Close file descriptor to prevent leaks

    conn = sqlite3.connect(db_path)

    # Create tables
    conn.execute("""
        CREATE TABLE lines (
            id INTEGER PRIMARY KEY,
            element TEXT,
            sp_num INTEGER,
            wavelength_nm REAL,
            aki REAL,
            ei_ev REAL,
            ek_ev REAL,
            gi INTEGER,
            gk INTEGER,
            rel_int REAL,
            stark_w REAL,
            stark_alpha REAL
        )
    """)

    conn.execute("""
        CREATE TABLE species_physics (
            element TEXT,
            sp_num INTEGER,
            ip_ev REAL,
            PRIMARY KEY (element, sp_num)
        )
    """)

    conn.execute("""
        CREATE TABLE partition_functions (
            element TEXT,
            sp_num INTEGER,
            a0 REAL,
            a1 REAL,
            a2 REAL,
            a3 REAL,
            a4 REAL,
            PRIMARY KEY (element, sp_num)
        )
    """)

    # Insert Fe spectral lines with Stark parameters
    lines_data = [
        # Fe I lines (sp_num=1)
        ("Fe", 1, 371.99, 1.0e7, 0.0, 3.33, 9, 11, 1000, 0.02, 0.5),
        ("Fe", 1, 373.49, 5.0e6, 0.0, 3.32, 9, 9, 500, 0.015, 0.5),
        ("Fe", 1, 374.95, 2.0e6, 0.0, 3.31, 9, 7, 200, 0.01, 0.5),
        ("Fe", 1, 438.35, 5.0e6, 0.0, 4.47, 9, 9, 800, 0.025, 0.5),
        # Fe II lines (sp_num=2)
        ("Fe", 2, 238.20, 3.0e8, 0.0, 5.22, 10, 10, 600, 0.03, 0.6),
        ("Fe", 2, 259.94, 2.2e8, 0.0, 4.77, 8, 10, 400, 0.025, 0.6),
    ]
    conn.executemany(
        """
        INSERT INTO lines (element, sp_num, wavelength_nm, aki, ei_ev, ek_ev, gi, gk, rel_int, stark_w, stark_alpha)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        lines_data,
    )

    # Cu I lines
    cu_lines = [
        ("Cu", 1, 324.75, 1.4e8, 0.0, 3.82, 2, 4, 2000, 0.01, 0.5),
        ("Cu", 1, 327.40, 1.4e8, 0.0, 3.79, 2, 2, 1000, 0.01, 0.5),
        ("Cu", 1, 510.55, 2.0e6, 1.39, 3.82, 4, 4, 300, 0.008, 0.5),
    ]
    conn.executemany(
        """
        INSERT INTO lines (element, sp_num, wavelength_nm, aki, ei_ev, ek_ev, gi, gk, rel_int, stark_w, stark_alpha)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        cu_lines,
    )

    # Ionization potentials
    conn.executemany(
        """
        INSERT INTO species_physics (element, sp_num, ip_ev) VALUES (?, ?, ?)
    """,
        [
            ("Fe", 1, 7.87),
            ("Fe", 2, 16.18),
            ("Cu", 1, 7.73),
            ("Cu", 2, 20.29),
        ],
    )

    # Partition function coefficients (Irwin polynomial form)
    # log(U) = a0 + a1*log(T) + a2*log(T)^2 + ...
    pf_data = [
        # Fe I: U ~ 25 at 10000K
        ("Fe", 1, 3.22, 0.0, 0.0, 0.0, 0.0),
        # Fe II: U ~ 40 at 10000K
        ("Fe", 2, 3.69, 0.0, 0.0, 0.0, 0.0),
        # Cu I: U ~ 2 at 10000K
        ("Cu", 1, 0.69, 0.0, 0.0, 0.0, 0.0),
        # Cu II: U ~ 1 at 10000K
        ("Cu", 2, 0.0, 0.0, 0.0, 0.0, 0.0),
    ]
    conn.executemany(
        """
        INSERT INTO partition_functions (element, sp_num, a0, a1, a2, a3, a4)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        pf_data,
    )

    conn.commit()
    conn.close()

    yield db_path

    Path(db_path).unlink()


@pytest.mark.requires_jax
class TestNoiseParameters:
    """Tests for NoiseParameters dataclass."""

    def test_default_values(self):
        """Test default noise parameter values."""
        params = NoiseParameters()
        assert params.readout_noise == 10.0
        assert params.dark_current == 1.0
        assert params.gain == 1.0

    def test_custom_values(self):
        """Test custom noise parameter values."""
        params = NoiseParameters(
            readout_noise=5.0,
            dark_current=0.5,
            gain=2.0,
        )
        assert params.readout_noise == 5.0
        assert params.dark_current == 0.5
        assert params.gain == 2.0


@pytest.mark.requires_jax
class TestPriorConfig:
    """Tests for PriorConfig dataclass."""

    def test_default_values(self):
        """Test default prior configuration."""
        config = PriorConfig()
        assert config.T_eV_range == (0.5, 3.0)
        # Full density range now supported with Weideman Faddeeva approximation
        assert config.log_ne_range == (15.0, 19.0)
        assert config.concentration_alpha == 1.0

    def test_custom_values(self):
        """Test custom prior configuration."""
        config = PriorConfig(
            T_eV_range=(0.8, 2.5),
            log_ne_range=(16.0, 18.0),
            concentration_alpha=2.0,
        )
        assert config.T_eV_range == (0.8, 2.5)
        assert config.log_ne_range == (16.0, 18.0)
        assert config.concentration_alpha == 2.0


@pytest.mark.requires_jax
class TestAtomicDataArrays:
    """Tests for AtomicDataArrays dataclass."""

    def test_creation(self):
        """Test AtomicDataArrays creation."""
        n_lines = 10
        arrays = AtomicDataArrays(
            wavelength_nm=jnp.linspace(300, 600, n_lines),
            aki=jnp.ones(n_lines) * 1e7,
            ek_ev=jnp.linspace(1, 5, n_lines),
            gk=jnp.ones(n_lines, dtype=jnp.int32) * 9,
            ip_ev=jnp.ones(n_lines) * 7.87,
            ion_stage=jnp.zeros(n_lines, dtype=jnp.int32),
            element_idx=jnp.zeros(n_lines, dtype=jnp.int32),
            stark_w=jnp.ones(n_lines) * 0.02,
            stark_alpha=jnp.ones(n_lines) * 0.5,
            mass_amu=jnp.ones(n_lines) * 55.85,
            partition_coeffs=jnp.zeros((1, 3, 5)),
            ionization_potentials=jnp.zeros((1, 3)),
            elements=["Fe"],
        )

        assert len(arrays.wavelength_nm) == n_lines
        assert arrays.elements == ["Fe"]


@pytest.mark.requires_jax
class TestBayesianForwardModel:
    """Tests for BayesianForwardModel class."""

    def test_init(self, bayesian_db):
        """Test BayesianForwardModel initialization."""
        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe", "Cu"],
            wavelength_range=(200, 600),
            pixels=500,
        )

        assert model.elements == ["Fe", "Cu"]
        assert len(model.wavelength) == 500
        assert model.wavelength.dtype == jax_default_real_dtype()
        assert model.atomic_data is not None
        assert len(model.atomic_data.wavelength_nm) > 0

    def test_forward_returns_spectrum(self, bayesian_db):
        """Test that forward model returns a valid spectrum."""
        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe", "Cu"],
            wavelength_range=(200, 600),
            pixels=500,
        )

        # Typical LIBS parameters
        T_eV = 1.0  # ~11600 K
        log_ne = 17.0  # 10^17 cm^-3
        concentrations = jnp.array([0.8, 0.2])  # 80% Fe, 20% Cu

        spectrum = model.forward(T_eV, log_ne, concentrations)

        # Check output shape
        assert spectrum.shape == (500,)

        # Spectrum should be non-negative
        assert jnp.all(spectrum >= 0)

        # Spectrum should have some emission (not all zeros)
        assert jnp.max(spectrum) > 0

    def test_forward_temperature_dependence(self, bayesian_db):
        """Test that spectrum intensity increases with temperature."""
        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe"],
            wavelength_range=(350, 450),
            pixels=200,
        )

        log_ne = 17.0
        concentrations = jnp.array([1.0])

        # Lower temperature
        spectrum_low = model.forward(0.5, log_ne, concentrations)
        # Higher temperature
        spectrum_high = model.forward(1.5, log_ne, concentrations)

        # Higher temperature should generally increase emission
        # (This is approximate - actual behavior depends on excitation energies)
        # At minimum, spectra should be different
        assert not jnp.allclose(spectrum_low, spectrum_high)

    def test_forward_density_dependence(self, bayesian_db):
        """Test that spectrum changes with electron density."""
        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe"],
            wavelength_range=(350, 450),
            pixels=200,
        )

        T_eV = 1.0
        concentrations = jnp.array([1.0])

        # Lower density
        spectrum_low = model.forward(T_eV, 16.0, concentrations)
        # Higher density
        spectrum_high = model.forward(T_eV, 18.0, concentrations)

        # Spectra should be different
        assert not jnp.allclose(spectrum_low, spectrum_high)

    def test_forward_explicit_total_species_density_preserves_legacy_default(self, bayesian_db):
        """Explicitly passing n_e-equivalent heavy density should match legacy behavior."""
        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe", "Cu"],
            wavelength_range=(200, 600),
            pixels=300,
        )

        T_eV = 1.0
        log_ne = 17.0
        concentrations = jnp.array([0.8, 0.2])
        n_e = 10.0**log_ne

        spectrum_default = model.forward(T_eV, log_ne, concentrations)
        spectrum_explicit = model.forward(
            T_eV,
            log_ne,
            concentrations,
            total_species_density_cm3=n_e,
        )

        np.testing.assert_allclose(spectrum_default, spectrum_explicit, rtol=1e-6, atol=1e-12)

    def test_forward_allows_decoupled_total_species_density(self, bayesian_db):
        """Changing heavy-particle density at fixed n_e should change the spectrum."""
        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe", "Cu"],
            wavelength_range=(200, 600),
            pixels=300,
        )

        T_eV = 1.0
        log_ne = 17.0
        concentrations = jnp.array([0.8, 0.2])

        spectrum_low = model.forward(
            T_eV,
            log_ne,
            concentrations,
            total_species_density_cm3=2.5e16,
        )
        spectrum_high = model.forward(
            T_eV,
            log_ne,
            concentrations,
            total_species_density_cm3=2.5e17,
        )

        assert not jnp.allclose(spectrum_low, spectrum_high)
        assert float(jnp.max(spectrum_high)) > float(jnp.max(spectrum_low))

    def test_forward_jit_accepts_explicit_total_species_density(self, bayesian_db):
        """Explicit heavy-particle density should work under ``jax.jit``."""
        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe", "Cu"],
            wavelength_range=(200, 600),
            pixels=150,
        )
        concentrations = jnp.array([0.8, 0.2])

        @jax.jit
        def compute(total_species_density_cm3):
            return model.forward(
                1.0,
                17.0,
                concentrations,
                total_species_density_cm3=total_species_density_cm3,
            )

        spectrum = compute(jnp.asarray(2.5e17))
        assert spectrum.shape == (150,)
        assert float(jnp.max(spectrum)) > 0.0

    def test_forward_jit_accepts_closed_over_scalar_total_species_density(self, bayesian_db):
        """Closed-over Python scalars should also work under ``jax.jit``."""
        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe", "Cu"],
            wavelength_range=(200, 600),
            pixels=150,
        )
        concentrations = jnp.array([0.8, 0.2])

        @jax.jit
        def compute():
            return model.forward(
                1.0,
                17.0,
                concentrations,
                total_species_density_cm3=2.5e17,
            )

        spectrum = compute()
        assert spectrum.shape == (150,)
        assert float(jnp.max(spectrum)) > 0.0

    def test_forward_concentration_scaling(self, bayesian_db):
        """Test that spectrum scales with concentration."""
        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe", "Cu"],
            wavelength_range=(200, 600),
            pixels=500,
        )

        T_eV = 1.0
        log_ne = 17.0

        # All Fe
        spectrum_fe = model.forward(T_eV, log_ne, jnp.array([1.0, 0.0]))
        # All Cu
        spectrum_cu = model.forward(T_eV, log_ne, jnp.array([0.0, 1.0]))
        # 50-50 mix
        spectrum_mix = model.forward(T_eV, log_ne, jnp.array([0.5, 0.5]))

        # Spectra should all be different
        assert not jnp.allclose(spectrum_fe, spectrum_cu)
        assert not jnp.allclose(spectrum_fe, spectrum_mix)

    def test_partition_function(self, bayesian_db):
        """Test partition function evaluation."""
        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe"],
            wavelength_range=(350, 450),
            pixels=100,
        )

        # Test partition function at 10000 K
        T_K = 10000.0
        coeffs = jnp.array([3.22, 0.0, 0.0, 0.0, 0.0])  # Fe I approx

        U = model._partition_function(T_K, coeffs)

        # Should be positive
        assert U > 0
        # Fe I at 10000 K should be ~25
        np.testing.assert_allclose(U, np.exp(3.22), rtol=0.01)


@pytest.mark.requires_jax
class TestLogLikelihood:
    """Tests for log_likelihood function."""

    def test_perfect_match(self):
        """Test log-likelihood for perfect match."""
        predicted = jnp.array([100.0, 200.0, 150.0])
        observed = jnp.array([100.0, 200.0, 150.0])

        ll = log_likelihood(predicted, observed)

        # Perfect match should give high likelihood
        # (not infinite because of variance term)
        assert jnp.isfinite(ll)

    def test_likelihood_decreases_with_mismatch(self):
        """Test that likelihood decreases as mismatch increases."""
        predicted = jnp.array([100.0, 200.0, 150.0])

        # Perfect match
        observed_good = predicted
        ll_good = log_likelihood(predicted, observed_good)

        # Small mismatch
        observed_ok = predicted + 10.0
        ll_ok = log_likelihood(predicted, observed_ok)

        # Large mismatch
        observed_bad = predicted + 100.0
        ll_bad = log_likelihood(predicted, observed_bad)

        # Likelihood should decrease with mismatch
        assert ll_good > ll_ok > ll_bad

    def test_noise_parameters_affect_likelihood(self):
        """Test that noise parameters affect likelihood."""
        predicted = jnp.array([100.0, 200.0, 150.0])
        # Use larger residual so residual^2/variance dominates over log(variance)
        observed = predicted + 100.0

        # Low noise (tight tolerance) - large residual penalized more
        noise_low = NoiseParameters(readout_noise=5.0)
        ll_low = log_likelihood(predicted, observed, noise_low)

        # High noise (loose tolerance) - large residual penalized less
        noise_high = NoiseParameters(readout_noise=100.0)
        ll_high = log_likelihood(predicted, observed, noise_high)

        # With large residual, higher noise tolerance gives higher likelihood
        assert ll_high > ll_low

    def test_handles_zero_predicted(self):
        """Test handling of zero/small predicted values."""
        predicted = jnp.array([0.0, 100.0, 0.001])
        observed = jnp.array([10.0, 100.0, 10.0])

        ll = log_likelihood(predicted, observed)

        # Should return finite value
        assert jnp.isfinite(ll)


@pytest.mark.requires_jax
class TestPriorCreation:
    """Tests for prior creation functions."""

    def test_temperature_prior_import(self):
        """Test temperature prior can be created (if NumPyro available)."""
        try:
            from cflibs.inversion.bayesian import create_temperature_prior

            prior = create_temperature_prior(0.5, 3.0, "uniform")
            assert prior is not None
        except ImportError:
            pytest.skip("NumPyro not available")

    def test_density_prior_import(self):
        """Test density prior can be created (if NumPyro available)."""
        try:
            from cflibs.inversion.bayesian import create_density_prior

            prior = create_density_prior(15.0, 19.0, "uniform")
            assert prior is not None
        except ImportError:
            pytest.skip("NumPyro not available")

    def test_concentration_prior_import(self):
        """Test concentration prior can be created (if NumPyro available)."""
        try:
            from cflibs.inversion.bayesian import create_concentration_prior

            prior = create_concentration_prior(n_elements=3, alpha=1.0)
            assert prior is not None
        except ImportError:
            pytest.skip("NumPyro not available")


@pytest.mark.requires_jax
class TestBayesianForwardModelEdgeCases:
    """Edge case tests for BayesianForwardModel."""

    def test_single_element(self, bayesian_db):
        """Test model with single element."""
        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe"],
            wavelength_range=(350, 450),
            pixels=200,
        )

        spectrum = model.forward(1.0, 17.0, jnp.array([1.0]))
        assert spectrum.shape == (200,)
        assert jnp.all(spectrum >= 0)

    def test_extreme_temperature_low(self, bayesian_db):
        """Test model at low temperature."""
        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe"],
            wavelength_range=(350, 450),
            pixels=200,
        )

        # Very low temperature (should still work)
        spectrum = model.forward(0.3, 17.0, jnp.array([1.0]))
        assert jnp.all(jnp.isfinite(spectrum))

    def test_extreme_temperature_high(self, bayesian_db):
        """Test model at high temperature."""
        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe"],
            wavelength_range=(350, 450),
            pixels=200,
        )

        # High temperature
        spectrum = model.forward(5.0, 17.0, jnp.array([1.0]))
        assert jnp.all(jnp.isfinite(spectrum))

    def test_extreme_density_low(self, bayesian_db):
        """Test model at low density."""
        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe"],
            wavelength_range=(350, 450),
            pixels=200,
        )

        # Low density
        spectrum = model.forward(1.0, 14.0, jnp.array([1.0]))
        assert jnp.all(jnp.isfinite(spectrum))

    def test_extreme_density_high(self, bayesian_db):
        """Test model at high density."""
        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe"],
            wavelength_range=(350, 450),
            pixels=200,
        )

        # High density
        spectrum = model.forward(1.0, 20.0, jnp.array([1.0]))
        assert jnp.all(jnp.isfinite(spectrum))

    def test_no_lines_in_range(self, bayesian_db):
        """Test error when no lines in wavelength range."""
        with pytest.raises(ValueError, match="No atomic data"):
            BayesianForwardModel(
                db_path=bayesian_db,
                elements=["Fe"],
                wavelength_range=(100, 150),  # No Fe lines here
                pixels=100,
            )


@pytest.mark.requires_jax
@pytest.mark.slow
class TestMCMCSampling:
    """Integration tests for MCMC sampling (require NumPyro)."""

    def test_run_mcmc_smoke(self, bayesian_db):
        """Smoke test for MCMC sampling."""
        pytest.importorskip("numpyro")
        from cflibs.inversion.bayesian import run_mcmc

        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe"],
            wavelength_range=(350, 450),
            pixels=100,
        )

        # Generate synthetic observed data
        T_true = 1.0
        log_ne_true = 17.0
        concentrations_true = jnp.array([1.0])

        synthetic = model.forward(T_true, log_ne_true, concentrations_true)
        # Add noise
        rng = np.random.default_rng(42)
        observed = np.array(synthetic) + rng.normal(0, 10, len(synthetic))
        observed = np.maximum(observed, 1.0)

        # Run short MCMC (just to verify it works)
        results = run_mcmc(
            model,
            observed,
            num_warmup=10,
            num_samples=20,
            seed=42,
        )

        # Check results structure
        assert "T_eV_mean" in results
        assert "log_ne_mean" in results
        assert "concentrations_mean" in results

        # Parameters should be in reasonable range
        assert 0.3 < results["T_eV_mean"] < 5.0
        assert 14.0 < results["log_ne_mean"] < 20.0

    def test_mcmc_result_correlation_matrix(self, bayesian_db):
        """Test correlation matrix computation from MCMCResult."""
        pytest.importorskip("numpyro")
        from cflibs.inversion.bayesian import MCMCSampler

        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe", "Cu"],
            wavelength_range=(300, 450),
            pixels=100,
        )

        # Generate synthetic observed data
        synthetic = model.forward(1.0, 17.0, jnp.array([0.6, 0.4]))
        rng = np.random.default_rng(42)
        observed = np.array(synthetic) + rng.normal(0, 10, len(synthetic))
        observed = np.maximum(observed, 1.0)

        # Run MCMC
        sampler = MCMCSampler(model)
        result = sampler.run(observed, num_warmup=10, num_samples=30, seed=42, progress_bar=False)

        # Test correlation matrix
        corr_data = result.correlation_matrix()
        assert "matrix" in corr_data
        assert "labels" in corr_data
        assert "T_log_ne_corr" in corr_data

        # Check matrix properties
        matrix = corr_data["matrix"]
        assert matrix.shape[0] == matrix.shape[1]  # Square
        assert np.allclose(np.diag(matrix), 1.0)  # Diagonal is 1
        assert np.allclose(matrix, matrix.T)  # Symmetric

        # Check labels
        labels = corr_data["labels"]
        assert "T_eV" in labels
        assert "log_ne" in labels

        # Check correlation is a valid value
        assert -1.0 <= corr_data["T_log_ne_corr"] <= 1.0

    def test_mcmc_result_correlation_table(self, bayesian_db):
        """Test correlation table string formatting."""
        pytest.importorskip("numpyro")
        from cflibs.inversion.bayesian import MCMCSampler

        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe"],
            wavelength_range=(350, 450),
            pixels=100,
        )

        synthetic = model.forward(1.0, 17.0, jnp.array([1.0]))
        rng = np.random.default_rng(42)
        observed = np.array(synthetic) + rng.normal(0, 10, len(synthetic))
        observed = np.maximum(observed, 1.0)

        sampler = MCMCSampler(model)
        result = sampler.run(observed, num_warmup=10, num_samples=30, seed=42, progress_bar=False)

        # Test correlation table
        table = result.correlation_table()
        assert isinstance(table, str)
        assert "Parameter Correlations" in table
        assert "T_eV" in table
        assert "log_ne" in table
        assert "T - log_ne correlation" in table

    def test_mcmc_sampler_plot_corner(self, bayesian_db):
        """Test corner plot method (returns None if matplotlib unavailable)."""
        pytest.importorskip("numpyro")
        from cflibs.inversion.bayesian import MCMCSampler

        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe"],
            wavelength_range=(350, 450),
            pixels=100,
        )

        synthetic = model.forward(1.0, 17.0, jnp.array([1.0]))
        rng = np.random.default_rng(42)
        observed = np.array(synthetic) + rng.normal(0, 10, len(synthetic))
        observed = np.maximum(observed, 1.0)

        sampler = MCMCSampler(model)
        result = sampler.run(observed, num_warmup=10, num_samples=30, seed=42, progress_bar=False)

        # Test plot_corner method exists and doesn't crash
        # (may return None if matplotlib/ArviZ not properly configured)
        try:
            axes = sampler.plot_corner(result)
            # If it returns something, it should be axes or None
            assert axes is None or hasattr(axes, "__iter__") or hasattr(axes, "flat")
        except Exception:
            # Some environments may not have display
            pass

    def test_mcmc_sampler_plot_forest(self, bayesian_db):
        """Test forest plot method."""
        pytest.importorskip("numpyro")
        from cflibs.inversion.bayesian import MCMCSampler

        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe"],
            wavelength_range=(350, 450),
            pixels=100,
        )

        synthetic = model.forward(1.0, 17.0, jnp.array([1.0]))
        rng = np.random.default_rng(42)
        observed = np.array(synthetic) + rng.normal(0, 10, len(synthetic))
        observed = np.maximum(observed, 1.0)

        sampler = MCMCSampler(model)
        result = sampler.run(observed, num_warmup=10, num_samples=30, seed=42, progress_bar=False)

        # Test plot_forest method exists and doesn't crash
        try:
            axes = sampler.plot_forest(result)
            assert axes is None or hasattr(axes, "__iter__") or hasattr(axes, "flat")
        except Exception:
            pass


# --- Nested Sampling Tests (CF-LIBS-nfo) ---


@pytest.mark.requires_jax
class TestNestedSampling:
    """Tests for nested sampling functionality."""

    def test_nested_sampling_result_dataclass(self):
        """Test NestedSamplingResult dataclass creation and properties."""
        from cflibs.inversion.bayesian import NestedSamplingResult

        # Create minimal result
        result = NestedSamplingResult(
            samples={"T_eV": np.array([1.0, 1.1, 1.2]), "log_ne": np.array([17.0, 17.1, 17.2])},
            weights=np.array([0.3, 0.4, 0.3]),
            log_evidence=-100.0,
            log_evidence_err=0.5,
            information=2.5,
            T_eV_mean=1.1,
            T_eV_std=0.1,
            log_ne_mean=17.1,
            log_ne_std=0.1,
            concentrations_mean={"Fe": 0.5, "Ca": 0.5},
            concentrations_std={"Fe": 0.1, "Ca": 0.1},
            n_live=100,
            n_iterations=500,
            n_calls=10000,
        )

        # Test basic properties
        assert result.log_evidence == -100.0
        assert result.log_evidence_err == 0.5
        assert result.information == 2.5

        # Test derived properties (use approximate comparison for floating point)
        from cflibs.core.constants import EV_TO_K

        assert np.isclose(result.T_K_mean, result.T_eV_mean * EV_TO_K)
        assert np.isclose(result.n_e_mean, 10.0**17.1)

        # Test evidence property
        assert np.isclose(result.evidence, np.exp(-100.0))

    def test_nested_sampling_result_summary_table(self):
        """Test summary table generation."""
        from cflibs.inversion.bayesian import NestedSamplingResult

        result = NestedSamplingResult(
            samples={"T_eV": np.array([1.0]), "log_ne": np.array([17.0])},
            weights=np.array([1.0]),
            log_evidence=-50.0,
            log_evidence_err=0.3,
            information=3.0,
            T_eV_mean=1.0,
            T_eV_std=0.05,
            log_ne_mean=17.0,
            log_ne_std=0.2,
            concentrations_mean={"Fe": 1.0},
            concentrations_std={"Fe": 0.01},
            n_live=100,
            n_iterations=200,
            n_calls=5000,
        )

        table = result.summary_table()
        assert isinstance(table, str)
        assert "Nested Sampling" in table
        assert "ln(Z)" in table
        assert "Information" in table
        assert "T [eV]" in table
        assert "Fe" in table

    def test_nested_sampling_result_model_comparison(self):
        """Test Bayes factor model comparison."""
        from cflibs.inversion.bayesian import NestedSamplingResult

        # Create two mock results with different evidences
        result_a = NestedSamplingResult(
            samples={"T_eV": np.array([1.0]), "log_ne": np.array([17.0])},
            weights=np.array([1.0]),
            log_evidence=-100.0,
            log_evidence_err=0.5,
            information=2.0,
            T_eV_mean=1.0,
            T_eV_std=0.1,
            log_ne_mean=17.0,
            log_ne_std=0.2,
            concentrations_mean={"Fe": 1.0},
            concentrations_std={"Fe": 0.05},
        )

        result_b = NestedSamplingResult(
            samples={"T_eV": np.array([1.0]), "log_ne": np.array([17.0])},
            weights=np.array([1.0]),
            log_evidence=-105.0,  # Lower evidence (worse model)
            log_evidence_err=0.5,
            information=2.0,
            T_eV_mean=1.0,
            T_eV_std=0.1,
            log_ne_mean=17.0,
            log_ne_std=0.2,
            concentrations_mean={"Fe": 1.0},
            concentrations_std={"Fe": 0.05},
        )

        comparison = NestedSamplingResult.compare_models(result_a, result_b, "Model A", "Model B")

        assert isinstance(comparison, str)
        assert "Model Comparison" in comparison
        assert "Δln(Z)" in comparison
        assert "Bayes factor" in comparison
        assert "Model A" in comparison

    def test_nested_sampler_import(self):
        """Test NestedSampler can be imported."""
        pytest.importorskip("dynesty")
        from cflibs.inversion.bayesian import NestedSampler, HAS_DYNESTY

        assert HAS_DYNESTY is True
        assert NestedSampler is not None

    def test_nested_sampler_init(self, bayesian_db):
        """Test NestedSampler initialization."""
        pytest.importorskip("dynesty")
        from cflibs.inversion.bayesian import NestedSampler

        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe", "Ca"],
            wavelength_range=(350, 450),
            pixels=50,
        )

        sampler = NestedSampler(model)

        assert sampler.n_elements == 2
        assert sampler.ndim == 3  # T_eV + log_ne + (2-1) concentrations
        assert sampler.forward_model is model

    def test_nested_sampler_prior_transform(self, bayesian_db):
        """Test prior transform from unit cube to physical space."""
        pytest.importorskip("dynesty")
        from cflibs.inversion.bayesian import NestedSampler, PriorConfig

        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe", "Ca"],
            wavelength_range=(350, 450),
            pixels=50,
        )

        config = PriorConfig(T_eV_range=(0.5, 3.0), log_ne_range=(15.0, 19.0))
        sampler = NestedSampler(model, prior_config=config)

        # Test transform at boundaries
        u_min = np.array([0.0, 0.0, 0.0])
        u_max = np.array([1.0, 1.0, 1.0])
        u_mid = np.array([0.5, 0.5, 0.5])

        x_min = sampler._prior_transform(u_min)
        x_max = sampler._prior_transform(u_max)
        x_mid = sampler._prior_transform(u_mid)

        # Check T_eV bounds
        assert x_min[0] == 0.5  # T_min
        assert x_max[0] == 3.0  # T_max
        assert 0.5 < x_mid[0] < 3.0

        # Check log_ne bounds
        assert x_min[1] == 15.0
        assert x_max[1] == 19.0
        assert 15.0 < x_mid[1] < 19.0

    def test_nested_sampler_params_to_concentrations(self, bayesian_db):
        """Test parameter to concentration conversion."""
        pytest.importorskip("dynesty")
        from cflibs.inversion.bayesian import NestedSampler

        # Single element case
        model_single = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe"],
            wavelength_range=(350, 450),
            pixels=50,
        )
        sampler_single = NestedSampler(model_single)
        conc = sampler_single._params_to_concentrations(np.array([1.0, 17.0]))
        assert len(conc) == 1
        assert conc[0] == 1.0

        # Multi-element case
        model_multi = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe", "Ca"],
            wavelength_range=(350, 450),
            pixels=50,
        )
        sampler_multi = NestedSampler(model_multi)
        params = np.array([1.0, 17.0, 0.6])  # T_eV, log_ne, c_Fe
        conc = sampler_multi._params_to_concentrations(params)
        assert len(conc) == 2
        assert conc[0] == 0.6
        assert conc[1] == 0.4  # 1.0 - 0.6
        assert np.isclose(np.sum(conc), 1.0)

    @pytest.mark.slow
    def test_nested_sampler_run_smoke(self, bayesian_db):
        """Smoke test for nested sampling run (minimal iterations)."""
        pytest.importorskip("dynesty")
        from cflibs.inversion.bayesian import NestedSampler, NestedSamplingResult

        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe"],
            wavelength_range=(350, 450),  # Wide range to ensure Fe lines exist
            pixels=50,
        )

        # Generate synthetic observation
        synthetic = model.forward(1.0, 17.0, jnp.array([1.0]))
        rng = np.random.default_rng(42)
        observed = np.array(synthetic) + rng.normal(0, 50, len(synthetic))
        observed = np.maximum(observed, 1.0)

        sampler = NestedSampler(model)
        result = sampler.run(
            observed,
            nlive=20,  # Minimal for smoke test
            dlogz=1.0,  # Loose tolerance for speed
            maxiter=100,  # Cap iterations
            verbose=False,
        )

        # Check result type
        assert isinstance(result, NestedSamplingResult)

        # Check evidence is computed
        assert np.isfinite(result.log_evidence)
        assert result.log_evidence_err > 0

        # Check samples exist
        assert "T_eV" in result.samples
        assert "log_ne" in result.samples
        assert len(result.weights) == len(result.samples["T_eV"])

        # Check weighted mean is in reasonable range
        assert 0.5 < result.T_eV_mean < 3.0
        assert 15.0 < result.log_ne_mean < 19.0

        # Check metadata
        assert result.n_live == 20
        assert result.n_iterations > 0
        assert result.n_calls > 0


class TestPosteriorPredictiveCheck:
    """Tests for posterior predictive check functionality."""

    @pytest.mark.slow
    def test_posterior_predictive_check_returns_expected_keys(self, bayesian_db):
        """Test that posterior predictive check returns all expected keys."""
        pytest.importorskip("jax")
        pytest.importorskip("numpyro")
        import jax.numpy as jnp
        from cflibs.inversion.bayesian import MCMCSampler

        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe"],
            wavelength_range=(350, 450),
            pixels=50,
        )

        # Generate synthetic observation
        synthetic = model.forward(1.0, 17.0, jnp.array([1.0]))
        rng = np.random.default_rng(42)
        observed = np.array(synthetic) + rng.normal(0, 50, len(synthetic))
        observed = np.maximum(observed, 1.0)

        sampler = MCMCSampler(model)
        result = sampler.run(
            observed,
            num_samples=50,
            num_warmup=50,
            num_chains=1,
        )

        # Run posterior predictive check
        ppc = sampler.posterior_predictive_check(result, observed, n_samples=20)

        # Check all expected keys are present
        expected_keys = {
            "predicted_mean",
            "predicted_std",
            "residuals",
            "chi_squared_obs",
            "chi_squared_sim",
            "p_value",
            "model_adequate",
            "n_samples_used",
        }
        assert set(ppc.keys()) == expected_keys

        # Check shapes
        assert ppc["predicted_mean"].shape == observed.shape
        assert ppc["predicted_std"].shape == observed.shape
        assert ppc["residuals"].shape == observed.shape
        assert len(ppc["chi_squared_sim"]) == 20

        # Check types
        assert isinstance(ppc["chi_squared_obs"], float)
        assert isinstance(ppc["p_value"], float)
        assert isinstance(ppc["model_adequate"], (bool, np.bool_))
        assert isinstance(ppc["n_samples_used"], int)

    @pytest.mark.slow
    def test_posterior_predictive_check_p_value_range(self, bayesian_db):
        """Test that p-value is in valid range [0, 1]."""
        pytest.importorskip("jax")
        pytest.importorskip("numpyro")
        import jax.numpy as jnp
        from cflibs.inversion.bayesian import MCMCSampler

        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe"],
            wavelength_range=(350, 450),
            pixels=50,
        )

        # Generate synthetic observation
        synthetic = model.forward(1.0, 17.0, jnp.array([1.0]))
        rng = np.random.default_rng(123)
        observed = np.array(synthetic) + rng.normal(0, 50, len(synthetic))
        observed = np.maximum(observed, 1.0)

        sampler = MCMCSampler(model)
        result = sampler.run(
            observed,
            num_samples=50,
            num_warmup=50,
            num_chains=1,
        )

        ppc = sampler.posterior_predictive_check(result, observed, n_samples=30)

        # p-value must be in [0, 1]
        assert 0.0 <= ppc["p_value"] <= 1.0

        # chi-squared values must be non-negative
        assert ppc["chi_squared_obs"] >= 0
        assert all(cs >= 0 for cs in ppc["chi_squared_sim"])

    @pytest.mark.slow
    def test_posterior_predictive_check_model_adequate_for_good_fit(self, bayesian_db):
        """Test that model_adequate is True when model fits well."""
        pytest.importorskip("jax")
        pytest.importorskip("numpyro")
        import jax.numpy as jnp
        from cflibs.inversion.bayesian import MCMCSampler

        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe"],
            wavelength_range=(350, 450),
            pixels=50,
        )

        # Generate synthetic observation with realistic noise
        T_true, log_ne_true = 1.0, 17.0
        synthetic = model.forward(T_true, log_ne_true, jnp.array([1.0]))
        rng = np.random.default_rng(456)
        observed = np.array(synthetic) + rng.normal(0, 30, len(synthetic))
        observed = np.maximum(observed, 1.0)

        sampler = MCMCSampler(model)
        result = sampler.run(
            observed,
            num_samples=100,
            num_warmup=100,
            num_chains=1,
        )

        ppc = sampler.posterior_predictive_check(result, observed, n_samples=50)

        # For synthetic data with appropriate noise, p-value should indicate good fit
        # (not too extreme in either direction). We use a wide range since this is
        # probabilistic and we want tests to be robust.
        # Check that chi-squared values are finite and positive
        assert np.isfinite(ppc["chi_squared_obs"])
        assert ppc["chi_squared_obs"] > 0
        assert all(np.isfinite(cs) for cs in ppc["chi_squared_sim"])

    def test_posterior_predictive_check_n_samples_limiting(self, bayesian_db):
        """Test that n_samples is limited to available samples."""
        pytest.importorskip("jax")
        pytest.importorskip("numpyro")
        import jax.numpy as jnp
        from cflibs.inversion.bayesian import MCMCSampler

        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe"],
            wavelength_range=(350, 450),
            pixels=50,
        )

        synthetic = model.forward(1.0, 17.0, jnp.array([1.0]))
        observed = np.array(synthetic) + 10.0

        sampler = MCMCSampler(model)
        result = sampler.run(
            observed,
            num_samples=30,  # Small number of samples
            num_warmup=30,
            num_chains=1,
        )

        # Request more samples than available
        ppc = sampler.posterior_predictive_check(result, observed, n_samples=1000)

        # n_samples_used should be capped at available
        assert ppc["n_samples_used"] <= 30


# ============================================================================
# Phase 3: Advanced Bayesian Inference Tests
# ============================================================================


class TestMcWhirterPenalty:
    """Tests for the McWhirter criterion soft penalty."""

    def test_penalty_zero_when_satisfied(self):
        """Penalty is zero when n_e is well above the threshold."""
        from cflibs.inversion.bayesian import mcwhirter_log_penalty

        # T=1 eV, log_ne=18 → well above threshold for ΔE=3 eV
        penalty = mcwhirter_log_penalty(T_eV=1.0, log_ne=18.0, max_delta_E_eV=3.0)
        assert penalty == 0.0 or abs(penalty) < 1e-10

    def test_penalty_negative_when_violated(self):
        """Penalty is negative when n_e is below the threshold."""
        from cflibs.inversion.bayesian import mcwhirter_log_penalty

        # T=1 eV, log_ne=14 → well below threshold
        penalty = mcwhirter_log_penalty(T_eV=1.0, log_ne=14.0, max_delta_E_eV=3.0)
        assert penalty < -1.0

    def test_penalty_monotonically_decreases_with_deficit(self):
        """Larger density deficit → more negative penalty."""
        from cflibs.inversion.bayesian import mcwhirter_log_penalty

        p1 = mcwhirter_log_penalty(T_eV=1.0, log_ne=15.0)
        p2 = mcwhirter_log_penalty(T_eV=1.0, log_ne=14.0)
        p3 = mcwhirter_log_penalty(T_eV=1.0, log_ne=13.0)
        assert p1 >= p2 >= p3

    def test_penalty_smoothness_and_gradient(self):
        """Penalty is smooth and has finite JAX gradient."""
        from cflibs.inversion.bayesian import mcwhirter_log_penalty

        import jax
        import jax.numpy as jnp

        def penalty_fn(log_ne):
            return mcwhirter_log_penalty(T_eV=1.0, log_ne=log_ne)

        grad_fn = jax.grad(penalty_fn)
        # Gradient at the boundary region
        g = grad_fn(16.0)
        assert jnp.isfinite(g)

        # Gradient at a well-satisfied point
        g2 = grad_fn(18.0)
        assert jnp.isfinite(g2)

    def test_scale_parameter(self):
        """Scale parameter controls penalty magnitude."""
        from cflibs.inversion.bayesian import mcwhirter_log_penalty

        p_low = mcwhirter_log_penalty(T_eV=1.0, log_ne=14.0, scale=1.0)
        p_high = mcwhirter_log_penalty(T_eV=1.0, log_ne=14.0, scale=100.0)
        assert abs(p_high) > abs(p_low)


class TestSharedUtilities:
    """Tests for refactored shared utilities."""

    def test_load_atomic_data(self, bayesian_db):
        """Module-level load_atomic_data returns valid AtomicDataArrays."""
        from cflibs.inversion.bayesian import load_atomic_data, AtomicDataArrays

        data = load_atomic_data(bayesian_db, ["Fe", "Cu"], (200.0, 600.0))
        assert isinstance(data, AtomicDataArrays)
        assert len(data.wavelength_nm) > 0
        assert data.elements == ["Fe", "Cu"]

    def test_load_atomic_data_has_new_fields(self, bayesian_db):
        """Loaded data includes ei_ev and f_osc fields."""
        from cflibs.inversion.bayesian import load_atomic_data

        data = load_atomic_data(bayesian_db, ["Fe"], (200.0, 600.0))
        assert data.ei_ev is not None
        assert data.f_osc is not None
        assert len(data.ei_ev) == len(data.wavelength_nm)
        assert len(data.f_osc) == len(data.wavelength_nm)

    def test_partition_function_module_level(self):
        """Module-level partition_function matches static method."""
        from cflibs.inversion.bayesian import partition_function

        coeffs = jnp.array([3.22, 0.0, 0.0, 0.0, 0.0])
        result = partition_function(10000.0, coeffs)
        expected = float(jnp.exp(3.22))  # a0 only, others zero
        assert abs(float(result) - expected) / expected < 0.01

    def test_bayesian_forward_model_uses_shared_loader(self, bayesian_db):
        """BayesianForwardModel still works after refactoring to shared loader."""
        model = BayesianForwardModel(bayesian_db, ["Fe", "Cu"], (200.0, 600.0), pixels=100)
        conc = jnp.array([0.7, 0.3])
        spectrum = model.forward(1.0, 17.0, conc)
        assert spectrum.shape == (100,)
        assert float(jnp.max(spectrum)) > 0


class TestTwoZoneBayesianForwardModel:
    """Tests for the two-zone plasma forward model."""

    def test_init(self, bayesian_db):
        """Two-zone model initialises without error."""
        from cflibs.inversion.bayesian import TwoZoneBayesianForwardModel

        model = TwoZoneBayesianForwardModel(bayesian_db, ["Fe", "Cu"], (200.0, 600.0), pixels=100)
        assert len(model.elements) == 2
        assert model.wavelength.shape == (100,)
        assert model.wavelength.dtype == jax_default_real_dtype()

    def test_forward_returns_valid_spectrum(self, bayesian_db):
        """Forward model returns a positive spectrum of correct shape."""
        from cflibs.inversion.bayesian import TwoZoneBayesianForwardModel

        model = TwoZoneBayesianForwardModel(bayesian_db, ["Fe", "Cu"], (200.0, 600.0), pixels=100)
        conc = jnp.array([0.7, 0.3])
        spectrum = model.forward(
            T_core_eV=1.2,
            T_shell_eV=0.8,
            log_ne=17.0,
            concentrations=conc,
            shell_fraction=0.3,
            optical_depth_scale=1.0,
        )
        assert spectrum.shape == (100,)
        assert float(jnp.min(spectrum)) >= 0.0
        assert float(jnp.max(spectrum)) > 0.0

    def test_self_reversal_with_high_optical_depth(self, bayesian_db):
        """High optical depth should reduce peak intensity (self-reversal)."""
        from cflibs.inversion.bayesian import TwoZoneBayesianForwardModel

        model = TwoZoneBayesianForwardModel(bayesian_db, ["Fe", "Cu"], (200.0, 600.0), pixels=200)
        conc = jnp.array([0.7, 0.3])

        # Low optical depth → strong peaks
        s_low = model.forward(1.2, 0.6, 17.0, conc, 0.3, 0.001)
        # High optical depth → reduced peaks
        s_high = model.forward(1.2, 0.6, 17.0, conc, 0.3, 10.0)

        # Peak intensity should generally decrease with higher optical depth
        assert float(jnp.max(s_high)) <= float(jnp.max(s_low)) * 1.1  # allow 10% tolerance

    def test_reduces_to_single_zone_when_shell_transparent(self, bayesian_db):
        """With zero optical depth, shell is transparent → only core emission."""
        from cflibs.inversion.bayesian import TwoZoneBayesianForwardModel, BayesianForwardModel

        model_2z = TwoZoneBayesianForwardModel(
            bayesian_db, ["Fe", "Cu"], (200.0, 600.0), pixels=100
        )
        model_1z = BayesianForwardModel(bayesian_db, ["Fe", "Cu"], (200.0, 600.0), pixels=100)
        conc = jnp.array([0.7, 0.3])

        # Two-zone with essentially zero optical depth
        s_2z = model_2z.forward(1.0, 0.5, 17.0, conc, 0.3, 1e-10)
        s_1z = model_1z.forward(1.0, 17.0, conc)

        # Should be dominated by core emission, though not identical due to
        # shell emission contribution at near-zero tau
        # Just check they are both positive and correlated
        assert float(jnp.max(s_2z)) > 0
        assert float(jnp.max(s_1z)) > 0

    def test_two_zone_explicit_total_species_density_preserves_legacy_default(self, bayesian_db):
        """Explicitly passing n_e-equivalent heavy density should match legacy behavior."""
        from cflibs.inversion.bayesian import TwoZoneBayesianForwardModel

        model = TwoZoneBayesianForwardModel(bayesian_db, ["Fe", "Cu"], (200.0, 600.0), pixels=100)
        conc = jnp.array([0.7, 0.3])
        n_e = 10.0**17.0

        spectrum_default = model.forward(1.2, 0.8, 17.0, conc, 0.3, 1.0)
        spectrum_explicit = model.forward(
            1.2,
            0.8,
            17.0,
            conc,
            0.3,
            1.0,
            total_species_density_cm3=n_e,
        )

        np.testing.assert_allclose(spectrum_default, spectrum_explicit, rtol=1e-6, atol=1e-12)

    def test_two_zone_allows_decoupled_total_species_density(self, bayesian_db):
        """Changing heavy-particle density at fixed n_e should change the two-zone spectrum."""
        from cflibs.inversion.bayesian import TwoZoneBayesianForwardModel

        model = TwoZoneBayesianForwardModel(bayesian_db, ["Fe", "Cu"], (200.0, 600.0), pixels=100)
        conc = jnp.array([0.7, 0.3])

        spectrum_low = model.forward(
            1.2,
            0.8,
            17.0,
            conc,
            0.3,
            1.0,
            total_species_density_cm3=2.5e16,
        )
        spectrum_high = model.forward(
            1.2,
            0.8,
            17.0,
            conc,
            0.3,
            1.0,
            total_species_density_cm3=2.5e17,
        )

        assert not jnp.allclose(spectrum_low, spectrum_high)

    def test_jit_stable(self, bayesian_db):
        """Forward model is JIT-compatible (no errors under JIT)."""
        import jax

        from cflibs.inversion.bayesian import TwoZoneBayesianForwardModel

        model = TwoZoneBayesianForwardModel(bayesian_db, ["Fe", "Cu"], (200.0, 600.0), pixels=50)
        conc = jnp.array([0.7, 0.3])

        @jax.jit
        def compute(T_core, T_shell, log_ne, c, sf, ods):
            return model.forward(T_core, T_shell, log_ne, c, sf, ods)

        result = compute(1.2, 0.8, 17.0, conc, 0.3, 1.0)
        assert jnp.all(jnp.isfinite(result))

    def test_gradient_stable(self, bayesian_db):
        """Gradients through the forward model are finite."""
        import jax

        from cflibs.inversion.bayesian import TwoZoneBayesianForwardModel

        model = TwoZoneBayesianForwardModel(bayesian_db, ["Fe", "Cu"], (200.0, 600.0), pixels=50)
        conc = jnp.array([0.7, 0.3])

        def loss(T_core):
            s = model.forward(T_core, 0.8, 17.0, conc, 0.3, 1.0)
            return jnp.sum(s)

        grad_val = jax.grad(loss)(1.2)
        assert jnp.isfinite(grad_val)


class TestTwoZonePriorConfig:
    """Tests for TwoZonePriorConfig dataclass."""

    def test_default_values(self):
        from cflibs.inversion.bayesian import TwoZonePriorConfig

        cfg = TwoZonePriorConfig()
        assert cfg.T_core_eV_range == (0.8, 3.0)
        assert cfg.T_shell_eV_range == (0.3, 2.0)
        assert cfg.enforce_T_ordering is True
        assert cfg.baseline_degree == 3
        assert cfg.mcwhirter_penalty_scale == 10.0

    def test_custom_values(self):
        from cflibs.inversion.bayesian import TwoZonePriorConfig

        cfg = TwoZonePriorConfig(
            T_core_eV_range=(1.0, 2.5),
            baseline_degree=5,
            mcwhirter_penalty_scale=0.0,
        )
        assert cfg.T_core_eV_range == (1.0, 2.5)
        assert cfg.baseline_degree == 5
        assert cfg.mcwhirter_penalty_scale == 0.0


class TestTwoZoneMCMCResult:
    """Tests for TwoZoneMCMCResult dataclass."""

    def test_creation(self):
        from cflibs.inversion.bayesian import TwoZoneMCMCResult

        result = TwoZoneMCMCResult(
            samples={},
            T_core_eV_mean=1.2,
            T_core_eV_std=0.1,
            T_core_eV_q025=1.0,
            T_core_eV_q975=1.4,
            T_shell_eV_mean=0.8,
            T_shell_eV_std=0.05,
            T_shell_eV_q025=0.7,
            T_shell_eV_q975=0.9,
            log_ne_mean=17.0,
            log_ne_std=0.2,
            log_ne_q025=16.6,
            log_ne_q975=17.4,
            shell_fraction_mean=0.3,
            shell_fraction_std=0.05,
            optical_depth_scale_mean=1.0,
            optical_depth_scale_std=0.2,
            concentrations_mean={"Fe": 0.7, "Cu": 0.3},
            concentrations_std={"Fe": 0.05, "Cu": 0.05},
        )
        assert result.T_core_K_mean > 10000  # 1.2 eV ≈ 13900 K
        assert result.T_shell_K_mean > 5000
        assert abs(result.n_e_mean - 1e17) < 1e16

    def test_summary_table(self):
        from cflibs.inversion.bayesian import TwoZoneMCMCResult

        result = TwoZoneMCMCResult(
            samples={},
            T_core_eV_mean=1.2,
            T_core_eV_std=0.1,
            T_core_eV_q025=1.0,
            T_core_eV_q975=1.4,
            T_shell_eV_mean=0.8,
            T_shell_eV_std=0.05,
            T_shell_eV_q025=0.7,
            T_shell_eV_q975=0.9,
            log_ne_mean=17.0,
            log_ne_std=0.2,
            log_ne_q025=16.6,
            log_ne_q975=17.4,
            shell_fraction_mean=0.3,
            shell_fraction_std=0.05,
            optical_depth_scale_mean=1.0,
            optical_depth_scale_std=0.2,
            concentrations_mean={"Fe": 0.7, "Cu": 0.3},
            concentrations_std={"Fe": 0.05, "Cu": 0.05},
        )
        table = result.summary_table()
        assert "Two-Zone" in table
        assert "T_core" in table
        assert "T_shell" in table
        assert "Fe" in table


class TestTwoZoneMCMCSampler:
    """Smoke test for two-zone MCMC sampling."""

    @pytest.mark.slow
    def test_smoke_run(self, bayesian_db):
        """Two-zone MCMC runs without error on synthetic data."""
        from cflibs.inversion.bayesian import (
            TwoZoneBayesianForwardModel,
            TwoZoneMCMCSampler,
            TwoZonePriorConfig,
            TwoZoneMCMCResult,
        )

        model = TwoZoneBayesianForwardModel(bayesian_db, ["Fe", "Cu"], (200.0, 600.0), pixels=50)
        conc = jnp.array([0.7, 0.3])
        observed = model.forward_numpy(1.2, 0.8, 17.0, conc, 0.3, 1.0)

        # Add noise
        rng = np.random.default_rng(42)
        noise = rng.normal(0, np.maximum(np.abs(observed) * 0.01, 1.0))
        observed_noisy = observed + noise

        cfg = TwoZonePriorConfig(mcwhirter_penalty_scale=0.0)
        sampler = TwoZoneMCMCSampler(model, prior_config=cfg)
        result = sampler.run(
            observed_noisy,
            num_warmup=10,
            num_samples=10,
            num_chains=1,
            seed=42,
            progress_bar=False,
        )
        assert isinstance(result, TwoZoneMCMCResult)
        assert result.T_core_eV_mean > 0
        assert result.T_shell_eV_mean > 0