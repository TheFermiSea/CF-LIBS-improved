"""
Tests for Physics-Informed Neural Networks module.

These tests validate:
1. Physics constraint functions (Boltzmann, Saha, closure)
2. PhysicsConstraintConfig and PINNConfig dataclasses
3. DifferentiableForwardModel spectrum computation
4. PINNEncoder architecture and forward pass
5. PhysicsConstraintLayer loss computation
6. PINNInverter training and inference (integration)

Requirements: JAX, Equinox (optional), Optax (optional)
"""

import pytest
import numpy as np
import sqlite3
import tempfile
from pathlib import Path

# Mark entire module as requiring JAX
pytestmark = [
    pytest.mark.requires_jax,
]

# Skip all tests if JAX is not available
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
import jax.random as random  # noqa: E402


class TestPhysicsConstraintConfig:
    """Tests for PhysicsConstraintConfig dataclass."""

    def test_default_values(self):
        """Test default physics constraint configuration."""
        from cflibs.inversion.pinn import PhysicsConstraintConfig

        config = PhysicsConstraintConfig()
        assert config.lambda_boltzmann == 1.0
        assert config.lambda_saha == 1.0
        assert config.lambda_closure == 10.0
        assert config.lambda_energy == 0.1
        assert config.temperature_range_eV == (0.3, 3.0)
        assert config.density_range_log == (15.0, 19.0)

    def test_custom_values(self):
        """Test custom physics constraint configuration."""
        from cflibs.inversion.pinn import PhysicsConstraintConfig

        config = PhysicsConstraintConfig(
            lambda_boltzmann=2.0,
            lambda_saha=0.5,
            lambda_closure=5.0,
            temperature_range_eV=(0.5, 2.5),
        )
        assert config.lambda_boltzmann == 2.0
        assert config.lambda_saha == 0.5
        assert config.lambda_closure == 5.0
        assert config.temperature_range_eV == (0.5, 2.5)


class TestPINNConfig:
    """Tests for PINNConfig dataclass."""

    def test_default_values(self):
        """Test default PINN configuration."""
        from cflibs.inversion.pinn import PINNConfig

        config = PINNConfig()
        assert config.encoder_hidden_dims == [256, 128, 64]
        assert config.activation == "gelu"
        assert config.dropout_rate == 0.1
        assert config.use_batch_norm is True
        assert config.n_ensemble == 1
        assert config.learning_rate == 1e-3

    def test_custom_hidden_dims(self):
        """Test custom hidden dimensions."""
        from cflibs.inversion.pinn import PINNConfig

        config = PINNConfig(
            encoder_hidden_dims=[512, 256, 128],
            activation="relu",
            n_ensemble=5,
        )
        assert config.encoder_hidden_dims == [512, 256, 128]
        assert config.activation == "relu"
        assert config.n_ensemble == 5


class TestPINNResult:
    """Tests for PINNResult dataclass."""

    def test_temperature_conversion(self):
        """Test temperature unit conversion."""
        from cflibs.inversion.pinn import PINNResult
        from cflibs.core.constants import EV_TO_K

        result = PINNResult(
            temperature_eV=1.0,
            temperature_uncertainty_eV=0.1,
            electron_density_cm3=1e17,
            density_uncertainty_cm3=1e16,
            concentrations={"Fe": 0.9, "Cu": 0.1},
            concentration_uncertainties={"Fe": 0.05, "Cu": 0.05},
            physics_loss={"closure": 0.001},
            total_loss=0.01,
            converged=True,
            epochs=1000,
        )

        assert result.temperature_K == pytest.approx(1.0 * EV_TO_K)
        assert result.log_ne == pytest.approx(17.0, rel=1e-3)

    def test_summary_generation(self):
        """Test summary string generation."""
        from cflibs.inversion.pinn import PINNResult

        result = PINNResult(
            temperature_eV=1.0,
            temperature_uncertainty_eV=0.1,
            electron_density_cm3=1e17,
            density_uncertainty_cm3=1e16,
            concentrations={"Fe": 0.9, "Cu": 0.1},
            concentration_uncertainties={"Fe": 0.05, "Cu": 0.05},
            physics_loss={"closure": 0.001, "positivity": 0.0},
            total_loss=0.01,
            converged=True,
            epochs=1000,
        )

        summary = result.summary()
        assert "PINN Inversion Result" in summary
        assert "Temperature" in summary
        assert "Fe" in summary
        assert "closure" in summary


class TestPhysicsConstraintFunctions:
    """Tests for physics constraint functions."""

    def test_closure_residual_perfect(self):
        """Test closure residual with perfect sum."""
        from cflibs.inversion.pinn import closure_residual

        concentrations = jnp.array([0.5, 0.3, 0.2])
        residual = closure_residual(concentrations)
        assert float(residual) == pytest.approx(0.0, abs=1e-10)

    def test_closure_residual_violation(self):
        """Test closure residual with sum != 1."""
        from cflibs.inversion.pinn import closure_residual

        # Sum to 0.9
        concentrations = jnp.array([0.5, 0.3, 0.1])
        residual = closure_residual(concentrations)
        assert float(residual) == pytest.approx(0.01, rel=1e-3)  # (0.9 - 1)^2

    def test_positivity_penalty_all_positive(self):
        """Test positivity penalty with all positive values."""
        from cflibs.inversion.pinn import positivity_penalty

        values = jnp.array([0.1, 0.5, 0.3, 0.1])
        penalty = positivity_penalty(values)
        assert float(penalty) == pytest.approx(0.0, abs=1e-10)

    def test_positivity_penalty_negative_values(self):
        """Test positivity penalty with negative values."""
        from cflibs.inversion.pinn import positivity_penalty

        values = jnp.array([0.5, -0.1, 0.3, -0.2])
        penalty = positivity_penalty(values)
        # Should penalize the negative values
        assert penalty > 0

    def test_range_penalty_inside(self):
        """Test range penalty with value inside range."""
        from cflibs.inversion.pinn import range_penalty

        penalty = range_penalty(1.0, low=0.5, high=2.0)
        # Inside range should have very small penalty
        assert float(penalty) < 0.1

    def test_range_penalty_outside(self):
        """Test range penalty with value outside range."""
        from cflibs.inversion.pinn import range_penalty

        penalty_below = range_penalty(0.1, low=0.5, high=2.0)
        penalty_above = range_penalty(3.0, low=0.5, high=2.0)

        # Outside range should have larger penalty
        assert penalty_below > 1.0
        assert penalty_above > 1.0

    def test_boltzmann_residual_equilibrium(self):
        """Test Boltzmann residual at equilibrium."""
        from cflibs.inversion.pinn import boltzmann_residual

        T_eV = 1.0
        # Set up populations that satisfy Boltzmann distribution
        energies = jnp.array([0.0, 1.0, 2.0, 3.0])
        degeneracies = jnp.array([2, 4, 6, 8])
        # Boltzmann populations: n_k = n_0 * (g_k/g_0) * exp(-E_k/T)
        n_0 = 1000.0
        populations = n_0 * (degeneracies / degeneracies[0]) * jnp.exp(-energies / T_eV)

        residual = boltzmann_residual(T_eV, energies, populations, degeneracies)
        assert float(residual) == pytest.approx(0.0, abs=1e-6)

    def test_boltzmann_residual_non_equilibrium(self):
        """Test Boltzmann residual away from equilibrium."""
        from cflibs.inversion.pinn import boltzmann_residual

        T_eV = 1.0
        energies = jnp.array([0.0, 1.0, 2.0, 3.0])
        degeneracies = jnp.array([2, 4, 6, 8])
        # Wrong populations (uniform instead of Boltzmann)
        populations = jnp.array([100.0, 100.0, 100.0, 100.0])

        residual = boltzmann_residual(T_eV, energies, populations, degeneracies)
        assert float(residual) > 0.1  # Should be significant

    def test_saha_residual_equilibrium(self):
        """Test Saha residual at ionization equilibrium."""
        from cflibs.inversion.pinn import saha_residual, SAHA_CONST_CM3

        T_eV = 1.0
        n_e = 1e17
        IP_eV = jnp.array([7.87])  # Fe

        # Partition functions (simplified)
        U_neutral = jnp.array([25.0])
        U_ion = jnp.array([15.0])

        # Calculate expected Saha ratio: (n_ion * n_e) / n_neutral = expected_ratio
        saha_factor = (SAHA_CONST_CM3 / n_e) * (T_eV**1.5)
        expected_ratio = (U_ion / U_neutral) * saha_factor * jnp.exp(-IP_eV / T_eV)

        # Set populations that satisfy Saha: n_ion = n_neutral * expected_ratio / n_e
        n_neutral = jnp.array([1000.0])
        n_ion = n_neutral * expected_ratio / n_e

        residual = saha_residual(T_eV, n_e, n_neutral, n_ion, U_neutral, U_ion, IP_eV)
        assert float(residual) == pytest.approx(0.0, abs=1e-4)


class TestDifferentiableForwardModel:
    """Tests for DifferentiableForwardModel."""

    @pytest.fixture
    def simple_forward_model(self):
        """Create a simple forward model for testing."""
        from cflibs.inversion.pinn import DifferentiableForwardModel

        wavelength = np.linspace(300, 400, 500)
        elements = ["Fe", "Cu"]

        # Simple line data
        line_positions = np.array([325.0, 350.0, 375.0, 330.0, 360.0])
        line_gA = np.array([1e7, 5e6, 2e6, 1e8, 5e7])
        line_Ek = np.array([3.0, 3.5, 4.0, 3.8, 4.2])
        line_element_idx = np.array([0, 0, 0, 1, 1])

        return DifferentiableForwardModel(
            wavelength=wavelength,
            elements=elements,
            line_positions=line_positions,
            line_gA=line_gA,
            line_Ek=line_Ek,
            line_element_idx=line_element_idx,
        )

    def test_initialization(self, simple_forward_model):
        """Test forward model initialization."""
        assert simple_forward_model.n_elements == 2
        assert len(simple_forward_model.wavelength) == 500
        assert simple_forward_model.elements == ["Fe", "Cu"]

    def test_forward_shape(self, simple_forward_model):
        """Test forward model output shape."""
        T_eV = 1.0
        log_ne = 17.0
        concentrations = jnp.array([0.9, 0.1])

        spectrum = simple_forward_model.forward(T_eV, log_ne, concentrations)
        assert spectrum.shape == (500,)

    def test_forward_non_negative(self, simple_forward_model):
        """Test that forward model produces non-negative spectra."""
        T_eV = 1.0
        log_ne = 17.0
        concentrations = jnp.array([0.5, 0.5])

        spectrum = simple_forward_model.forward(T_eV, log_ne, concentrations)
        assert jnp.all(spectrum >= 0)

    def test_forward_temperature_dependence(self, simple_forward_model):
        """Test that spectrum changes with temperature."""
        log_ne = 17.0
        concentrations = jnp.array([0.9, 0.1])

        spectrum_low_T = simple_forward_model.forward(0.5, log_ne, concentrations)
        spectrum_high_T = simple_forward_model.forward(2.0, log_ne, concentrations)

        # Higher temperature should generally increase intensity for typical lines
        # (due to Boltzmann factor at moderate energies)
        assert not jnp.allclose(spectrum_low_T, spectrum_high_T)

    def test_forward_concentration_dependence(self, simple_forward_model):
        """Test that spectrum changes with concentrations."""
        T_eV = 1.0
        log_ne = 17.0

        spectrum_fe = simple_forward_model.forward(T_eV, log_ne, jnp.array([1.0, 0.0]))
        spectrum_cu = simple_forward_model.forward(T_eV, log_ne, jnp.array([0.0, 1.0]))
        spectrum_mix = simple_forward_model.forward(T_eV, log_ne, jnp.array([0.5, 0.5]))

        # All should be different
        assert not jnp.allclose(spectrum_fe, spectrum_cu)
        assert not jnp.allclose(spectrum_fe, spectrum_mix)


# Skip Equinox-dependent tests if not available
equinox = pytest.importorskip("equinox", reason="Equinox required for neural network tests")
optax = pytest.importorskip("optax", reason="Optax required for training tests")


class TestPhysicsConstraintLayer:
    """Tests for PhysicsConstraintLayer."""

    def test_initialization(self):
        """Test physics constraint layer initialization."""
        from cflibs.inversion.pinn import PhysicsConstraintLayer

        layer = PhysicsConstraintLayer(
            n_elements=3,
            ionization_potentials=np.array([7.87, 7.73, 5.99]),
        )
        assert layer.n_elements == 3
        assert len(layer.ionization_potentials) == 3

    def test_forward_loss_computation(self):
        """Test physics constraint loss computation."""
        from cflibs.inversion.pinn import PhysicsConstraintLayer

        layer = PhysicsConstraintLayer(
            n_elements=2,
            ionization_potentials=np.array([7.87, 7.73]),
        )

        losses = layer(
            T_eV=1.0,
            log_ne=17.0,
            concentrations=jnp.array([0.9, 0.1]),
        )

        # Should contain expected loss keys
        assert "closure" in losses
        assert "positivity" in losses
        assert "T_range" in losses
        assert "ne_range" in losses

        # Valid parameters should have small losses
        assert float(losses["closure"]) < 1.0
        assert float(losses["positivity"]) < 0.1

    def test_closure_loss_for_bad_concentrations(self):
        """Test closure loss penalizes bad concentrations."""
        from cflibs.inversion.pinn import PhysicsConstraintLayer

        layer = PhysicsConstraintLayer(
            n_elements=2,
            ionization_potentials=np.array([7.87, 7.73]),
        )

        # Good concentrations (sum to 1)
        losses_good = layer(1.0, 17.0, jnp.array([0.6, 0.4]))
        # Bad concentrations (sum to 0.9)
        losses_bad = layer(1.0, 17.0, jnp.array([0.5, 0.4]))

        assert float(losses_bad["closure"]) > float(losses_good["closure"])

    def test_range_loss_for_out_of_bounds(self):
        """Test range loss penalizes out-of-bounds parameters."""
        from cflibs.inversion.pinn import PhysicsConstraintLayer, PhysicsConstraintConfig

        config = PhysicsConstraintConfig(
            temperature_range_eV=(0.5, 2.0),
            density_range_log=(16.0, 18.0),
        )
        layer = PhysicsConstraintLayer(
            n_elements=2,
            ionization_potentials=np.array([7.87, 7.73]),
            config=config,
        )

        # In-bounds
        losses_ok = layer(1.0, 17.0, jnp.array([0.6, 0.4]))
        # Temperature out of bounds
        losses_T_bad = layer(3.0, 17.0, jnp.array([0.6, 0.4]))
        # Density out of bounds
        losses_ne_bad = layer(1.0, 19.5, jnp.array([0.6, 0.4]))

        assert float(losses_T_bad["T_range"]) > float(losses_ok["T_range"])
        assert float(losses_ne_bad["ne_range"]) > float(losses_ok["ne_range"])


class TestMLP:
    """Tests for MLP neural network."""

    def test_initialization(self):
        """Test MLP initialization."""
        from cflibs.inversion.pinn import MLP

        key = random.PRNGKey(0)
        mlp = MLP(
            in_features=100,
            hidden_dims=[64, 32],
            out_features=10,
            activation="gelu",
            key=key,
        )
        assert len(mlp.layers) == 3  # input->64, 64->32, 32->output

    def test_forward_shape(self):
        """Test MLP forward pass shape."""
        from cflibs.inversion.pinn import MLP

        key = random.PRNGKey(0)
        mlp = MLP(
            in_features=100,
            hidden_dims=[64, 32],
            out_features=10,
            key=key,
        )

        x = jnp.ones(100)
        y = mlp(x)
        assert y.shape == (10,)

    def test_different_activations(self):
        """Test MLP with different activation functions."""
        from cflibs.inversion.pinn import MLP

        for activation in ["relu", "gelu", "tanh", "swish"]:
            key = random.PRNGKey(0)
            mlp = MLP(
                in_features=50,
                hidden_dims=[32],
                out_features=5,
                activation=activation,
                key=key,
            )
            x = jnp.ones(50)
            y = mlp(x)
            assert y.shape == (5,)


class TestPINNEncoder:
    """Tests for PINNEncoder network."""

    def test_initialization(self):
        """Test PINN encoder initialization."""
        from cflibs.inversion.pinn import PINNEncoder

        key = random.PRNGKey(0)
        encoder = PINNEncoder(
            n_wavelengths=500,
            n_elements=3,
            hidden_dims=[128, 64, 32],
            key=key,
        )
        assert encoder.n_elements == 3

    def test_forward_outputs(self):
        """Test PINN encoder forward pass outputs."""
        from cflibs.inversion.pinn import PINNEncoder

        key = random.PRNGKey(0)
        encoder = PINNEncoder(
            n_wavelengths=500,
            n_elements=3,
            T_range=(0.5, 2.0),
            ne_range=(16.0, 18.0),
            key=key,
        )

        spectrum = jnp.ones(500)
        T_eV, log_ne, concentrations = encoder(spectrum)

        # Check output types and shapes
        assert T_eV.shape == ()  # scalar
        assert log_ne.shape == ()  # scalar
        assert concentrations.shape == (3,)

        # Check bounds
        assert 0.5 <= float(T_eV) <= 2.0
        assert 16.0 <= float(log_ne) <= 18.0

        # Check concentrations sum to 1
        assert float(jnp.sum(concentrations)) == pytest.approx(1.0, abs=1e-5)

        # Check concentrations are all positive
        assert jnp.all(concentrations >= 0)

    def test_encoder_differentiable(self):
        """Test that encoder is differentiable."""
        from cflibs.inversion.pinn import PINNEncoder

        key = random.PRNGKey(0)
        encoder = PINNEncoder(
            n_wavelengths=100,
            n_elements=2,
            hidden_dims=[32, 16],
            key=key,
        )

        @equinox.filter_value_and_grad
        def loss_fn(enc):
            spectrum = jnp.ones(100)
            T, log_ne, conc = enc(spectrum)
            return T + log_ne + jnp.sum(conc)

        # Should not raise
        loss, grads = loss_fn(encoder)
        assert grads is not None
        assert loss is not None


class TestPINNInverter:
    """Tests for PINNInverter (integration tests)."""

    @pytest.fixture
    def simple_inverter(self):
        """Create a simple PINN inverter for testing."""
        from cflibs.inversion.pinn import PINNInverter, PINNConfig, PhysicsConstraintConfig

        physics_config = PhysicsConstraintConfig(
            temperature_range_eV=(0.5, 2.0),
            density_range_log=(16.0, 18.0),
        )
        config = PINNConfig(
            encoder_hidden_dims=[64, 32],
            n_ensemble=1,
            learning_rate=1e-3,
            physics_config=physics_config,
        )

        return PINNInverter(
            n_wavelengths=100,
            elements=["Fe", "Cu"],
            ionization_potentials=np.array([7.87, 7.73]),
            config=config,
            seed=42,
        )

    def test_initialization(self, simple_inverter):
        """Test PINN inverter initialization."""
        assert simple_inverter.n_elements == 2
        assert simple_inverter.n_wavelengths == 100
        assert len(simple_inverter.encoders) == 1  # n_ensemble = 1

    def test_invert_without_training(self, simple_inverter):
        """Test that invert works without training (with warning)."""
        spectrum = np.random.rand(100)
        result = simple_inverter.invert(spectrum)

        assert result is not None
        assert "Fe" in result.concentrations
        assert "Cu" in result.concentrations
        assert 0.5 <= result.temperature_eV <= 2.0

    @pytest.mark.slow
    def test_training_basic(self, simple_inverter):
        """Test basic training loop."""
        # Generate simple synthetic data
        n_samples = 50
        spectra = np.random.rand(n_samples, 100)
        temperatures = np.random.uniform(0.5, 2.0, n_samples)
        log_densities = np.random.uniform(16.0, 18.0, n_samples)
        concentrations = np.random.dirichlet([1, 1], n_samples)

        history = simple_inverter.train(
            spectra,
            temperatures,
            log_densities,
            concentrations,
            epochs=10,
            batch_size=10,
            validation_split=0.2,
            verbose=False,
        )

        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) > 0
        assert simple_inverter.trained

    @pytest.mark.slow
    def test_invert_after_training(self, simple_inverter):
        """Test inversion after training."""
        # Generate synthetic data
        n_samples = 50
        spectra = np.random.rand(n_samples, 100)
        temperatures = np.random.uniform(0.5, 2.0, n_samples)
        log_densities = np.random.uniform(16.0, 18.0, n_samples)
        concentrations = np.random.dirichlet([1, 1], n_samples)

        simple_inverter.train(
            spectra,
            temperatures,
            log_densities,
            concentrations,
            epochs=10,
            batch_size=10,
            verbose=False,
        )

        # Invert a spectrum
        result = simple_inverter.invert(spectra[0])

        assert result.converged is True
        assert result.epochs == 10
        assert result.temperature_eV > 0
        assert sum(result.concentrations.values()) == pytest.approx(1.0, abs=0.01)


class TestSyntheticDataGeneration:
    """Tests for synthetic training data generation."""

    @pytest.fixture
    def forward_model(self):
        """Create forward model for data generation."""
        from cflibs.inversion.pinn import DifferentiableForwardModel

        return DifferentiableForwardModel(
            wavelength=np.linspace(300, 400, 200),
            elements=["Fe", "Cu"],
            line_positions=np.array([325.0, 350.0, 375.0, 330.0, 360.0]),
            line_gA=np.array([1e7, 5e6, 2e6, 1e8, 5e7]),
            line_Ek=np.array([3.0, 3.5, 4.0, 3.8, 4.2]),
            line_element_idx=np.array([0, 0, 0, 1, 1]),
        )

    def test_generate_data_shapes(self, forward_model):
        """Test generated data shapes."""
        from cflibs.inversion.pinn import generate_synthetic_training_data

        spectra, temps, log_nes, concs = generate_synthetic_training_data(
            forward_model,
            n_samples=20,
            seed=42,
        )

        assert spectra.shape == (20, 200)
        assert temps.shape == (20,)
        assert log_nes.shape == (20,)
        assert concs.shape == (20, 2)

    def test_generate_data_ranges(self, forward_model):
        """Test generated data is within specified ranges."""
        from cflibs.inversion.pinn import generate_synthetic_training_data

        spectra, temps, log_nes, concs = generate_synthetic_training_data(
            forward_model,
            n_samples=100,
            T_range=(0.8, 1.5),
            log_ne_range=(16.5, 17.5),
            seed=42,
        )

        assert np.all(temps >= 0.8) and np.all(temps <= 1.5)
        assert np.all(log_nes >= 16.5) and np.all(log_nes <= 17.5)
        assert np.allclose(concs.sum(axis=1), 1.0)

    def test_generate_data_non_negative_spectra(self, forward_model):
        """Test that generated spectra are non-negative."""
        from cflibs.inversion.pinn import generate_synthetic_training_data

        spectra, _, _, _ = generate_synthetic_training_data(
            forward_model,
            n_samples=50,
            noise_level=0.1,
            seed=42,
        )

        assert np.all(spectra >= 0)


class TestCreatePINNFromDatabase:
    """Tests for create_pinn_from_database function."""

    @pytest.fixture
    def pinn_db(self):
        """Create a database for PINN initialization."""
        import os

        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE species_physics (
                element TEXT,
                sp_num INTEGER,
                ip_ev REAL,
                PRIMARY KEY (element, sp_num)
            )
        """)
        conn.executemany(
            """
            INSERT INTO species_physics (element, sp_num, ip_ev) VALUES (?, ?, ?)
        """,
            [
                ("Fe", 1, 7.87),
                ("Cu", 1, 7.73),
                ("Mn", 1, 7.43),
            ],
        )
        conn.commit()
        conn.close()

        yield db_path
        Path(db_path).unlink()

    def test_create_from_database(self, pinn_db):
        """Test creating PINN from database."""
        from cflibs.inversion.pinn import create_pinn_from_database

        inverter = create_pinn_from_database(
            pinn_db,
            elements=["Fe", "Cu"],
            wavelength_range=(300, 400),
            n_wavelengths=500,
        )

        assert inverter.n_elements == 2
        assert inverter.n_wavelengths == 500
        assert inverter.elements == ["Fe", "Cu"]

    def test_create_missing_element(self, pinn_db):
        """Test creating PINN with missing element uses default IP."""
        from cflibs.inversion.pinn import create_pinn_from_database

        # "Ni" is not in the database
        inverter = create_pinn_from_database(
            pinn_db,
            elements=["Fe", "Ni"],
            wavelength_range=(300, 400),
            n_wavelengths=500,
        )

        # Should still create (with warning and default IP)
        assert inverter.n_elements == 2