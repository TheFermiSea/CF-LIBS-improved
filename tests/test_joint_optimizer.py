"""
Tests for multi-element joint optimization.

These tests validate the simultaneous optimization of temperature, electron density,
and element concentrations using the JointOptimizer class.

Requirements: JAX
"""

import pytest
import numpy as np

# Mark entire module as requiring JAX
pytestmark = pytest.mark.requires_jax

# Skip all tests if JAX is not available
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402


from cflibs.inversion.solve.joint_optimizer import (  # noqa: E402
    JointOptimizer,
    JointOptimizationResult,
    MultiStartJointOptimizer,
    LossType,
    ConvergenceStatus,
    create_simple_forward_model,
)


@pytest.fixture
def simple_forward_model():
    """Create a simple forward model for testing."""
    elements = ["Fe", "Cu"]
    line_centers = {
        "Fe": [400.0, 430.0, 440.0],
        "Cu": [500.0, 510.0, 520.0],
    }
    line_strengths = {
        "Fe": [1.0, 0.5, 0.3],
        "Cu": [1.0, 0.6, 0.4],
    }
    return create_simple_forward_model(elements, line_centers, line_strengths)


@pytest.fixture
def wavelength_grid():
    """Create wavelength grid for testing."""
    return np.linspace(350, 600, 200)


@pytest.fixture
def optimizer(simple_forward_model, wavelength_grid):
    """Create JointOptimizer instance for testing."""
    return JointOptimizer(
        forward_model=simple_forward_model,
        elements=["Fe", "Cu"],
        wavelength=wavelength_grid,
        loss_type=LossType.CHI_SQUARED,
        max_iterations=100,
    )


@pytest.fixture
def synthetic_spectrum(simple_forward_model, wavelength_grid):
    """Generate synthetic spectrum with known parameters."""
    T_true = 1.5  # eV
    n_e_true = 1e17  # cm^-3
    conc_true = {"Fe": 0.7, "Cu": 0.3}

    conc_arr = jnp.array([conc_true["Fe"], conc_true["Cu"]])
    wl = jnp.array(wavelength_grid)

    spectrum = simple_forward_model(T_true, n_e_true, conc_arr, wl)
    spectrum = np.array(spectrum)

    return {
        "spectrum": spectrum,
        "T_true": T_true,
        "n_e_true": n_e_true,
        "conc_true": conc_true,
    }


class TestJointOptimizationResult:
    """Tests for the JointOptimizationResult dataclass."""

    def test_result_creation(self):
        """Test basic result creation."""
        result = JointOptimizationResult(
            temperature_eV=1.2,
            electron_density_cm3=1e17,
            concentrations={"Fe": 0.7, "Cu": 0.3},
            initial_temperature_eV=1.0,
            initial_electron_density_cm3=1e17,
            initial_concentrations={"Fe": 0.5, "Cu": 0.5},
            final_loss=0.01,
            chi_squared=100.0,
            reduced_chi_squared=1.05,
            degrees_of_freedom=95,
            convergence_status=ConvergenceStatus.CONVERGED,
            iterations=50,
        )

        assert result.temperature_eV == 1.2
        assert result.electron_density_cm3 == 1e17
        assert result.concentrations["Fe"] == 0.7
        assert result.is_converged

    def test_temperature_K_property(self):
        """Test temperature Kelvin property."""
        result = JointOptimizationResult(
            temperature_eV=1.0,  # ~11604 K
            electron_density_cm3=1e17,
            concentrations={"Fe": 1.0},
            initial_temperature_eV=1.0,
            initial_electron_density_cm3=1e17,
            initial_concentrations={"Fe": 1.0},
            final_loss=0.0,
        )

        # 1 eV ~ 11604 K
        np.testing.assert_allclose(result.temperature_K, 11604.5, rtol=0.01)

    def test_log_ne_property(self):
        """Test log10(n_e) property."""
        result = JointOptimizationResult(
            temperature_eV=1.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": 1.0},
            initial_temperature_eV=1.0,
            initial_electron_density_cm3=1e17,
            initial_concentrations={"Fe": 1.0},
            final_loss=0.0,
        )

        assert result.log_ne == 17.0

    def test_goodness_of_fit_interpretation(self):
        """Test reduced chi-squared interpretation."""
        # Good fit
        result = JointOptimizationResult(
            temperature_eV=1.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": 1.0},
            initial_temperature_eV=1.0,
            initial_electron_density_cm3=1e17,
            initial_concentrations={"Fe": 1.0},
            final_loss=0.0,
            reduced_chi_squared=1.1,
        )
        assert "good fit" in result.goodness_of_fit

        # Overfit
        result.reduced_chi_squared = 0.3
        assert "overfitting" in result.goodness_of_fit

        # Poor fit
        result.reduced_chi_squared = 5.0
        assert "poor fit" in result.goodness_of_fit

    def test_summary(self):
        """Test result summary generation."""
        result = JointOptimizationResult(
            temperature_eV=1.2,
            electron_density_cm3=1e17,
            concentrations={"Fe": 0.7, "Cu": 0.3},
            initial_temperature_eV=1.0,
            initial_electron_density_cm3=1e17,
            initial_concentrations={"Fe": 0.5, "Cu": 0.5},
            final_loss=0.01,
            chi_squared=100.0,
            reduced_chi_squared=1.05,
            degrees_of_freedom=95,
            convergence_status=ConvergenceStatus.CONVERGED,
            iterations=50,
            parameter_uncertainties={"T_eV": 0.05, "C_Fe": 0.02},
        )

        summary = result.summary()
        assert "Joint Optimization Result" in summary
        assert "Fe" in summary
        assert "Cu" in summary
        assert "converged" in summary
        assert "Chi-squared" in summary


class TestJointOptimizer:
    """Tests for the JointOptimizer class."""

    def test_init(self, simple_forward_model, wavelength_grid):
        """Test JointOptimizer initialization."""
        optimizer = JointOptimizer(
            forward_model=simple_forward_model,
            elements=["Fe", "Cu"],
            wavelength=wavelength_grid,
        )

        assert optimizer.n_elements == 2
        assert "Fe" in optimizer.elements
        assert "Cu" in optimizer.elements
        assert optimizer.n_wavelength == 200
        assert optimizer.n_params == 4  # log(T) + log10(n_e) + 2 concentrations

    def test_init_with_loss_type_string(self, simple_forward_model, wavelength_grid):
        """Test initialization with string loss type."""
        optimizer = JointOptimizer(
            forward_model=simple_forward_model,
            elements=["Fe", "Cu"],
            wavelength=wavelength_grid,
            loss_type="chi_squared",
        )

        assert optimizer.loss_type == LossType.CHI_SQUARED

    def test_pack_unpack_params(self, optimizer):
        """Test parameter packing and unpacking roundtrip."""
        T_eV = 1.5
        n_e = 5e17
        conc = {"Fe": 0.7, "Cu": 0.3}

        packed = optimizer._pack_params(T_eV, n_e, conc)
        T_out, ne_out, conc_out = optimizer._unpack_params(packed)

        # Temperature and density should round-trip
        np.testing.assert_allclose(float(T_out), T_eV, rtol=0.01)
        np.testing.assert_allclose(float(ne_out), n_e, rtol=0.01)

        # Concentrations should sum to 1 (softmax)
        np.testing.assert_allclose(float(jnp.sum(conc_out)), 1.0, rtol=0.01)

        # Ratios should be preserved
        ratio_in = conc["Fe"] / conc["Cu"]
        ratio_out = float(conc_out[0] / conc_out[1])
        np.testing.assert_allclose(ratio_out, ratio_in, rtol=0.05)

    def test_softmax_simplex_constraint(self, optimizer):
        """Test that softmax enforces simplex constraint."""
        # Various concentration scenarios
        test_cases = [
            {"Fe": 0.9, "Cu": 0.1},
            {"Fe": 0.1, "Cu": 0.9},
            {"Fe": 0.5, "Cu": 0.5},
            {"Fe": 0.01, "Cu": 0.99},
        ]

        for conc in test_cases:
            packed = optimizer._pack_params(1.0, 1e17, conc)
            _, _, conc_out = optimizer._unpack_params(packed)

            # Should sum to 1
            np.testing.assert_allclose(float(jnp.sum(conc_out)), 1.0, rtol=1e-6)

            # All positive
            assert jnp.all(conc_out > 0)

    def test_optimize_recovers_parameters(self, optimizer, synthetic_spectrum):
        """Test that optimization recovers true parameters."""
        data = synthetic_spectrum

        # Add small noise
        rng = np.random.default_rng(42)
        noisy_spectrum = data["spectrum"] + rng.normal(0, 5, len(data["spectrum"]))
        noisy_spectrum = np.maximum(noisy_spectrum, 1.0)
        uncertainties = np.sqrt(noisy_spectrum)

        result = optimizer.optimize(
            noisy_spectrum,
            uncertainties=uncertainties,
            initial_T_eV=1.0,
            initial_n_e=1e17,
        )

        # Should recover temperature within ~30%
        np.testing.assert_allclose(result.temperature_eV, data["T_true"], rtol=0.3)

        # Concentrations should be close
        assert result.concentrations["Fe"] > 0.5  # True is 0.7
        assert result.concentrations["Cu"] > 0.1  # True is 0.3

        # Should sum to 1
        total_conc = sum(result.concentrations.values())
        np.testing.assert_allclose(total_conc, 1.0, rtol=1e-6)

    def test_optimize_with_default_init(self, optimizer, synthetic_spectrum):
        """Test optimization with default initial values."""
        data = synthetic_spectrum
        spectrum = data["spectrum"] + 1.0  # Ensure positive

        result = optimizer.optimize(spectrum)

        # Should return valid result
        assert result.temperature_eV > 0
        assert result.electron_density_cm3 > 0
        assert sum(result.concentrations.values()) > 0.99

    def test_optimize_spectrum_length_mismatch(self, optimizer):
        """Test error handling for mismatched spectrum length."""
        bad_spectrum = np.zeros(50)  # Optimizer expects 200 points

        with pytest.raises(ValueError, match="does not match"):
            optimizer.optimize(bad_spectrum)

    def test_optimize_returns_convergence_info(self, optimizer, synthetic_spectrum):
        """Test that optimization returns convergence information."""
        data = synthetic_spectrum

        result = optimizer.optimize(
            data["spectrum"],
            initial_T_eV=1.5,  # Close to true
            initial_n_e=1e17,
            initial_concentrations={"Fe": 0.6, "Cu": 0.4},
        )

        # Should have convergence info
        assert result.iterations > 0
        assert result.final_loss >= 0
        assert result.gradient_norm >= 0
        assert result.convergence_status in list(ConvergenceStatus)

    def test_optimize_returns_chi_squared(self, optimizer, synthetic_spectrum):
        """Test chi-squared computation."""
        data = synthetic_spectrum
        uncertainties = np.sqrt(np.maximum(data["spectrum"], 1.0))

        result = optimizer.optimize(
            data["spectrum"],
            uncertainties=uncertainties,
        )

        # Should have chi-squared stats
        assert result.chi_squared >= 0
        assert result.reduced_chi_squared >= 0
        assert result.degrees_of_freedom > 0

    def test_optimize_with_huber_loss(
        self, simple_forward_model, wavelength_grid, synthetic_spectrum
    ):
        """Test optimization with Huber loss for robustness."""
        optimizer = JointOptimizer(
            forward_model=simple_forward_model,
            elements=["Fe", "Cu"],
            wavelength=wavelength_grid,
            loss_type=LossType.HUBER,
        )

        data = synthetic_spectrum

        # Add outliers
        rng = np.random.default_rng(42)
        spectrum_with_outliers = data["spectrum"].copy()
        outlier_idx = rng.choice(len(spectrum_with_outliers), size=10, replace=False)
        spectrum_with_outliers[outlier_idx] += 1000  # Large outliers

        result = optimizer.optimize(spectrum_with_outliers)

        # Should still get reasonable result despite outliers
        assert result.temperature_eV > 0
        assert result.temperature_eV < 10  # Reasonable range

    def test_optimize_with_regularization(
        self, simple_forward_model, wavelength_grid, synthetic_spectrum
    ):
        """Test optimization with entropy regularization."""
        optimizer = JointOptimizer(
            forward_model=simple_forward_model,
            elements=["Fe", "Cu"],
            wavelength=wavelength_grid,
            regularization=0.1,
        )

        data = synthetic_spectrum

        result = optimizer.optimize(data["spectrum"])

        # Should return valid result
        assert result.temperature_eV > 0
        assert sum(result.concentrations.values()) > 0.99


class TestMultiStartJointOptimizer:
    """Tests for the MultiStartJointOptimizer class."""

    def test_init(self, optimizer):
        """Test MultiStartJointOptimizer initialization."""
        multi = MultiStartJointOptimizer(optimizer, n_starts=3)

        assert multi.n_starts == 3
        assert multi.optimizer is optimizer

    def test_multi_start_optimization(self, optimizer, synthetic_spectrum):
        """Test multi-start optimization."""
        multi = MultiStartJointOptimizer(optimizer, n_starts=3, seed=42)
        data = synthetic_spectrum

        result = multi.optimize(
            data["spectrum"],
            T_eV_range=(0.5, 2.5),
            n_e_range=(1e16, 1e18),
        )

        # Should return valid result
        assert result.temperature_eV > 0
        assert result.electron_density_cm3 > 0
        assert sum(result.concentrations.values()) > 0.99

        # Should have metadata about runs
        assert "n_starts" in result.metadata
        assert "n_successful" in result.metadata

    def test_multi_start_finds_better_minimum(self, simple_forward_model, wavelength_grid):
        """Test that multi-start can find better minima than single start."""
        # Create a more challenging optimization problem
        optimizer = JointOptimizer(
            forward_model=simple_forward_model,
            elements=["Fe", "Cu"],
            wavelength=wavelength_grid,
            max_iterations=50,  # Limited iterations
        )

        # Generate spectrum far from default initial guess
        T_true = 2.0
        n_e_true = 1e18
        conc_arr = jnp.array([0.3, 0.7])
        wl = jnp.array(wavelength_grid)
        spectrum = np.array(simple_forward_model(T_true, n_e_true, conc_arr, wl))
        spectrum = np.maximum(spectrum, 1.0)

        # Single start from default
        single_result = optimizer.optimize(
            spectrum,
            initial_T_eV=1.0,
            initial_n_e=1e17,
        )

        # Multi-start
        multi = MultiStartJointOptimizer(optimizer, n_starts=5, seed=42)
        multi_result = multi.optimize(
            spectrum,
            T_eV_range=(0.5, 3.0),
            n_e_range=(1e16, 1e19),
        )

        # Multi-start should find equal or better solution
        assert multi_result.final_loss <= single_result.final_loss * 1.1


class TestManyElementOptimization:
    """Tests for optimization with many elements (scalability)."""

    def test_ten_element_optimization(self, wavelength_grid):
        """Test optimization with 10 elements."""
        elements = ["Fe", "Cu", "Zn", "Ni", "Cr", "Mn", "Co", "Ti", "V", "Al"]

        # Create forward model with 10 elements
        line_centers = {el: [350 + i * 25] for i, el in enumerate(elements)}
        line_strengths = {el: [1.0] for el in elements}

        forward_model = create_simple_forward_model(elements, line_centers, line_strengths)

        optimizer = JointOptimizer(
            forward_model=forward_model,
            elements=elements,
            wavelength=wavelength_grid,
            max_iterations=50,
        )

        # Generate synthetic spectrum
        T_true = 1.5
        n_e_true = 1e17
        conc_arr = jnp.array([0.1] * 10)  # Equal concentrations
        spectrum = np.array(forward_model(T_true, n_e_true, conc_arr, jnp.array(wavelength_grid)))
        spectrum = np.maximum(spectrum, 1.0)

        result = optimizer.optimize(spectrum)

        # Should handle 10 elements
        assert len(result.concentrations) == 10
        assert result.temperature_eV > 0

        # Concentrations should sum to 1
        total = sum(result.concentrations.values())
        np.testing.assert_allclose(total, 1.0, rtol=1e-6)

    def test_many_element_parameter_count(self, wavelength_grid):
        """Test parameter count scales correctly with elements."""
        for n_el in [2, 5, 10, 15]:
            elements = [f"El{i}" for i in range(n_el)]
            line_centers = {el: [400 + i * 10] for i, el in enumerate(elements)}
            line_strengths = {el: [1.0] for el in elements}

            forward_model = create_simple_forward_model(elements, line_centers, line_strengths)

            optimizer = JointOptimizer(
                forward_model=forward_model,
                elements=elements,
                wavelength=wavelength_grid,
            )

            # Parameters = 2 (T, n_e) + n_elements
            assert optimizer.n_params == 2 + n_el


class TestProfileLikelihood:
    """Tests for profile likelihood computation."""

    def test_profile_likelihood_temperature(self, optimizer, synthetic_spectrum):
        """Test profile likelihood for temperature."""
        data = synthetic_spectrum

        # First optimize
        result = optimizer.optimize(
            data["spectrum"],
            initial_T_eV=data["T_true"],
            initial_n_e=data["n_e_true"],
            initial_concentrations=data["conc_true"],
        )

        # Compute profile likelihood
        param_values, profile_loss = optimizer.profile_likelihood(
            result,
            "T_eV",
            data["spectrum"],
            n_points=20,
            sigma_range=2.0,
        )

        # Should have parabolic shape near minimum
        assert len(param_values) == 20
        assert len(profile_loss) == 20

        # Minimum should be near optimum
        min_idx = np.argmin(profile_loss)
        assert profile_loss[min_idx] <= profile_loss[0]
        assert profile_loss[min_idx] <= profile_loss[-1]


class TestEdgeCases:
    """Edge case tests for JointOptimizer."""

    def test_zero_spectrum(self, optimizer):
        """Test handling of zero spectrum."""
        spectrum = np.ones(200) * 1e-10

        # Should not crash
        result = optimizer.optimize(spectrum)
        assert result.temperature_eV > 0

    def test_very_noisy_spectrum(self, optimizer, synthetic_spectrum):
        """Test handling of very noisy spectrum."""
        data = synthetic_spectrum

        # Pure noise
        rng = np.random.default_rng(42)
        noisy = data["spectrum"] + rng.normal(0, 100, len(data["spectrum"]))
        noisy = np.maximum(noisy, 1.0)

        # Should return some result
        result = optimizer.optimize(noisy)
        assert result.temperature_eV > 0
        # May not converge well
        assert result.iterations > 0

    def test_single_element(self, simple_forward_model, wavelength_grid):
        """Test with single element."""

        # Create single-element forward model
        @jax.jit
        def single_el_model(T, ne, conc, wl):
            sigma = 0.1 * jnp.sqrt(T)
            intensity = conc[0] * 1000 * jnp.exp(-2.0 / T)
            profile = jnp.exp(-0.5 * ((wl - 450) / sigma) ** 2)
            return intensity * profile / (sigma * jnp.sqrt(2 * jnp.pi))

        optimizer = JointOptimizer(
            forward_model=single_el_model,
            elements=["Fe"],
            wavelength=wavelength_grid,
        )

        # Single element should have concentration = 1
        spectrum = np.array(
            single_el_model(1.5, 1e17, jnp.array([1.0]), jnp.array(wavelength_grid))
        )
        spectrum = np.maximum(spectrum, 1.0)

        result = optimizer.optimize(spectrum)

        # Single element must have concentration ~1
        np.testing.assert_allclose(result.concentrations["Fe"], 1.0, rtol=1e-5)

    def test_extreme_temperature_guess(self, optimizer, synthetic_spectrum):
        """Test recovery from extreme initial temperature.

        Note: With simplified forward models, extreme starting points may
        converge to local minima. The important thing is that the optimizer
        returns valid results without crashing.
        """
        data = synthetic_spectrum

        # Very low initial temperature - should at least be positive
        result_low = optimizer.optimize(
            data["spectrum"],
            initial_T_eV=0.1,  # Very low
            initial_n_e=1e17,
        )
        assert result_low.temperature_eV > 0  # Must be positive
        assert result_low.convergence_status is not None

        # Very high initial temperature - should return valid result
        result_high = optimizer.optimize(
            data["spectrum"],
            initial_T_eV=10.0,  # Very high
            initial_n_e=1e17,
        )
        assert result_high.temperature_eV > 0  # Must be positive
        assert result_high.convergence_status is not None

        # Concentrations should always sum to 1 regardless of starting point
        np.testing.assert_allclose(sum(result_low.concentrations.values()), 1.0, rtol=1e-6)
        np.testing.assert_allclose(sum(result_high.concentrations.values()), 1.0, rtol=1e-6)

    def test_profile_likelihood_concentration(self):
        """Test profile likelihood computation for concentration parameter."""
        elements = ["Fe", "Mg"]
        line_centers = {"Fe": [500.0], "Mg": [510.0]}
        line_strengths = {"Fe": [1.0], "Mg": [1.0]}
        wavelength = np.linspace(490, 520, 100)

        forward_model = create_simple_forward_model(elements, line_centers, line_strengths)
        optimizer = JointOptimizer(forward_model, elements, wavelength)

        # Perfect synthetic spectrum
        measured = forward_model(1.0, 1e17, jnp.array([0.7, 0.3]), wavelength)

        result = optimizer.optimize(
            measured,
            initial_T_eV=1.0,
            initial_n_e=1e17,
            initial_concentrations={"Fe": 0.5, "Mg": 0.5},
            method="BFGS",
        )

        param_values, profile_loss = optimizer.profile_likelihood(
            result, "C_Fe", measured, n_points=5, sigma_range=2.0
        )

        assert len(param_values) == 5
        assert len(profile_loss) == 5
        # The minimum profile loss should be very close to zero at the optimum
        assert np.min(profile_loss) < 1e-3

    def test_profile_likelihood_log_ne(self):
        """Test profile likelihood computation for log_ne parameter."""
        elements = ["Fe"]
        line_centers = {"Fe": [500.0]}
        line_strengths = {"Fe": [1.0]}
        wavelength = np.linspace(490, 510, 100)

        forward_model = create_simple_forward_model(elements, line_centers, line_strengths)
        optimizer = JointOptimizer(forward_model, elements, wavelength)

        measured = forward_model(1.0, 1e17, jnp.array([1.0]), wavelength)

        result = optimizer.optimize(measured, initial_T_eV=1.0, initial_n_e=1e17, method="BFGS")

        param_values, profile_loss = optimizer.profile_likelihood(
            result, "log_ne", measured, n_points=5, sigma_range=2.0
        )

        assert len(param_values) == 5
        assert len(profile_loss) == 5
        # The minimum profile loss should be very close to zero at the optimum
        assert np.min(profile_loss) < 1e-3


class TestFullSoftmaxJacobianUncertainty:
    """Bug (d) audit Family 7: concentration uncertainty must propagate the
    theta-block covariance through the FULL softmax Jacobian
    J_ij = C_i (delta_ij - C_j), not the diagonal-only C_i(1-C_i) sigma_i.
    The diagonal version is 30-50% low on correlated/degenerate compositions."""

    def _sigma_c_full_jacobian(self, theta0, cov_theta):
        """Analytical sigma_C via Sigma_C = J cov_theta J^T (the fixed path)."""
        from cflibs.inversion.physics.softmax_closure import softmax_jacobian

        J = np.array(softmax_jacobian(jnp.array(theta0)))
        cov_C = J @ cov_theta @ J.T
        return np.sqrt(np.abs(np.diag(cov_C)))

    def _sigma_c_diagonal_only(self, theta0, cov_theta):
        """Old diagonal-only approximation C_i(1-C_i) sigma_theta_i."""
        from cflibs.inversion.physics.softmax_closure import softmax_closure

        C = np.array(softmax_closure(jnp.array(theta0)))
        sigma_theta = np.sqrt(np.diag(cov_theta))
        return C * (1.0 - C) * sigma_theta

    def _sigma_c_monte_carlo(self, theta0, cov_theta, n=300000, seed=0):
        """Independent MC oracle: theta ~ N(theta0, cov_theta) -> softmax -> std."""
        from cflibs.inversion.physics.softmax_closure import softmax_closure

        rng = np.random.default_rng(seed)
        samples = rng.multivariate_normal(theta0, cov_theta, size=n)
        C = np.array(softmax_closure(jnp.array(samples)))
        return C.std(axis=0)

    def test_full_jacobian_matches_monte_carlo_degenerate_5element(self):
        """Full Jacobian sigma_C within ~5% of MC on a degenerate 5-element
        composition; diagonal-only is ~30% low on the minor elements.

        For independent theta errors (diagonal cov_theta) the full softmax
        variance for element i is
            Var_full(C_i) = C_i^2 [ (1-C_i)^2 sigma_i^2 + sum_{j!=i} C_j^2 sigma_j^2 ],
        whereas the diagonal-only approximation keeps only the first term:
            Var_diag(C_i) = C_i^2 (1-C_i)^2 sigma_i^2.
        The dropped cross terms make diagonal-only strictly low. The deficit is
        largest for MINOR elements coexisting with a dominant one (the
        cross-term sum_{j!=i} C_j^2 is then ~ C_dominant^2 >> (1-C_i)^2), where
        diagonal-only undershoots the truth by ~30%.
        """
        # Dominant + four minor elements: a degenerate, highly unequal simplex
        # point where the off-diagonal softmax coupling dominates the minor
        # elements' variance.
        C = np.array([0.90, 0.025, 0.025, 0.025, 0.025])
        theta0 = np.log(C)
        theta0 = theta0 - theta0.mean()  # centered representative
        sigma_theta = 0.15
        cov_theta = np.diag(np.full(C.size, sigma_theta**2))

        sigma_full = self._sigma_c_full_jacobian(theta0, cov_theta)
        sigma_diag = self._sigma_c_diagonal_only(theta0, cov_theta)
        sigma_mc = self._sigma_c_monte_carlo(theta0, cov_theta)

        # Full Jacobian agrees with the MC oracle within ~5% for every element.
        np.testing.assert_allclose(sigma_full, sigma_mc, rtol=0.05)

        # Diagonal-only is substantially LOW on the minor elements (~30% low),
        # while the full Jacobian is right: this is the bug the fix corrects.
        minor_ratio = (sigma_diag / sigma_mc)[1:]
        assert np.all(minor_ratio < 0.75), f"diagonal-only not low enough: {minor_ratio}"
        # And full is much closer to MC than diagonal-only is, for the minor set.
        full_err = np.abs(sigma_full - sigma_mc)[1:]
        diag_err = np.abs(sigma_diag - sigma_mc)[1:]
        assert np.all(full_err < diag_err)

    def test_optimizer_reports_full_jacobian_sigma_c(self):
        """End-to-end: JointOptimizer.optimize must report the full-Jacobian
        sigma_C, not the diagonal-only approximation.

        The softmax over-parameterization makes the loss Hessian singular along
        the all-ones theta direction, so the real optimizer's Hessian-based
        uncertainty block is frequently skipped on the toy model. To exercise
        the PRODUCTION sigma_C propagation deterministically, we patch
        jax.hessian to return a fixed, well-conditioned positive-definite matrix
        and assert the optimizer's reported sigma_C equals the full-Jacobian
        propagation of that covariance (and differs from diagonal-only).
        """
        from unittest.mock import patch
        import cflibs.inversion.solve.joint_optimizer as jo

        elements = ["Fe", "Mg", "Ca"]
        line_centers = {"Fe": [500.0], "Mg": [510.0], "Ca": [520.0]}
        line_strengths = {"Fe": [1.0], "Mg": [1.0], "Ca": [1.0]}
        wavelength = np.linspace(480, 540, 150)

        forward_model = create_simple_forward_model(elements, line_centers, line_strengths)
        optimizer = JointOptimizer(forward_model, elements, wavelength, max_iterations=100)

        conc_true = jnp.array([0.5, 0.3, 0.2])
        measured = np.array(forward_model(1.0, 1e17, conc_true, wavelength))

        # n_params = 2 (log_T, log_ne) + 3 elements = 5.
        n = optimizer.n_params
        rng = np.random.default_rng(0)
        A = rng.normal(size=(n, n))
        fixed_hessian = A @ A.T + np.eye(n) * 5.0  # SPD, well conditioned

        def fake_hessian(_fn):
            return lambda _x: jnp.array(fixed_hessian)

        with patch.object(jo.jax, "hessian", fake_hessian):
            result = optimizer.optimize(
                measured,
                initial_T_eV=1.0,
                initial_n_e=1e17,
                initial_concentrations={"Fe": 1 / 3, "Mg": 1 / 3, "Ca": 1 / 3},
                method="BFGS",
            )

        sigma_reported = np.array([result.parameter_uncertainties[f"C_{el}"] for el in elements])
        assert np.all(np.isfinite(sigma_reported))
        assert np.all(sigma_reported > 0.0)

        # Reconstruct the optimizer's covariance exactly as the production code
        # does: cov = inv(H) * reduced_chi2 (or 1.0 if non-positive).
        scale = result.reduced_chi_squared if result.reduced_chi_squared > 0 else 1.0
        cov = np.linalg.inv(fixed_hessian) * scale
        cov_theta = cov[2:, 2:]
        theta0 = np.array(
            optimizer._pack_params(
                result.temperature_eV, result.electron_density_cm3, result.concentrations
            )[2:]
        )

        sigma_full = self._sigma_c_full_jacobian(theta0, cov_theta)
        sigma_diag = self._sigma_c_diagonal_only(theta0, cov_theta)

        # Reported sigma_C equals the full-Jacobian propagation (the fix).
        np.testing.assert_allclose(sigma_reported, sigma_full, rtol=1e-4, atol=1e-9)
        # And it is observably different from the old diagonal-only estimate.
        assert np.any(
            np.abs(sigma_full - sigma_diag) > 0.05 * np.maximum(sigma_full, 1e-12)
        ), "full Jacobian indistinguishable from diagonal-only on this fit"
