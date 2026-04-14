"""
Physics-Informed Neural Networks (PINNs) for CF-LIBS analysis.

This module implements neural network architectures with embedded plasma physics
constraints for calibration-free LIBS analysis. The approach bridges data-driven
machine learning with physics-based CF-LIBS to:

- Reduce training data requirements via physics regularization
- Improve extrapolation to new conditions through physical constraints
- Ensure thermodynamic consistency (Boltzmann, Saha equilibrium)
- Provide uncertainty quantification via ensemble methods

Core Components
---------------
1. **PhysicsConstraintLayer**: Encodes physics constraints as differentiable
   penalty terms (Boltzmann equilibrium, Saha ionization, sum-to-one)

2. **DifferentiableForwardModel**: JAX-based forward model that maps plasma
   parameters to synthetic spectra (fully differentiable for backprop)

3. **PINNEncoder**: Neural network encoder with optional physics constraints
   that maps spectra to latent plasma parameters (T, n_e, concentrations)

4. **PINNInverter**: End-to-end PINN-based inversion combining encoder and
   physics-informed loss function

Theory
------
The PINN loss function combines data fit and physics compliance:

    L_total = L_data + lambda_boltz * L_boltzmann + lambda_saha * L_saha
              + lambda_closure * L_closure

Where:
- L_data: MSE between predicted and observed spectra
- L_boltzmann: Penalty for violating Boltzmann distribution
- L_saha: Penalty for violating Saha ionization equilibrium
- L_closure: Penalty for concentrations not summing to 1

References
----------
- Raissi, Perdikaris & Karniadakis (2019): Physics-informed neural networks
- Tognoni et al. (2010): CF-LIBS state of the art
- Lu et al. (2021): DeepXDE for physics-informed deep learning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

from cflibs.core.constants import (
    SAHA_CONST_CM3,
    EV_TO_K,
)
from cflibs.core.logging_config import get_logger

logger = get_logger("inversion.pinn")

# Optional JAX/Equinox imports
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, value_and_grad, vmap
    import jax.random as random

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None

    def jit(f):
        return f

    grad = None
    value_and_grad = None
    vmap = None

# Optional Equinox (JAX neural network library) - lightweight alternative to Flax
try:
    import equinox as eqx

    HAS_EQUINOX = True
except ImportError:
    HAS_EQUINOX = False
    eqx = None

# Optional Optax for optimizers
try:
    import optax

    HAS_OPTAX = True
except ImportError:
    HAS_OPTAX = False
    optax = None


class ConstraintType(Enum):
    """Types of physics constraints for PINNs."""

    BOLTZMANN = "boltzmann"
    SAHA = "saha"
    CLOSURE = "closure"
    ENERGY_CONSERVATION = "energy_conservation"


@dataclass
class PhysicsConstraintConfig:
    """
    Configuration for physics constraints in PINN training.

    Attributes
    ----------
    lambda_boltzmann : float
        Weight for Boltzmann equilibrium constraint (default: 1.0)
    lambda_saha : float
        Weight for Saha ionization balance constraint (default: 1.0)
    lambda_closure : float
        Weight for concentration closure constraint (default: 10.0)
    lambda_energy : float
        Weight for energy conservation constraint (default: 0.1)
    temperature_range_eV : Tuple[float, float]
        Valid temperature range for soft clamping [eV]
    density_range_log : Tuple[float, float]
        Valid log10(n_e) range for soft clamping
    """

    lambda_boltzmann: float = 1.0
    lambda_saha: float = 1.0
    lambda_closure: float = 10.0
    lambda_energy: float = 0.1
    temperature_range_eV: Tuple[float, float] = (0.3, 3.0)
    density_range_log: Tuple[float, float] = (15.0, 19.0)


@dataclass
class PINNConfig:
    """
    Configuration for PINN architecture and training.

    Attributes
    ----------
    encoder_hidden_dims : List[int]
        Hidden layer dimensions for encoder network
    activation : str
        Activation function: 'relu', 'gelu', 'tanh', 'swish'
    dropout_rate : float
        Dropout rate for regularization (default: 0.1)
    use_batch_norm : bool
        Whether to use batch normalization
    n_ensemble : int
        Number of ensemble members for uncertainty (default: 1)
    learning_rate : float
        Initial learning rate (default: 1e-3)
    physics_config : PhysicsConstraintConfig
        Physics constraint configuration
    """

    encoder_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    activation: str = "gelu"
    dropout_rate: float = 0.1
    use_batch_norm: bool = True
    n_ensemble: int = 1
    learning_rate: float = 1e-3
    physics_config: PhysicsConstraintConfig = field(default_factory=PhysicsConstraintConfig)


@dataclass
class PINNResult:
    """
    Result container for PINN-based inversion.

    Attributes
    ----------
    temperature_eV : float
        Recovered temperature in eV
    temperature_uncertainty_eV : float
        Temperature uncertainty (from ensemble)
    electron_density_cm3 : float
        Recovered electron density in cm^-3
    density_uncertainty_cm3 : float
        Density uncertainty (from ensemble)
    concentrations : Dict[str, float]
        Element concentrations
    concentration_uncertainties : Dict[str, float]
        Concentration uncertainties
    physics_loss : Dict[str, float]
        Individual physics constraint losses
    total_loss : float
        Final total loss value
    converged : bool
        Whether training converged
    epochs : int
        Number of training epochs
    """

    temperature_eV: float
    temperature_uncertainty_eV: float
    electron_density_cm3: float
    density_uncertainty_cm3: float
    concentrations: Dict[str, float]
    concentration_uncertainties: Dict[str, float]
    physics_loss: Dict[str, float]
    total_loss: float
    converged: bool
    epochs: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def temperature_K(self) -> float:
        """Temperature in Kelvin."""
        return self.temperature_eV * EV_TO_K

    @property
    def log_ne(self) -> float:
        """Log10 of electron density."""
        return np.log10(self.electron_density_cm3)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "PINN Inversion Result",
            "=" * 50,
            f"Temperature: {self.temperature_eV:.4f} +/- {self.temperature_uncertainty_eV:.4f} eV",
            f"            ({self.temperature_K:.0f} K)",
            (
                f"Electron density: {self.electron_density_cm3:.2e} +/- "
                f"{self.density_uncertainty_cm3:.2e} cm^-3"
            ),
            "",
            "Concentrations:",
        ]
        for el, c in self.concentrations.items():
            unc = self.concentration_uncertainties.get(el, 0.0)
            lines.append(f"  {el}: {c:.4f} +/- {unc:.4f}")

        lines.extend(
            [
                "",
                "Physics Constraint Losses:",
            ]
        )
        for name, loss in self.physics_loss.items():
            lines.append(f"  {name}: {loss:.4e}")

        lines.extend(
            [
                "",
                f"Total loss: {self.total_loss:.4e}",
                f"Converged: {self.converged} ({self.epochs} epochs)",
            ]
        )
        return "\n".join(lines)


# ===========================================================================
# Physics Constraint Functions (JAX-compatible)
# ===========================================================================


def boltzmann_residual(
    T_eV: float,
    energies_eV: Any,
    populations: Any,
    degeneracies: Any,
) -> Any:
    """
    Compute residual from Boltzmann distribution.

    The Boltzmann distribution requires:
        n_k / n_j = (g_k / g_j) * exp(-(E_k - E_j) / kT)

    This function computes the deviation from this equilibrium.

    Parameters
    ----------
    T_eV : float
        Temperature in eV
    energies_eV : array
        Energy levels [eV]
    populations : array
        Population densities (unnormalized)
    degeneracies : array
        Statistical weights (g values)

    Returns
    -------
    float
        Mean squared residual from Boltzmann equilibrium
    """
    if not HAS_JAX:
        raise ImportError("JAX required for physics constraints")

    # Reference level (ground state or first level)
    E_ref = energies_eV[0]
    g_ref = degeneracies[0]
    n_ref = populations[0]

    # Expected populations from Boltzmann
    expected = (
        n_ref * (degeneracies / g_ref) * jnp.exp(-(energies_eV - E_ref) / jnp.maximum(T_eV, 0.1))
    )

    # Safe comparison avoiding division by zero
    expected_safe = jnp.maximum(expected, 1e-30)
    populations_safe = jnp.maximum(populations, 1e-30)

    # Log-space comparison for numerical stability
    log_ratio = jnp.log(populations_safe) - jnp.log(expected_safe)

    return jnp.mean(log_ratio**2)


def saha_residual(
    T_eV: float,
    n_e_cm3: float,
    n_neutral: Any,
    n_ion: Any,
    U_neutral: Any,
    U_ion: Any,
    IP_eV: Any,
) -> Any:
    """
    Compute residual from Saha ionization equilibrium.

    The Saha equation requires:
        n_ion * n_e / n_neutral = (2 * U_ion / U_neutral) * S(T)

    where S(T) = (SAHA_CONST / n_e) * T^1.5 * exp(-IP / kT)

    Parameters
    ----------
    T_eV : float
        Temperature in eV
    n_e_cm3 : float
        Electron density [cm^-3]
    n_neutral : array
        Neutral species densities
    n_ion : array
        Ionic species densities
    U_neutral : array
        Partition functions for neutrals
    U_ion : array
        Partition functions for ions
    IP_eV : array
        Ionization potentials [eV]

    Returns
    -------
    float
        Mean squared residual from Saha equilibrium
    """
    if not HAS_JAX:
        raise ImportError("JAX required for physics constraints")

    # Saha factor
    T_safe = jnp.maximum(T_eV, 0.1)
    saha_factor = (SAHA_CONST_CM3 / n_e_cm3) * (T_safe**1.5)

    # Expected ratio
    expected_ratio = (U_ion / jnp.maximum(U_neutral, 1.0)) * saha_factor * jnp.exp(-IP_eV / T_safe)

    # Actual ratio
    actual_ratio = (n_ion * n_e_cm3) / jnp.maximum(n_neutral, 1e-30)

    # Log-space residual
    log_expected = jnp.log(jnp.maximum(expected_ratio, 1e-30))
    log_actual = jnp.log(jnp.maximum(actual_ratio, 1e-30))

    return jnp.mean((log_actual - log_expected) ** 2)


def closure_residual(concentrations: Any) -> Any:
    """
    Compute residual from concentration closure (sum to 1).

    Parameters
    ----------
    concentrations : array
        Element concentrations

    Returns
    -------
    float
        Squared deviation from sum = 1
    """
    if not HAS_JAX:
        raise ImportError("JAX required for physics constraints")

    return (jnp.sum(concentrations) - 1.0) ** 2


def positivity_penalty(values: Any, eps: float = 1e-6) -> Any:
    """
    Soft penalty for non-positive values.

    Uses a ReLU-like penalty: max(0, -x)^2

    Parameters
    ----------
    values : array
        Values that should be positive
    eps : float
        Small positive value for numerical stability

    Returns
    -------
    float
        Sum of squared penalties for negative values
    """
    if not HAS_JAX:
        raise ImportError("JAX required for physics constraints")

    return jnp.sum(jnp.maximum(-values + eps, 0.0) ** 2)


def range_penalty(value: float, low: float, high: float, sharpness: float = 10.0) -> Any:
    """
    Soft penalty for values outside a valid range.

    Uses smooth sigmoid-like penalty to avoid discontinuities.

    Parameters
    ----------
    value : float
        Value to constrain
    low : float
        Lower bound
    high : float
        Upper bound
    sharpness : float
        Penalty sharpness (higher = sharper transition)

    Returns
    -------
    float
        Penalty value (0 inside range, increasing outside)
    """
    if not HAS_JAX:
        raise ImportError("JAX required for physics constraints")

    # Penalty for being below low
    below_penalty = jax.nn.softplus(sharpness * (low - value))
    # Penalty for being above high
    above_penalty = jax.nn.softplus(sharpness * (value - high))

    return below_penalty + above_penalty


# ===========================================================================
# Neural Network Components (Equinox-based)
# ===========================================================================


if HAS_JAX and HAS_EQUINOX:

    class PhysicsConstraintLayer(eqx.Module):
        """
        Layer that computes physics constraint losses.

        This layer wraps the physics constraint functions and provides a unified
        interface for computing all constraint violations.

        Attributes
        ----------
        config : PhysicsConstraintConfig
            Constraint weights and ranges
        n_elements : int
            Number of elements
        ionization_potentials : jnp.ndarray
            IP values for Saha calculation [eV]
        """

        config: PhysicsConstraintConfig = eqx.field(static=True)
        n_elements: int = eqx.field(static=True)
        ionization_potentials: jnp.ndarray

        def __init__(
            self,
            n_elements: int,
            ionization_potentials: np.ndarray,
            config: PhysicsConstraintConfig = PhysicsConstraintConfig(),
        ):
            self.config = config
            self.n_elements = n_elements
            self.ionization_potentials = jnp.array(ionization_potentials)

        def __call__(
            self,
            T_eV: float,
            log_ne: float,
            concentrations: jnp.ndarray,
            partition_funcs_neutral: Optional[jnp.ndarray] = None,
            partition_funcs_ion: Optional[jnp.ndarray] = None,
        ) -> Dict[str, float]:
            """
            Compute all physics constraint losses.

            Parameters
            ----------
            T_eV : float
                Temperature in eV
            log_ne : float
                Log10 of electron density
            concentrations : array
                Element concentrations
            partition_funcs_neutral : array, optional
                Partition functions for neutral species
            partition_funcs_ion : array, optional
                Partition functions for ionic species

            Returns
            -------
            dict
                Dictionary of constraint losses
            """
            n_e = 10.0**log_ne
            losses = {}

            # Closure constraint (concentrations sum to 1)
            losses["closure"] = self.config.lambda_closure * closure_residual(concentrations)

            # Positivity constraint (all concentrations >= 0)
            losses["positivity"] = positivity_penalty(concentrations)

            # Range constraints for temperature and density
            losses["T_range"] = range_penalty(
                T_eV,
                self.config.temperature_range_eV[0],
                self.config.temperature_range_eV[1],
            )
            losses["ne_range"] = range_penalty(
                log_ne,
                self.config.density_range_log[0],
                self.config.density_range_log[1],
            )

            # Saha constraint (if partition functions provided)
            if partition_funcs_neutral is not None and partition_funcs_ion is not None:
                # For Saha, we need neutral and ion populations
                # Approximate from concentrations and Saha equation itself
                T_safe = jnp.maximum(T_eV, 0.1)
                saha_factor = (SAHA_CONST_CM3 / n_e) * (T_safe**1.5)
                saha_ratios = (
                    (partition_funcs_ion / jnp.maximum(partition_funcs_neutral, 1.0))
                    * saha_factor
                    * jnp.exp(-self.ionization_potentials / T_safe)
                )

                # Saha ratio should be physically reasonable
                # Penalize extreme ratios that would give unrealistic ionization
                log_saha = jnp.log10(jnp.maximum(saha_ratios, 1e-30))
                # Typical LIBS: Saha ratio between 0.01 and 100
                saha_penalty = jnp.mean(jnp.maximum(jnp.abs(log_saha) - 2.0, 0.0) ** 2)
                losses["saha"] = self.config.lambda_saha * saha_penalty

            return losses

        def total_loss(
            self,
            T_eV: float,
            log_ne: float,
            concentrations: jnp.ndarray,
            partition_funcs_neutral: Optional[jnp.ndarray] = None,
            partition_funcs_ion: Optional[jnp.ndarray] = None,
        ) -> float:
            """Compute sum of all constraint losses."""
            losses = self(
                T_eV, log_ne, concentrations, partition_funcs_neutral, partition_funcs_ion
            )
            return sum(losses.values())

    class MLP(eqx.Module):
        """
        Multi-layer perceptron with configurable architecture.

        Supports GELU, ReLU, Tanh, and Swish activations.
        Optionally includes batch normalization and dropout.
        """

        layers: List[eqx.nn.Linear]
        activation_fn: Callable = eqx.field(static=True)
        dropout_rate: float = eqx.field(static=True)

        def __init__(
            self,
            in_features: int,
            hidden_dims: List[int],
            out_features: int,
            activation: str = "gelu",
            dropout_rate: float = 0.0,
            *,
            key: jax.Array,
        ):
            """
            Initialize MLP.

            Parameters
            ----------
            in_features : int
                Input dimension
            hidden_dims : list
                Hidden layer dimensions
            out_features : int
                Output dimension
            activation : str
                Activation function name
            dropout_rate : float
                Dropout probability
            key : jax.Array
                Random key
            """
            # Build layer dimensions
            dims = [in_features] + list(hidden_dims) + [out_features]
            keys = random.split(key, len(dims) - 1)

            self.layers = [
                eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]) for i in range(len(dims) - 1)
            ]

            # Set activation function
            activations = {
                "relu": jax.nn.relu,
                "gelu": jax.nn.gelu,
                "tanh": jnp.tanh,
                "swish": jax.nn.swish,
                "silu": jax.nn.silu,
            }
            self.activation_fn = activations.get(activation.lower(), jax.nn.gelu)
            self.dropout_rate = dropout_rate

        def __call__(self, x: jnp.ndarray, *, key: Optional[jax.Array] = None) -> jnp.ndarray:
            """
            Forward pass through MLP.

            Parameters
            ----------
            x : array
                Input tensor
            key : jax.Array, optional
                Random key for dropout (inference if None)

            Returns
            -------
            array
                Output tensor
            """
            for i, layer in enumerate(self.layers[:-1]):
                x = layer(x)
                x = self.activation_fn(x)

                # Apply dropout if training (key provided)
                if key is not None and self.dropout_rate > 0:
                    key, subkey = random.split(key)
                    mask = random.bernoulli(subkey, 1.0 - self.dropout_rate, x.shape)
                    x = x * mask / (1.0 - self.dropout_rate)

            # Final layer (no activation)
            x = self.layers[-1](x)
            return x

    class PINNEncoder(eqx.Module):
        """
        Physics-informed encoder network for LIBS spectra.

        Maps observed spectra to plasma parameters (T, n_e, concentrations)
        with built-in physics constraints.

        Architecture:
        - Shared feature extractor (MLP)
        - Separate heads for T, log_ne, and concentrations
        - Softmax output for concentrations (ensures sum to 1)
        - Softplus outputs for T and n_e (ensures positivity)
        """

        feature_extractor: MLP
        T_head: eqx.nn.Linear
        ne_head: eqx.nn.Linear
        conc_head: eqx.nn.Linear
        n_elements: int = eqx.field(static=True)
        T_range: Tuple[float, float] = eqx.field(static=True)
        ne_range: Tuple[float, float] = eqx.field(static=True)

        def __init__(
            self,
            n_wavelengths: int,
            n_elements: int,
            hidden_dims: List[int] = [256, 128, 64],
            activation: str = "gelu",
            dropout_rate: float = 0.1,
            T_range: Tuple[float, float] = (0.3, 3.0),
            ne_range: Tuple[float, float] = (15.0, 19.0),
            *,
            key: jax.Array,
        ):
            """
            Initialize PINN encoder.

            Parameters
            ----------
            n_wavelengths : int
                Number of wavelength channels in input spectrum
            n_elements : int
                Number of elements to predict concentrations for
            hidden_dims : list
                Hidden layer dimensions
            activation : str
                Activation function
            dropout_rate : float
                Dropout rate
            T_range : tuple
                Valid temperature range [eV] for output scaling
            ne_range : tuple
                Valid log10(n_e) range for output scaling
            key : jax.Array
                Random key
            """
            keys = random.split(key, 4)

            self.n_elements = n_elements
            self.T_range = T_range
            self.ne_range = ne_range

            # Shared feature extractor
            self.feature_extractor = MLP(
                n_wavelengths,
                hidden_dims[:-1],
                hidden_dims[-1],
                activation=activation,
                dropout_rate=dropout_rate,
                key=keys[0],
            )

            # Output heads
            self.T_head = eqx.nn.Linear(hidden_dims[-1], 1, key=keys[1])
            self.ne_head = eqx.nn.Linear(hidden_dims[-1], 1, key=keys[2])
            self.conc_head = eqx.nn.Linear(hidden_dims[-1], n_elements, key=keys[3])

        def __call__(
            self, spectrum: jnp.ndarray, *, key: Optional[jax.Array] = None
        ) -> Tuple[float, float, jnp.ndarray]:
            """
            Encode spectrum to plasma parameters.

            Parameters
            ----------
            spectrum : array
                Observed spectrum (n_wavelengths,)
            key : jax.Array, optional
                Random key for dropout

            Returns
            -------
            T_eV : float
                Temperature in eV
            log_ne : float
                Log10 of electron density
            concentrations : array
                Element concentrations (sum to 1)
            """
            # Extract features
            features = self.feature_extractor(spectrum, key=key)

            # Temperature: sigmoid scaled to valid range
            T_logit = self.T_head(features).squeeze()
            T_eV = self.T_range[0] + jax.nn.sigmoid(T_logit) * (self.T_range[1] - self.T_range[0])

            # Electron density: sigmoid scaled to valid range
            ne_logit = self.ne_head(features).squeeze()
            log_ne = self.ne_range[0] + jax.nn.sigmoid(ne_logit) * (
                self.ne_range[1] - self.ne_range[0]
            )

            # Concentrations: softmax ensures sum to 1 and all positive
            conc_logits = self.conc_head(features)
            concentrations = jax.nn.softmax(conc_logits)

            return T_eV, log_ne, concentrations


class DifferentiableForwardModel:
    """
    Differentiable forward model for LIBS spectra.

    This is a JAX-compatible forward model that computes synthetic spectra
    from plasma parameters. It can be used both for training PINNs and for
    physics-based regularization.

    Uses simplified physics (Gaussian line profiles) for computational
    efficiency during training. For high-fidelity spectra, use the
    BayesianForwardModel from the bayesian module.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength grid [nm]
    elements : List[str]
        Element names
    line_positions : np.ndarray
        Central wavelengths of spectral lines [nm]
    line_gA : np.ndarray
        g*A values for lines [s^-1]
    line_Ek : np.ndarray
        Upper level energies [eV]
    line_element_idx : np.ndarray
        Element index for each line
    instrument_fwhm : float
        Instrument FWHM [nm]
    """

    def __init__(
        self,
        wavelength: np.ndarray,
        elements: List[str],
        line_positions: np.ndarray,
        line_gA: np.ndarray,
        line_Ek: np.ndarray,
        line_element_idx: np.ndarray,
        instrument_fwhm: float = 0.05,
    ):
        if not HAS_JAX:
            raise ImportError("JAX required for DifferentiableForwardModel")

        self.wavelength = jnp.array(wavelength)
        self.elements = elements
        self.n_elements = len(elements)

        # Line data
        self.line_positions = jnp.array(line_positions)
        self.line_gA = jnp.array(line_gA)
        self.line_Ek = jnp.array(line_Ek)
        self.line_element_idx = jnp.array(line_element_idx, dtype=jnp.int32)
        self.instrument_fwhm = instrument_fwhm

        logger.info(
            f"DifferentiableForwardModel: {len(elements)} elements, "
            f"{len(line_positions)} lines, {len(wavelength)} wavelengths"
        )

    @staticmethod
    @jit
    def _compute_spectrum(
        wavelength: jnp.ndarray,
        T_eV: float,
        log_ne: float,
        concentrations: jnp.ndarray,
        line_positions: jnp.ndarray,
        line_gA: jnp.ndarray,
        line_Ek: jnp.ndarray,
        line_element_idx: jnp.ndarray,
        sigma_inst: float,
    ) -> jnp.ndarray:
        """
        JIT-compiled spectrum computation.

        Uses Boltzmann factors for line intensities and Gaussian profiles
        with Doppler, instrument, and Stark broadening contributions.

        Parameters
        ----------
        wavelength : array
            Wavelength grid [nm]
        T_eV : float
            Temperature in eV
        log_ne : float
            Log10 of electron density [cm^-3]
        concentrations : array
            Element concentrations
        line_positions : array
            Central wavelengths [nm]
        line_gA : array
            g*A values [s^-1]
        line_Ek : array
            Upper level energies [eV]
        line_element_idx : array
            Element index per line
        sigma_inst : float
            Instrument Gaussian sigma [nm]
        """
        n_e = 10.0**log_ne

        # Temperature-dependent Gaussian width (simplified Doppler)
        sigma_doppler = 0.01 * jnp.sqrt(T_eV / 0.86)  # Approximate scaling

        # Stark broadening: FWHM_stark ~ w * (n_e / 1e16) [nm]
        # Typical Stark width parameter w ~ 0.01-0.1 nm at n_e=1e16 cm^-3
        # for visible LIBS lines. We use a representative value.
        # (CF-LIBS-improved-3sf fix): without this term the synthetic
        # spectrum is invariant to n_e, breaking physics regularization.
        stark_w_param = 0.02  # representative Stark width parameter [nm]
        fwhm_stark = stark_w_param * (n_e / 1e16)
        sigma_stark = fwhm_stark / 2.355  # convert FWHM to Gaussian sigma

        sigma_total = jnp.sqrt(sigma_inst**2 + sigma_doppler**2 + sigma_stark**2)

        # Line intensities from Boltzmann
        boltzmann_factor = jnp.exp(-line_Ek / jnp.maximum(T_eV, 0.1))
        line_intensity = line_gA * boltzmann_factor

        # Scale by concentrations
        element_conc = concentrations[line_element_idx]
        line_intensity = line_intensity * element_conc

        # Compute Gaussian profiles and sum
        diff = wavelength[:, None] - line_positions[None, :]
        profiles = jnp.exp(-0.5 * (diff / sigma_total) ** 2) / (sigma_total * jnp.sqrt(2 * jnp.pi))

        spectrum = jnp.sum(line_intensity * profiles, axis=1)
        return spectrum

    def forward(
        self,
        T_eV: float,
        log_ne: float,
        concentrations: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute synthetic spectrum.

        Parameters
        ----------
        T_eV : float
            Temperature in eV
        log_ne : float
            Log10 of electron density (affects Stark broadening widths)
        concentrations : array
            Element concentrations

        Returns
        -------
        array
            Synthetic spectrum
        """
        sigma_inst = self.instrument_fwhm / 2.355
        return self._compute_spectrum(
            self.wavelength,
            T_eV,
            log_ne,
            concentrations,
            self.line_positions,
            self.line_gA,
            self.line_Ek,
            self.line_element_idx,
            sigma_inst,
        )


class PINNInverter:
    """
    Physics-Informed Neural Network inverter for CF-LIBS.

    This class provides end-to-end PINN-based spectrum inversion combining:
    - Neural network encoder for fast inference
    - Physics constraint layers for regularization
    - Optional differentiable forward model for data augmentation
    - Ensemble methods for uncertainty quantification

    Training minimizes:
        L = L_reconstruction + sum(lambda_i * L_physics_i)

    Where L_reconstruction is the MSE between predicted and observed spectra,
    and L_physics_i are the physics constraint losses.

    Parameters
    ----------
    n_wavelengths : int
        Number of wavelength channels
    elements : List[str]
        Element names
    ionization_potentials : np.ndarray
        Ionization potentials for each element [eV]
    config : PINNConfig
        Network and training configuration
    forward_model : DifferentiableForwardModel, optional
        Forward model for physics-based loss
    seed : int
        Random seed

    Example
    -------
    >>> inverter = PINNInverter(
    ...     n_wavelengths=2048,
    ...     elements=["Fe", "Cu"],
    ...     ionization_potentials=np.array([7.87, 7.73]),
    ... )
    >>> # Train on synthetic data
    >>> inverter.train(
    ...     spectra_train, T_train, ne_train, conc_train,
    ...     epochs=1000, batch_size=32
    ... )
    >>> # Inference
    >>> result = inverter.invert(observed_spectrum)
    >>> print(result.summary())
    """

    def __init__(
        self,
        n_wavelengths: int,
        elements: List[str],
        ionization_potentials: np.ndarray,
        config: Optional[PINNConfig] = None,
        forward_model: Optional[DifferentiableForwardModel] = None,
        seed: int = 42,
        partition_funcs_neutral: Optional[np.ndarray] = None,
        partition_funcs_ion: Optional[np.ndarray] = None,
    ):
        if not HAS_JAX:
            raise ImportError("JAX required for PINNInverter")
        if not HAS_EQUINOX:
            raise ImportError(
                "Equinox required for PINNInverter. Install with: pip install equinox"
            )
        if not HAS_OPTAX:
            raise ImportError("Optax required for PINNInverter. Install with: pip install optax")
        if config is None:
            config = PINNConfig()

        self.n_wavelengths = n_wavelengths
        self.elements = elements
        self.n_elements = len(elements)
        self.config = config
        self.forward_model = forward_model

        # Store partition functions for Saha constraint in physics layer
        # (CF-LIBS-improved-9o3): Without these, the Saha branch in
        # PhysicsConstraintLayer.__call__ is never entered and the physics
        # loss degenerates to pure data fitting.
        self.partition_funcs_neutral = (
            jnp.array(partition_funcs_neutral) if partition_funcs_neutral is not None else None
        )
        self.partition_funcs_ion = (
            jnp.array(partition_funcs_ion) if partition_funcs_ion is not None else None
        )

        # Initialize random key
        key = random.PRNGKey(seed)

        # Create encoder(s)
        keys = random.split(key, config.n_ensemble)
        self.encoders = [
            PINNEncoder(
                n_wavelengths=n_wavelengths,
                n_elements=self.n_elements,
                hidden_dims=config.encoder_hidden_dims,
                activation=config.activation,
                dropout_rate=config.dropout_rate,
                T_range=config.physics_config.temperature_range_eV,
                ne_range=config.physics_config.density_range_log,
                key=keys[i],
            )
            for i in range(config.n_ensemble)
        ]

        # Physics constraint layer
        self.physics_layer = PhysicsConstraintLayer(
            n_elements=self.n_elements,
            ionization_potentials=ionization_potentials,
            config=config.physics_config,
        )

        # Optimizer
        self.optimizer = optax.adam(config.learning_rate)
        self.opt_states = [
            self.optimizer.init(eqx.filter(enc, eqx.is_array)) for enc in self.encoders
        ]

        # Training state
        self.trained = False
        self.train_history: Dict[str, List[float]] = {}

        logger.info(
            f"PINNInverter initialized: {n_wavelengths} wavelengths, "
            f"{self.n_elements} elements, {config.n_ensemble} ensemble members"
        )

    def _loss_fn(
        self,
        encoder: "PINNEncoder",
        spectrum: jnp.ndarray,
        T_true: float,
        log_ne_true: float,
        conc_true: jnp.ndarray,
        key: jax.Array,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute total loss for training.

        Returns total loss and dictionary of component losses.
        """
        # Forward pass through encoder
        T_pred, log_ne_pred, conc_pred = encoder(spectrum, key=key)

        # Data loss (supervised)
        loss_T = (T_pred - T_true) ** 2
        loss_ne = (log_ne_pred - log_ne_true) ** 2
        loss_conc = jnp.mean((conc_pred - conc_true) ** 2)

        data_loss = loss_T + loss_ne + 10.0 * loss_conc

        # Physics constraint losses
        # Pass partition functions so the Saha constraint is evaluated
        # (CF-LIBS-improved-9o3 fix)
        physics_losses = self.physics_layer(
            T_pred,
            log_ne_pred,
            conc_pred,
            partition_funcs_neutral=self.partition_funcs_neutral,
            partition_funcs_ion=self.partition_funcs_ion,
        )
        physics_loss = sum(physics_losses.values())

        total_loss = data_loss + physics_loss

        losses = {
            "total": total_loss,
            "data": data_loss,
            "T": loss_T,
            "ne": loss_ne,
            "conc": loss_conc,
            **physics_losses,
        }

        return total_loss, losses

    def _train_step(
        self,
        encoder: "PINNEncoder",
        opt_state: Any,
        spectrum: jnp.ndarray,
        T_true: float,
        log_ne_true: float,
        conc_true: jnp.ndarray,
        key: jax.Array,
    ) -> Tuple["PINNEncoder", Any, Dict[str, float]]:
        """Single training step."""

        def loss_fn(enc):
            return self._loss_fn(enc, spectrum, T_true, log_ne_true, conc_true, key)[0]

        loss, grads = value_and_grad(loss_fn)(encoder)
        updates, new_opt_state = self.optimizer.update(
            grads, opt_state, eqx.filter(encoder, eqx.is_array)
        )
        new_encoder = eqx.apply_updates(encoder, updates)

        # Get full losses for logging
        _, losses = self._loss_fn(new_encoder, spectrum, T_true, log_ne_true, conc_true, key)

        return new_encoder, new_opt_state, losses

    def train(
        self,
        spectra: np.ndarray,
        temperatures: np.ndarray,
        log_densities: np.ndarray,
        concentrations: np.ndarray,
        epochs: int = 1000,
        batch_size: int = 32,
        validation_split: float = 0.1,
        early_stopping_patience: int = 50,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the PINN encoder on labeled data.

        Parameters
        ----------
        spectra : np.ndarray
            Training spectra, shape (n_samples, n_wavelengths)
        temperatures : np.ndarray
            True temperatures [eV], shape (n_samples,)
        log_densities : np.ndarray
            True log10(n_e), shape (n_samples,)
        concentrations : np.ndarray
            True concentrations, shape (n_samples, n_elements)
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        validation_split : float
            Fraction of data for validation
        early_stopping_patience : int
            Epochs without improvement before stopping
        verbose : bool
            Print progress

        Returns
        -------
        dict
            Training history with loss values
        """
        n_samples = spectra.shape[0]
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val

        # Split data
        idx = np.random.permutation(n_samples)
        train_idx, val_idx = idx[:n_train], idx[n_train:]

        X_train = jnp.array(spectra[train_idx])
        T_train = jnp.array(temperatures[train_idx])
        ne_train = jnp.array(log_densities[train_idx])
        C_train = jnp.array(concentrations[train_idx])

        X_val = jnp.array(spectra[val_idx])
        T_val = jnp.array(temperatures[val_idx])
        ne_val = jnp.array(log_densities[val_idx])
        C_val = jnp.array(concentrations[val_idx])

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_T_loss": [],
            "train_physics_loss": [],
        }

        key = random.PRNGKey(42)
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            key, *batch_keys = random.split(key, n_train // batch_size + 2)
            epoch_losses = []

            # Shuffle training data
            perm = np.random.permutation(n_train)

            for i in range(0, n_train, batch_size):
                batch_idx = perm[i : i + batch_size]
                if len(batch_idx) == 0:
                    continue

                # Train each ensemble member
                for j in range(len(self.encoders)):
                    k = batch_keys[i // batch_size % len(batch_keys)]

                    # Average over batch
                    batch_loss = 0.0
                    for b in batch_idx:
                        self.encoders[j], self.opt_states[j], losses = self._train_step(
                            self.encoders[j],
                            self.opt_states[j],
                            X_train[b],
                            T_train[b],
                            ne_train[b],
                            C_train[b],
                            k,
                        )
                        batch_loss += float(losses["total"])

                    epoch_losses.append(batch_loss / len(batch_idx))

            # Compute validation loss
            val_losses = []
            for i in range(n_val):
                _, losses = self._loss_fn(
                    self.encoders[0], X_val[i], T_val[i], ne_val[i], C_val[i], key
                )
                val_losses.append(float(losses["total"]))

            train_loss = np.mean(epoch_losses)
            val_loss = np.mean(val_losses)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch}")
                break

            if verbose and epoch % 100 == 0:
                logger.info(f"Epoch {epoch}: train_loss={train_loss:.4e}, val_loss={val_loss:.4e}")

        self.trained = True
        self.train_history = history
        return history

    def invert(
        self,
        spectrum: np.ndarray,
        return_ensemble: bool = False,
    ) -> PINNResult:
        """
        Invert observed spectrum to plasma parameters.

        Parameters
        ----------
        spectrum : np.ndarray
            Observed spectrum, shape (n_wavelengths,)
        return_ensemble : bool
            Include ensemble predictions in metadata

        Returns
        -------
        PINNResult
            Inversion results with uncertainties
        """
        if not self.trained:
            logger.warning("Encoder not trained, results may be unreliable")

        spectrum_jax = jnp.array(spectrum)

        # Get predictions from all ensemble members
        T_preds = []
        ne_preds = []
        conc_preds = []

        for encoder in self.encoders:
            T, log_ne, conc = encoder(spectrum_jax, key=None)  # No dropout for inference
            T_preds.append(float(T))
            ne_preds.append(float(log_ne))
            conc_preds.append(np.array(conc))

        # Compute ensemble statistics
        T_mean = np.mean(T_preds)
        T_std = np.std(T_preds) if len(T_preds) > 1 else 0.0

        log_ne_mean = np.mean(ne_preds)
        log_ne_std = np.std(ne_preds) if len(ne_preds) > 1 else 0.0

        conc_mean = np.mean(conc_preds, axis=0)
        conc_std = np.std(conc_preds, axis=0) if len(conc_preds) > 1 else np.zeros(self.n_elements)

        # Compute physics losses (pass partition functions for Saha constraint)
        physics_losses = self.physics_layer(
            T_mean,
            log_ne_mean,
            jnp.array(conc_mean),
            partition_funcs_neutral=self.partition_funcs_neutral,
            partition_funcs_ion=self.partition_funcs_ion,
        )
        total_loss = sum(physics_losses.values())

        concentrations = {el: float(conc_mean[i]) for i, el in enumerate(self.elements)}
        concentration_uncertainties = {el: float(conc_std[i]) for i, el in enumerate(self.elements)}

        metadata = {}
        if return_ensemble:
            metadata["T_ensemble"] = T_preds
            metadata["log_ne_ensemble"] = ne_preds
            metadata["conc_ensemble"] = conc_preds

        return PINNResult(
            temperature_eV=T_mean,
            temperature_uncertainty_eV=T_std,
            electron_density_cm3=10.0**log_ne_mean,
            density_uncertainty_cm3=10.0**log_ne_mean * np.log(10) * log_ne_std,
            concentrations=concentrations,
            concentration_uncertainties=concentration_uncertainties,
            physics_loss={k: float(v) for k, v in physics_losses.items()},
            total_loss=float(total_loss),
            converged=self.trained,
            epochs=len(self.train_history.get("train_loss", [])),
            metadata=metadata,
        )


# ===========================================================================
# Convenience Functions
# ===========================================================================


def create_pinn_from_database(
    db_path: str,
    elements: List[str],
    wavelength_range: Tuple[float, float],
    n_wavelengths: int = 2048,
    config: Optional[PINNConfig] = None,
    seed: int = 42,
) -> PINNInverter:
    """
    Create a PINNInverter from an atomic database.

    Convenience function that loads atomic data and initializes the PINN.

    Parameters
    ----------
    db_path : str
        Path to atomic database
    elements : list
        Element names
    wavelength_range : tuple
        (min_nm, max_nm) wavelength range
    n_wavelengths : int
        Number of wavelength points
    config : PINNConfig, optional
        PINN configuration
    seed : int
        Random seed

    Returns
    -------
    PINNInverter
        Initialized (untrained) PINN inverter
    """
    import sqlite3

    conn = sqlite3.connect(db_path)

    # Get ionization potentials
    ips = []
    if elements:
        placeholders = ",".join(["?"] * len(elements))
        query = f"SELECT element, ip_ev FROM species_physics WHERE sp_num=1 AND element IN ({placeholders})"
        cursor = conn.execute(query, tuple(elements))

        # Map results
        results = dict(cursor.fetchall())

        # Maintain order
        for el in elements:
            if el in results:
                ips.append(results[el])
            else:
                logger.warning(f"No IP for {el}, using default 7.0 eV")
                ips.append(7.0)

    conn.close()

    if config is None:
        config = PINNConfig()

    return PINNInverter(
        n_wavelengths=n_wavelengths,
        elements=elements,
        ionization_potentials=np.array(ips),
        config=config,
        seed=seed,
    )


def generate_synthetic_training_data(
    forward_model: DifferentiableForwardModel,
    n_samples: int,
    T_range: Tuple[float, float] = (0.5, 2.0),
    log_ne_range: Tuple[float, float] = (16.0, 18.0),
    noise_level: float = 0.05,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic training data for PINN.

    Creates spectra with known ground truth for supervised training.

    Parameters
    ----------
    forward_model : DifferentiableForwardModel
        Forward model for spectrum generation
    n_samples : int
        Number of samples to generate
    T_range : tuple
        Temperature range [eV]
    log_ne_range : tuple
        Log10(n_e) range
    noise_level : float
        Relative noise level
    seed : int
        Random seed

    Returns
    -------
    spectra : np.ndarray
        Generated spectra (n_samples, n_wavelengths)
    temperatures : np.ndarray
        True temperatures [eV]
    log_densities : np.ndarray
        True log10(n_e)
    concentrations : np.ndarray
        True concentrations (n_samples, n_elements)
    """
    rng = np.random.default_rng(seed)
    n_elements = forward_model.n_elements

    # Generate random parameters
    temperatures = rng.uniform(T_range[0], T_range[1], n_samples)
    log_densities = rng.uniform(log_ne_range[0], log_ne_range[1], n_samples)

    # Random concentrations (Dirichlet)
    concentrations = rng.dirichlet(np.ones(n_elements), n_samples)

    # Generate spectra
    spectra = []
    for i in range(n_samples):
        spectrum = forward_model.forward(
            temperatures[i],
            log_densities[i],
            jnp.array(concentrations[i]),
        )
        # Add noise
        noise = rng.normal(0, noise_level * np.mean(spectrum), len(spectrum))
        spectrum = np.array(spectrum) + noise
        spectrum = np.maximum(spectrum, 0)  # Ensure non-negative
        spectra.append(spectrum)

    return np.array(spectra), temperatures, log_densities, concentrations
