import jax
import jax.numpy as jnp
from typing import NamedTuple

class HVRFParams(NamedTuple):
    """
    Parameters for the High-Voltage Radio-Frequency (HVRF) enhancement model.
    
    Attributes:
        enhancement_scale: (xi_0) The maximum scaling factor for intensities.
        decay_constant: (beta) Controls the rate of saturation with power.
        power_threshold: (P_min) The power level below which enhancement is negligible.
    """
    enhancement_scale: float
    decay_constant: float
    power_threshold: float

def apply_hvrf_enhancement(
    intensities: jnp.ndarray,
    applied_power: float,
    params: HVRFParams
) -> jnp.ndarray:
    """
    Applies HVRF enhancement factor to spectral intensities.
    
    The model uses a smooth softplus transition for the power threshold
    to ensure continuity, which improves NUTS sampler convergence.
    
    I_enhanced = I * (1 + xi_0 * exp(-beta / delta_p))
    """
    xi_0 = params.enhancement_scale
    beta = params.decay_constant
    P_min = params.power_threshold
    
    # Use softplus for a smooth threshold transition.
    delta_p = jax.nn.softplus(applied_power - P_min)
    
    # Safe exponentiation to avoid division by zero gradients.
    safe_delta_p = jnp.maximum(1e-10, delta_p)
    factor = 1.0 + xi_0 * jnp.exp(-beta / safe_delta_p)
    
    return intensities * factor
