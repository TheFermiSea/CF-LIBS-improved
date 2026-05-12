import jax
import jax.numpy as jnp
from jax import lax
from typing import Optional, Any

def compute_saha_boltzmann(temperature: float, electron_density: float, atomic_data: Any) -> jnp.ndarray:
    """
    Computes ionization and excitation populations based on Saha-Boltzmann equilibrium.
    This is constant for a given plasma state (T, Ne) across all wavelengths.
    """
    # Placeholder for the actual physics logic involving partition functions and Boltzmann factors.
    # In the full implementation, this returns a vector of populations for each transition.
    return jnp.ones((10,))  # Dummy population vector for refactor demonstration

def compute_cross_section_scaling(params: dict) -> float:
    """Computes scaling factors for self-absorption cross-sections (sigma)."""
    # Placeholder for physics logic regarding transition strengths and path lengths.
    return 1.0  # Dummy sigma scaling

def forward_model(
    params: dict,
    wavelengths: jnp.ndarray,
    atomic_data: Any,
    line_mask: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Main forward model for LIBS spectral synthesis.
    
    Args:
        params: Plasma parameters (T, Ne, etc.).
        wavelengths: Wavelength grid to simulate.
        atomic_data: Database of atomic transitions.
        line_mask: Optional array to mask or scale specific lines (multiplied into epsilon_line).
    
    Returns:
        Synthesized spectral intensities.
    """
    T = params.get('T', 10000.0)
    Ne = params.get('Ne', 1e17)
    
    # Efficiency Refactoring: Hoist Saha and sigma calculations OUT of the lax.scan body.
    # As per efficiency review Finding #1, these are independent of wavelength and
    # redundant to compute inside the scan loop.
    saha_pop = compute_saha_boltzmann(T, Ne, atomic_data)
    sigma = compute_cross_section_scaling(params)
    
    # Apply line mask if provided (equivalent to multiplying epsilon_line per transition)
    if line_mask is not None:
        saha_pop = saha_pop * line_mask

    def scan_body(carry, wl):
        # Inside the scan, we only perform wavelength-dependent calculations.
        # This includes broadening dispatch (Lorentzian/Gaussian/Voigt) and self-absorption.
        
        # Simulated line contribution using hoisted saha_pop
        # In a real kernel, this would involve a sum over lines or a specialized broadening function.
        epsilon_line = jnp.sum(saha_pop / (1.0 + (wl - 300.0)**2)) # Simplified Lorentzian
        
        # Self-absorption logic (e.g., Beer-Lambert law integration)
        kappa = epsilon_line * sigma
        intensity = 1.0 - jnp.exp(-kappa)
        
        return carry, intensity

    _, intensities = lax.scan(scan_body, None, wavelengths)
    return intensities

def _forward_model_per_chunk(
    params: dict,
    wavelengths: jnp.ndarray,
    atomic_data: Any,
    line_mask: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Internal wrapper for chunked spectral synthesis.
    Retired: now a thin wrapper around forward_model to ensure identical behavior
    and benefit from hoisting-based performance improvements.
    """
    return forward_model(params, wavelengths, atomic_data, line_mask=line_mask)
