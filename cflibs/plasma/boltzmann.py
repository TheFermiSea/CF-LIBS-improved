import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, Optional, Tuple

class AtomicData(NamedTuple):
    """
    Container for atomic transition data.
    """
    energy_level: jnp.ndarray    # Upper level energy E_k (eV)
    g_stat: jnp.ndarray          # Statistical weight g_k
    transition_prob: jnp.ndarray # Transition probability A_ki (s^-1)
    wavelength: jnp.ndarray      # Wavelength lambda (nm)

def boltzmann_fit(
    intensities: jnp.ndarray,
    atomic_data: AtomicData,
    weights: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Performs a weighted linear Boltzmann plot fit to estimate plasma temperature.
    
    The Boltzmann equation:
    ln(I * lambda / (g * A)) = -E_k / (k_B * T) + ln(F * C / U(T))
    
    Args:
        intensities: Integrated line intensities.
        atomic_data: AtomicData named tuple containing E_k, g_k, A_ki, and lambda.
        weights: Optional weights for each line (e.g., 1/sigma^2).
        
    Returns:
        Tuple of (temperature in K, intercept).
    """
    k_B = 8.617333262e-5  # eV/K
    
    # y = ln(I * lambda / (g * A))
    # We use jnp.maximum to avoid log(0) or negative intensities
    y = jnp.log(jnp.maximum(intensities, 1e-10) * atomic_data.wavelength / 
                (atomic_data.g_stat * atomic_data.transition_prob))
    x = atomic_data.energy_level
    
    if weights is None:
        weights = jnp.ones_like(y)
    
    # Weighted linear regression: y = slope * x + intercept
    # slope = -1 / (k_B * T)
    
    sum_w = jnp.sum(weights)
    sum_wx = jnp.sum(weights * x)
    sum_wy = jnp.sum(weights * y)
    sum_wxx = jnp.sum(weights * x * x)
    sum_wxy = jnp.sum(weights * x * y)
    
    delta = sum_w * sum_wxx - sum_wx**2
    slope = (sum_w * sum_wxy - sum_wx * sum_wy) / delta
    intercept = (sum_wxx * sum_wy - sum_wx * sum_wxy) / delta
    
    temperature = -1.0 / (k_B * slope)
    
    return temperature, intercept

def saha_boltzmann_ratio(
    T: jnp.ndarray,
    ne: jnp.ndarray,
    E_ion: float,
    U_z: float,
    U_zplus1: float
) -> jnp.ndarray:
    """
    Calculates the ratio of ion to neutral density (n_i / n_n) using the Saha-Boltzmann equation.
    
    n_e * (n_i / n_n) = 2 * (U_i / U_n) * (2 * pi * m_e * k_B * T / h^2)^(3/2) * exp(-E_ion / (k_B * T))
    
    Args:
        T: Plasma temperature (K).
        ne: Electron density (cm^-3).
        E_ion: Ionization energy of the neutral species (eV).
        U_z: Partition function of the neutral species at temperature T.
        U_zplus1: Partition function of the singly ionized species at temperature T.
        
    Returns:
        Ratio n_i / n_n.
    """
    k_B = 8.617333262e-5  # eV/K
    
    # Saha factor constant: 2 * (2 * pi * m_e * k_B / h^2)^(3/2)
    # In units of cm^-3 * K^(-3/2): approx 6.022e21
    saha_const = 6.02214076e21
    
    ratio = (2.0 / ne) * (U_zplus1 / U_z) * saha_const * (T**1.5) * jnp.exp(-E_ion / (k_B * T))
    return ratio

def improved_boltzmann_fit_kthv(
    intensities: jnp.ndarray,
    atomic_data: AtomicData,
    k: int = 5,
    iterations: int = 3
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Improved Boltzmann fit with iterative outlier rejection using the k-th value (kthv).
    
    Args:
        intensities: Integrated line intensities.
        atomic_data: AtomicData named tuple.
        k: The k-th value index for thresholding residuals (outlier rejection).
        iterations: Number of outlier rejection iterations.
        
    Returns:
        Tuple of (temperature in K, intercept).
    """
    weights = jnp.ones_like(intensities)
    
    for _ in range(iterations):
        T, intercept = boltzmann_fit(intensities, atomic_data, weights)
        
        # Calculate residuals
        k_B = 8.617333262e-5
        slope = -1.0 / (k_B * T)
        y_pred = slope * atomic_data.energy_level + intercept
        y_true = jnp.log(jnp.maximum(intensities, 1e-10) * atomic_data.wavelength / 
                         (atomic_data.g_stat * atomic_data.transition_prob))
        
        residuals = jnp.abs(y_true - y_pred)
        
        # kthv: use the k-th largest residual as a threshold
        # We sort residuals and pick the k-th value from the end (or start)
        # Here we'll use a robust threshold based on the k-th smallest residual
        sorted_residuals = jnp.sort(residuals)
        threshold = sorted_residuals[jnp.minimum(k, len(residuals) - 1)] * 2.0
        
        weights = jnp.where(residuals > threshold, 0.0, 1.0)
        
    return boltzmann_fit(intensities, atomic_data, weights)

def saha_boltzmann_anchor(
    T: jnp.ndarray,
    ne: jnp.ndarray,
    species_data: dict
) -> jnp.ndarray:
    """
    Calculates the Saha-Boltzmann anchor for multi-element CF-LIBS.
    """
    # Implementation placeholder for Saha-Boltzmann consistency across species
    return jnp.array(0.0)
