"""
Hydrogen-alpha Voigt Refinement Factor (HVRF) for electron density estimation.
This module provides utilities for extracting n_e from the H-alpha line (656.28 nm)
using improved Stark broadening models that account for ion dynamics and 
temperature dependence.
"""

import jax.numpy as jnp
from jax import jit

@jit
def get_ne_from_halpha_fwhm(fwhm_stark_nm, t_e_k=10000.0):
    """
    Calculates electron density n_e (cm^-3) from the Stark FWHM of the H-alpha line.
    Uses the Gigosos-Cardenoso model (1996) which is an improvement over the 
    Griem semi-empirical formula for LIBS conditions.
    
    Parameters:
        fwhm_stark_nm (float): The Stark component of the FWHM in nanometers.
        t_e_k (float): Electron temperature in Kelvin. Default 10000K.
        
    Returns:
        float: Electron density in cm^-3.
    """
    # Gigosos & Cardenoso (1996) fit: n_e = 10^17 * (FWHM / 1.098)^1.471
    # Temperature dependence is weak but included here for completeness.
    t_norm = t_e_k / 10000.0
    alpha_ref = 1.098 * jnp.power(t_norm, 0.05)
    ne = 1e17 * jnp.power(fwhm_stark_nm / alpha_ref, 1.471)
    
    return ne

@jit
def get_halpha_stark_fwhm(ne_cm3, t_e_k=10000.0):
    """
    Inverse of get_ne_from_halpha_fwhm. Calculates expected Stark FWHM (nm)
    for a given electron density and temperature.
    """
    t_norm = t_e_k / 10000.0
    alpha_ref = 1.098 * jnp.power(t_norm, 0.05)
    fwhm = alpha_ref * jnp.power(ne_cm3 / 1e17, 1.0 / 1.471)
    return fwhm

@jit
def calculate_halpha_doppler_fwhm(t_gas_k):
    """
    Calculates the Doppler FWHM (nm) for the H-alpha line (656.28 nm).
    Assumes M=1.008 for Hydrogen.
    """
    # FWHM_D = 7.16e-7 * lambda * sqrt(T/M)
    return 7.16e-7 * 656.28 * jnp.sqrt(t_gas_k / 1.008)

@jit
def halpha_pseudo_voigt(x, center, fwhm_l, fwhm_g, amplitude, background):
    """
    Pseudo-Voigt profile approximation for H-alpha line fitting.
    """
    # Approximate Voigt FWHM
    f_v = jnp.power(jnp.power(fwhm_g, 5) + 
                    2.69269 * jnp.power(fwhm_g, 4) * fwhm_l + 
                    2.42843 * jnp.power(fwhm_g, 3) * jnp.power(fwhm_l, 2) + 
                    4.47163 * jnp.power(fwhm_g, 2) * jnp.power(fwhm_l, 3) + 
                    0.07842 * fwhm_g * jnp.power(fwhm_l, 4) + 
                    jnp.power(fwhm_l, 5), 0.2)
    
    # Mixing parameter eta
    ratio = fwhm_l / f_v
    eta = 1.36603 * ratio - 0.47719 * jnp.power(ratio, 2) + 0.11116 * jnp.power(ratio, 3)
    
    # Gaussian part
    sigma = fwhm_g / (2 * jnp.sqrt(2 * jnp.log(2)))
    g = (1.0 / (sigma * jnp.sqrt(2 * jnp.pi))) * jnp.exp(-0.5 * jnp.power((x - center) / sigma, 2))
    
    # Lorentzian part
    gamma = fwhm_l / 2.0
    l = (1.0 / jnp.pi) * (gamma / (jnp.power(x - center, 2) + jnp.power(gamma, 2)))
    
    return background + amplitude * (eta * l + (1 - eta) * g)

def estimate_hvrf_correction(ne_initial, t_e, instrument_fwhm_nm):
    """
    Calculates the HVRF correction factor for a given instrument resolution.
    This helps in deconvolution of the Voigt profile when the Gaussian 
    component (instrumental + Doppler) is significant.
    
    Returns the ratio of Stark width to total Voigt width.
    """
    fwhm_doppler = calculate_halpha_doppler_fwhm(t_e)
    fwhm_g = jnp.sqrt(instrument_fwhm_nm**2 + fwhm_doppler**2)
    fwhm_l = get_halpha_stark_fwhm(ne_initial, t_e)
    
    # Approximate Voigt FWHM
    f_v = 0.5346 * fwhm_l + jnp.sqrt(0.2166 * (fwhm_l**2) + fwhm_g**2)
    
    return fwhm_l / f_v
