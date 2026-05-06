"""
Hermann two-region (core+corona) plasma forward model and solver.

This module implements the classical Hermann two-region model for CF-LIBS,
consisting of a hot uniform core and a cooler peripheral shell (corona)
with shared composition and electron density.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy.optimize import least_squares

try:
    import jax.numpy as jnp
    from jax import jit
    HAS_JAX = True
    from cflibs.radiation.profiles import _voigt_profile_kernel_jax
except ImportError:
    HAS_JAX = False
    jnp = None
    def jit(f): return f
    _voigt_profile_kernel_jax = None

from cflibs.core.logging_config import get_logger
from cflibs.core.constants import H_PLANCK, C_LIGHT, EV_TO_K, M_PROTON, SAHA_CONST_CM3
from cflibs.inversion.solve.bayesian import (
    BayesianForwardModel,
    _as_jax_real,
    _resolve_total_species_density_cm3,
    _compute_instrument_sigma,
)

logger = get_logger("inversion.solve.hermann")

class HermannForwardModel(BayesianForwardModel):
    """
    Deterministic Hermann two-region forward model.

    Model: I(lambda) = I_core(lambda) * exp(-tau_shell(lambda)) + 
                      B(lambda, T_shell) * (1 - exp(-tau_shell(lambda)))
    
    The core is assumed to be an optically thin zone of temperature T_core.
    The shell has temperature T_shell and optical thickness tau_shell (proportional to L_shell).
    Both regions share composition and electron density.
    """

    @jit
    def _compute_hermann_spectrum(
        self,
        T_core_eV: float,
        T_shell_eV: float,
        n_e: float,
        tau_shell: float,
        concentrations: jnp.ndarray,
        total_species_density_cm3: Optional[float] = None,
    ) -> jnp.ndarray:
        """
        Compute two-region spectrum with JAX.
        """
        data = self.atomic_data
        T_core_eV = _as_jax_real(T_core_eV)
        T_shell_eV = _as_jax_real(T_shell_eV)
        n_e = _as_jax_real(n_e)
        tau_shell = _as_jax_real(tau_shell)
        concentrations = _as_jax_real(concentrations)
        total_density = _resolve_total_species_density_cm3(n_e, total_species_density_cm3)

        # Constants
        hc_ev_nm = 1239.84193
        
        # --- Shell Region Physics ---
        T_shell_K = T_shell_eV * EV_TO_K
        U0_s = self._partition_function(T_shell_K, data.partition_coeffs[:, 0])
        U1_s = self._partition_function(T_shell_K, data.partition_coeffs[:, 1])
        IP_I = data.ionization_potentials[:, 0]
        
        saha_s = (SAHA_CONST_CM3 / n_e) * (T_shell_eV**1.5)
        ratio_s = saha_s * (U1_s / U0_s) * jnp.exp(-IP_I / T_shell_eV)
        
        frac_0_s = 1.0 / (1.0 + ratio_s)
        frac_1_s = ratio_s / (1.0 + ratio_s)
        
        el_idx = data.element_idx
        ion_stage = data.ion_stage
        pop_frac_s = jnp.where(ion_stage == 0, frac_0_s[el_idx], frac_1_s[el_idx])
        U_val_s = jnp.where(ion_stage == 0, U0_s[el_idx], U1_s[el_idx])
        
        N_species_s = concentrations[el_idx] * total_density * pop_frac_s
        n_upper_s = N_species_s * (data.gk / U_val_s) * jnp.exp(-data.ek_ev / T_shell_eV)
        
        wavelength_m = data.wavelength_nm * 1e-9
        epsilon_s_line = (
            (H_PLANCK * C_LIGHT / (4 * jnp.pi * wavelength_m))
            * data.aki
            * n_upper_s
        )
        
        # Planck source function B_lambda(T_shell) at line centers
        hc_over_lambda_kt_s = hc_ev_nm / (data.wavelength_nm * T_shell_eV)
        planck_s_line = (2 * H_PLANCK * C_LIGHT**2 / wavelength_m**5) / (jnp.exp(hc_over_lambda_kt_s) - 1.0)
        
        # kappa_shell (integrated per line)
        kappa_s_line = epsilon_s_line / planck_s_line

        # --- Core Region Physics ---
        T_core_K = T_core_eV * EV_TO_K
        U0_c = self._partition_function(T_core_K, data.partition_coeffs[:, 0])
        U1_c = self._partition_function(T_core_K, data.partition_coeffs[:, 1])
        saha_c = (SAHA_CONST_CM3 / n_e) * (T_core_eV**1.5)
        ratio_c = saha_c * (U1_c / U0_c) * jnp.exp(-IP_I / T_core_eV)
        frac_0_c = 1.0 / (1.0 + ratio_c)
        frac_1_c = ratio_c / (1.0 + ratio_c)
        pop_frac_c = jnp.where(ion_stage == 0, frac_0_c[el_idx], frac_1_c[el_idx])
        U_val_c = jnp.where(ion_stage == 0, U0_c[el_idx], U1_c[el_idx])
        N_species_c = concentrations[el_idx] * total_density * pop_frac_c
        n_upper_c = N_species_c * (data.gk / U_val_c) * jnp.exp(-data.ek_ev / T_core_eV)
        
        epsilon_c_line = (
            (H_PLANCK * C_LIGHT / (4 * jnp.pi * wavelength_m))
            * data.aki
            * n_upper_c
        )
        
        # Core intensity (integrated per line, assuming L_core = 1 cm = 0.01 m for normalization)
        I_core_line = epsilon_c_line * 0.01

        # --- Line Broadening (shared parameters across zones) ---
        mass_kg = data.mass_amu * M_PROTON
        sigma_doppler = data.wavelength_nm * jnp.sqrt(
            2.0 * T_core_eV * 1.60218e-19 / (mass_kg * C_LIGHT**2)
        )
        sigma_inst = _compute_instrument_sigma(
            data.wavelength_nm, self.instrument_fwhm_nm, self.resolving_power
        )
        sigma_total = jnp.sqrt(sigma_doppler**2 + sigma_inst**2)

        # Stark broadening
        REF_NE = 1.0e16
        REF_T_EV = 0.86173
        binding_energy = jnp.maximum(data.ip_ev - data.ek_ev, 0.1)
        n_eff = (ion_stage + 1) * jnp.sqrt(13.605 / binding_energy)
        w_est = 2.0e-5 * (data.wavelength_nm / 500.0) ** 2 * (n_eff**4)
        w_ref = jnp.where(jnp.isnan(data.stark_w), w_est, data.stark_w)
        gamma_stark = w_ref * (n_e / REF_NE) * jnp.power(jnp.maximum(T_shell_eV, 0.1) / REF_T_EV, -data.stark_alpha)

        # --- Grid-based Radiative Transfer ---
        diff = self.wavelength[:, None] - data.wavelength_nm[None, :]
        profile = _voigt_profile_kernel_jax(diff, sigma_total[None, :], gamma_stark[None, :])
        
        # Spectral quantities summed on grid
        epsilon_s_grid = jnp.sum(epsilon_s_line * profile, axis=1)
        kappa_s_grid = jnp.sum(kappa_s_line * profile, axis=1)
        I_core_grid = jnp.sum(I_core_line * profile, axis=1)
        
        # Planck function on grid
        wl_m_grid = self.wavelength * 1e-9
        hc_over_lambda_kt_grid = hc_ev_nm / (self.wavelength * T_shell_eV)
        planck_grid = (2 * H_PLANCK * C_LIGHT**2 / wl_m_grid**5) / (jnp.exp(hc_over_lambda_kt_grid) - 1.0)
        
        # Spectral optical depth of the shell
        tau_grid = (epsilon_s_grid / planck_grid) * tau_shell
        
        # Hermann equation: I = I_core * exp(-tau) + B * (1 - exp(-tau))
        exp_tau = jnp.exp(-jnp.clip(tau_grid, 0, 50))
        intensity = I_core_grid * exp_tau + planck_grid * (1.0 - exp_tau)
        
        return intensity * 1e-15

    def forward(
        self,
        T_core_eV: float,
        T_shell_eV: float,
        log_ne: float,
        tau_shell: float,
        concentrations: jnp.ndarray,
        total_species_density_cm3: Optional[float] = None,
    ) -> jnp.ndarray:
        """
        Deterministic forward model evaluation.
        """
        n_e = jnp.power(10.0, _as_jax_real(log_ne))
        return self._compute_hermann_spectrum(
            T_core_eV, T_shell_eV, n_e, tau_shell, concentrations,
            total_species_density_cm3=total_species_density_cm3
        )

class HermannSolver:
    """
    Deterministic solver strategy using the Hermann two-region model.
    """
    def __init__(self, forward_model: HermannForwardModel):
        self.forward_model = forward_model

    def solve(
        self,
        observed: np.ndarray,
        initial_guess: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Fit the Hermann model to an observed spectrum using nonlinear least squares.
        """
        elements = self.forward_model.elements
        n_elements = len(elements)
        
        if initial_guess is None:
            initial_guess = {
                "T_core": 1.2, "T_shell": 0.8, "log_ne": 17.0, "tau_shell": 0.1,
                "concentrations": np.ones(n_elements) / n_elements
            }

        def pack(p):
            conc = p["concentrations"]
            logits = np.log(conc[:-1] / conc[-1])
            return np.concatenate([[p["T_core"], p["T_shell"], p["log_ne"], p["tau_shell"]], logits])

        def unpack(x):
            T_c, T_s, lne, tau = x[:4]
            logits = x[4:]
            exp_logits = np.exp(np.clip(logits, -20, 20))
            denom = 1.0 + np.sum(exp_logits)
            conc = np.concatenate([exp_logits / denom, [1.0 / denom]])
            return T_c, T_s, lne, tau, conc

        def fun(x):
            T_c, T_s, lne, tau, conc = unpack(x)
            pred = self.forward_model.forward(T_c, T_s, lne, tau, jnp.array(conc))
            return np.array(pred - observed)

        x0 = pack(initial_guess)
        lb = np.concatenate([[0.1, 0.1, 14.0, 0.0], [-20.0] * (n_elements - 1)])
        ub = np.concatenate([[5.0, 5.0, 20.0, 10.0], [20.0] * (n_elements - 1)])
        
        res = least_squares(fun, x0, bounds=(lb, ub), method='trf', ftol=1e-3)
        T_c, T_s, lne, tau, conc = unpack(res.x)
        
        return {
            "T_core_eV": float(T_c), 
            "T_shell_eV": float(T_s), 
            "log_ne": float(lne),
            "tau_shell": float(tau),
            "concentrations": {el: float(conc[i]) for i, el in enumerate(elements)},
            "rmse": float(np.sqrt(np.mean(res.fun**2))), 
            "success": res.success,
            "message": res.message
        }
