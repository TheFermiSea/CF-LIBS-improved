"""
Batch forward model composing all GPU kernels for manifold generation.

Composes four stages per DERV-05 Eq. (01-03.4):
  Stage 1 -- Direct Saha ionization fractions (given T_eV, n_e)
  Stage 2 -- Boltzmann level populations
  Stage 3 -- Line emissivities
  Stage 4 -- Voigt broadening + spectral assembly

The batch API uses a single-level vmap over parameter tuples (T_eV, n_e, C)
with shared atomic data and wavelength grid:

    batch_forward_model = jit(vmap(single_spectrum_forward,
                                   in_axes=(0, 0, 0, None, None)))

This is the core engine for GPU-accelerated manifold generation.

# ASSERT_CONVENTION: T_eV [eV], n_e [cm^-3], C [dimensionless sum=1],
#   lambda [nm], epsilon [W m^-3 sr^-1], S [W m^-3 sr^-1 nm^-1],
#   gamma = HWHM [nm], sigma = std dev [nm]

References
----------
Kawahara et al. (2022) arXiv:2105.14782 -- ExoJAX: prior art for JAX
    vmap-based spectral batch generation on GPU.
Tognoni et al. (2010) Spectrochim. Acta B 65 -- CF-LIBS methodology.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from cflibs.core.constants import (
    C_LIGHT,
    EV_TO_J,
    EV_TO_K,
    H_PLANCK,
    M_PROTON,
    SAHA_CONST_CM3,
)
from cflibs.core.jax_runtime import HAS_JAX

if HAS_JAX:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap

    from cflibs.plasma.partition import polynomial_partition_function_jax
    from cflibs.radiation.profiles import voigt_spectrum_jax
else:
    jnp = None  # type: ignore[assignment]

    def jit(f):  # type: ignore[misc]
        return f

    def vmap(f, *a, **kw):  # type: ignore[misc]
        return f


# ---------------------------------------------------------------------------
# Atomic data container for the batch forward model
# ---------------------------------------------------------------------------


class BatchAtomicData(NamedTuple):
    """Packed atomic data arrays for batch forward model.

    All arrays are JAX-compatible (or numpy for fallback).

    Attributes
    ----------
    line_wavelengths : array, shape (N_lines,)
        Line center wavelengths [nm].
    line_A_ki : array, shape (N_lines,)
        Einstein A coefficients [s^-1].
    line_g_k : array, shape (N_lines,)
        Upper level statistical weights [dimensionless].
    line_E_k : array, shape (N_lines,)
        Upper level energies [eV].
    line_element_idx : array, shape (N_lines,)
        Integer index mapping each line to its element (0-indexed).
    line_ion_stage : array, shape (N_lines,)
        Ionization stage of each line (0 = neutral I, 1 = singly ionized II).
    line_stark_w : array, shape (N_lines,)
        Stark HWHM at reference n_e = 1e16 cm^-3 [nm].
    line_mass_amu : array, shape (N_lines,)
        Atomic mass of the element for each line [amu].
    ionization_potentials : array, shape (N_elements, max_stages-1)
        Ionization potentials [eV].  Index 0 = I->II, 1 = II->III.
    partition_coeffs : array, shape (N_elements, max_stages, 5)
        Polynomial partition function coefficients (Irwin form).
    n_elements : int
        Number of elements.
    n_stages : int
        Number of ionization stages per element.
    """

    line_wavelengths: Any
    line_A_ki: Any
    line_g_k: Any
    line_E_k: Any
    line_element_idx: Any
    line_ion_stage: Any
    line_stark_w: Any
    line_mass_amu: Any
    ionization_potentials: Any
    partition_coeffs: Any
    n_elements: int
    n_stages: int


# Register as JAX pytree
if HAS_JAX:
    _BATCH_ATOMIC_FIELDS = BatchAtomicData._fields

    def _batch_atomic_flatten(data: BatchAtomicData):
        # Treat n_elements and n_stages as auxiliary (static) data
        children = [getattr(data, f) for f in _BATCH_ATOMIC_FIELDS[:-2]]
        aux = (data.n_elements, data.n_stages)
        return children, aux

    def _batch_atomic_unflatten(aux, children):
        return BatchAtomicData(*children, *aux)

    jax.tree_util.register_pytree_node(
        BatchAtomicData, _batch_atomic_flatten, _batch_atomic_unflatten
    )


def pack_atomic_data(
    elements: list[str],
    line_data: list[dict[str, Any]],
    max_stages: int = 3,
) -> BatchAtomicData:
    """Pack atomic line and species data into BatchAtomicData arrays.

    This is a convenience function for constructing BatchAtomicData from
    a list of line dictionaries (typically extracted from an AtomicDatabase).

    Parameters
    ----------
    elements : list of str
        Element symbols in order (defines the element index mapping).
    line_data : list of dict
        Each dict has keys: 'wavelength_nm', 'A_ki', 'g_k', 'E_k_eV',
        'element', 'ion_stage' (0=I, 1=II), 'stark_w_nm' (HWHM at 1e16),
        'mass_amu'.
    max_stages : int
        Maximum ionization stages per element (default 3: I, II, III).

    Returns
    -------
    BatchAtomicData
        Packed arrays ready for single_spectrum_forward / batch_forward_model.

    Notes
    -----
    For production use, also provide ionization_potentials and
    partition_coeffs via the returned NamedTuple fields.  This function
    sets them to zeros; callers should replace them.
    """
    elem_to_idx = {e: i for i, e in enumerate(elements)}
    n_lines = len(line_data)
    n_elem = len(elements)

    wl = np.zeros(n_lines, dtype=np.float64)
    A_ki = np.zeros(n_lines, dtype=np.float64)
    g_k = np.zeros(n_lines, dtype=np.float64)
    E_k = np.zeros(n_lines, dtype=np.float64)
    elem_idx = np.zeros(n_lines, dtype=np.int32)
    ion_stage = np.zeros(n_lines, dtype=np.int32)
    stark_w = np.zeros(n_lines, dtype=np.float64)
    mass_amu = np.zeros(n_lines, dtype=np.float64)

    for i, ld in enumerate(line_data):
        wl[i] = ld["wavelength_nm"]
        A_ki[i] = ld["A_ki"]
        g_k[i] = ld["g_k"]
        E_k[i] = ld["E_k_eV"]
        elem_idx[i] = elem_to_idx[ld["element"]]
        ion_stage[i] = ld["ion_stage"]
        stark_w[i] = ld.get("stark_w_nm", 0.01)
        mass_amu[i] = ld["mass_amu"]

    ip = np.zeros((n_elem, max_stages - 1), dtype=np.float64)
    pf = np.zeros((n_elem, max_stages, 5), dtype=np.float64)
    # Default partition function: log(U) ~ log(2) (ground state g~2)
    pf[:, :, 0] = np.log(2.0)

    return BatchAtomicData(
        line_wavelengths=np.asarray(wl),
        line_A_ki=np.asarray(A_ki),
        line_g_k=np.asarray(g_k),
        line_E_k=np.asarray(E_k),
        line_element_idx=np.asarray(elem_idx),
        line_ion_stage=np.asarray(ion_stage),
        line_stark_w=np.asarray(stark_w),
        line_mass_amu=np.asarray(mass_amu),
        ionization_potentials=np.asarray(ip),
        partition_coeffs=np.asarray(pf),
        n_elements=n_elem,
        n_stages=max_stages,
    )


# ---------------------------------------------------------------------------
# Constants for JAX
# ---------------------------------------------------------------------------
_SAHA_CONST = float(SAHA_CONST_CM3)
_EV_TO_K = float(EV_TO_K)
_HC_OVER_4PI = float(H_PLANCK * C_LIGHT / (4.0 * np.pi))  # J*m
_EV_TO_J = float(EV_TO_J)
_M_PROTON = float(M_PROTON)
_C_LIGHT = float(C_LIGHT)


# ---------------------------------------------------------------------------
# JAX implementation
# ---------------------------------------------------------------------------

if HAS_JAX:

    @jit
    def _saha_ionization_fractions(
        T_eV: jnp.ndarray,
        n_e: jnp.ndarray,
        ip: jnp.ndarray,
        pf_coeffs: jnp.ndarray,
        n_stages: int,
    ) -> jnp.ndarray:
        """Compute ionization fractions for one element via direct Saha.

        Parameters
        ----------
        T_eV : scalar
            Temperature [eV].
        n_e : scalar
            Electron density [cm^-3].
        ip : array, shape (max_stages-1,)
            Ionization potentials [eV].
        pf_coeffs : array, shape (max_stages, 5)
            Partition function polynomial coefficients.
        n_stages : int
            Number of ionization stages.

        Returns
        -------
        fractions : array, shape (max_stages,)
            Ionization fractions f_z (sum = 1).
        """
        T_K = T_eV * _EV_TO_K

        # Evaluate partition functions for all stages
        U = polynomial_partition_function_jax(T_K, pf_coeffs)  # (max_stages,)
        U = jnp.maximum(U, 1e-30)

        # Saha prefactor: SAHA_CONST / n_e * T_eV^1.5
        saha_pref = _SAHA_CONST / jnp.maximum(n_e, 1.0) * T_eV**1.5

        # Compute cumulative Saha products
        # S[z] = saha_pref * U[z+1]/U[z] * exp(-IP[z]/T_eV)
        # f_0 = 1 / (1 + S0 + S0*S1 + S0*S1*S2 + ...)
        # For max_stages=3: f_0 = 1/(1 + S0 + S0*S1), f_1 = S0*f_0, f_2 = S0*S1*f_0

        max_st = pf_coeffs.shape[0]
        max_trans = ip.shape[0]

        # Build cumulative products
        # P[0] = 1 (neutral)
        # P[z] = S[0] * S[1] * ... * S[z-1]
        def _scan_step(carry, z):
            # z is transition index (0-indexed: 0 = I->II, 1 = II->III)
            s = saha_pref * U[z + 1] / U[z] * jnp.exp(-ip[z] / jnp.maximum(T_eV, 1e-10))
            valid = (z < max_trans) & (ip[z] > 0) & (z < n_stages - 1)
            new_carry = jnp.where(valid, carry * s, 0.0)
            return new_carry, new_carry

        _, P_tail = jax.lax.scan(_scan_step, jnp.float64(1.0), jnp.arange(max_trans))
        populations = jnp.concatenate([jnp.array([1.0]), P_tail])[:max_st]

        # Mask invalid stages
        stage_mask = jnp.arange(max_st) < n_stages
        populations = jnp.where(stage_mask, populations, 0.0)

        # Normalize to fractions
        total = jnp.sum(populations)
        fractions = populations / jnp.maximum(total, 1e-300)
        return fractions

    @jit
    def single_spectrum_forward(
        T_eV: jnp.ndarray,
        n_e: jnp.ndarray,
        concentrations: jnp.ndarray,
        wl_grid: jnp.ndarray,
        atomic_data: BatchAtomicData,
    ) -> jnp.ndarray:
        """Compute a single CF-LIBS synthetic spectrum.

        Composes four stages: Saha -> Boltzmann -> emissivity -> Voigt.

        Parameters
        ----------
        T_eV : scalar
            Electron temperature [eV].
        n_e : scalar
            Electron density [cm^-3].
        concentrations : array, shape (N_elements,)
            Number fractions C_i (sum = 1).
        wl_grid : array, shape (N_wl,)
            Wavelength grid [nm].
        atomic_data : BatchAtomicData
            Packed atomic data arrays.

        Returns
        -------
        spectrum : array, shape (N_wl,)
            Synthetic spectrum S(lambda) [W m^-3 sr^-1 nm^-1].
        """
        T_eV = jnp.asarray(T_eV, dtype=jnp.float64)
        n_e = jnp.asarray(n_e, dtype=jnp.float64)

        n_elem = atomic_data.n_elements
        n_stages = atomic_data.n_stages
        ip = jnp.asarray(atomic_data.ionization_potentials, dtype=jnp.float64)
        pf_coeffs = jnp.asarray(atomic_data.partition_coeffs, dtype=jnp.float64)

        # ----- Stage 1: Saha ionization fractions for each element -----
        def _elem_fractions(elem_idx):
            return _saha_ionization_fractions(
                T_eV, n_e, ip[elem_idx], pf_coeffs[elem_idx], n_stages
            )

        # (N_elements, max_stages)
        all_fractions = vmap(_elem_fractions)(jnp.arange(n_elem))

        # ----- Stage 2+3: Boltzmann populations + line emissivities -----
        # For each line: n_k = C_s * n_e * f_z * (g_k / U_z) * exp(-E_k / T_eV)
        # epsilon_l = hc / (4*pi*lambda_l) * A_ki * n_k * 1e6  [W m^-3 sr^-1]

        line_wl = jnp.asarray(atomic_data.line_wavelengths, dtype=jnp.float64)
        line_Aki = jnp.asarray(atomic_data.line_A_ki, dtype=jnp.float64)
        line_gk = jnp.asarray(atomic_data.line_g_k, dtype=jnp.float64)
        line_Ek = jnp.asarray(atomic_data.line_E_k, dtype=jnp.float64)
        line_elem = jnp.asarray(atomic_data.line_element_idx, dtype=jnp.int32)
        line_stage = jnp.asarray(atomic_data.line_ion_stage, dtype=jnp.int32)
        line_stark = jnp.asarray(atomic_data.line_stark_w, dtype=jnp.float64)
        line_mass = jnp.asarray(atomic_data.line_mass_amu, dtype=jnp.float64)

        T_K = T_eV * _EV_TO_K

        # Evaluate partition functions for all (element, stage) pairs
        pf_flat = pf_coeffs.reshape(-1, 5)
        U_flat = polynomial_partition_function_jax(T_K, pf_flat)
        U_all = U_flat.reshape(n_elem, n_stages)  # (N_elem, max_stages)

        # Per-line: gather the relevant quantities
        C_line = concentrations[line_elem]  # (N_lines,)
        f_line = all_fractions[line_elem, line_stage]  # (N_lines,)
        U_line = jnp.maximum(U_all[line_elem, line_stage], 1e-30)  # (N_lines,)

        # Boltzmann factor: g_k / U(T) * exp(-E_k / T_eV)
        boltz = line_gk / U_line * jnp.exp(-line_Ek / jnp.maximum(T_eV, 1e-10))

        # Upper level population [cm^-3]
        # n_k = C_s * n_e * f_z * boltz
        # Using n_e as proxy for total number density (standard CF-LIBS approx)
        n_k = C_line * n_e * f_line * boltz

        # Line emissivity [W m^-3 sr^-1]
        # epsilon = hc / (4*pi*lambda) * A_ki * n_k * 1e6
        # lambda in nm -> convert to m: lambda * 1e-9
        # hc/(4*pi) has units J*m
        # epsilon = _HC_OVER_4PI / (lambda_m) * A_ki * n_k * 1e6
        lambda_m = line_wl * 1e-9  # nm -> m
        emissivities = _HC_OVER_4PI / jnp.maximum(lambda_m, 1e-30) * line_Aki * n_k * 1e6

        # ----- Stage 4: Voigt broadening -----
        # Gaussian sigma: Doppler width
        # sigma_D = lambda * sqrt(kT / (m_atom * c^2)) [nm]
        mass_kg = line_mass * _M_PROTON
        sigma_D = line_wl * jnp.sqrt(T_eV * _EV_TO_J / (mass_kg * _C_LIGHT**2))

        # Lorentzian gamma: Stark HWHM
        # gamma_S = stark_w * (n_e / 1e16) [nm]
        gamma_S = line_stark * (n_e / 1e16)

        # Ensure positive widths
        sigma_D = jnp.maximum(sigma_D, 1e-6)  # minimum 1e-6 nm
        gamma_S = jnp.maximum(gamma_S, 1e-6)  # minimum 1e-6 nm

        # Use voigt_spectrum_jax for broadcasting outer product assembly
        spectrum = voigt_spectrum_jax(wl_grid, line_wl, emissivities, sigma_D, gamma_S)

        return spectrum

    # Batch via vmap per DERV-05 Eq. (01-03.2)
    batch_forward_model = jit(vmap(single_spectrum_forward, in_axes=(0, 0, 0, None, None)))

else:
    # NumPy fallback for CPU-only machines without JAX
    def single_spectrum_forward(
        T_eV: float,
        n_e: float,
        concentrations: np.ndarray,
        wl_grid: np.ndarray,
        atomic_data: BatchAtomicData,
    ) -> np.ndarray:
        """NumPy fallback for single spectrum computation.

        This is slow but correct -- used for CI testing when JAX is unavailable.
        """
        from cflibs.radiation.profiles import apply_voigt_broadening

        T_eV = float(T_eV)
        n_e = float(n_e)
        concentrations = np.asarray(concentrations, dtype=np.float64)
        n_elem = atomic_data.n_elements
        n_stages = atomic_data.n_stages
        ip = np.asarray(atomic_data.ionization_potentials)
        pf_coeffs = np.asarray(atomic_data.partition_coeffs)

        # Stage 1: Saha fractions per element (direct)
        from cflibs.plasma.partition import polynomial_partition_function

        T_K = T_eV * _EV_TO_K
        all_fractions = np.zeros((n_elem, n_stages))

        for ie in range(n_elem):
            populations = np.zeros(n_stages)
            populations[0] = 1.0
            cum_prod = 1.0
            for z in range(n_stages - 1):
                coeffs_z = list(pf_coeffs[ie, z])
                coeffs_z1 = list(pf_coeffs[ie, z + 1])
                U_z = max(polynomial_partition_function(T_K, coeffs_z), 1e-30)
                U_z1 = polynomial_partition_function(T_K, coeffs_z1)
                if ip[ie, z] <= 0:
                    break
                S = (
                    _SAHA_CONST
                    / max(n_e, 1.0)
                    * T_eV**1.5
                    * U_z1
                    / U_z
                    * np.exp(-ip[ie, z] / max(T_eV, 1e-10))
                )
                cum_prod *= S
                populations[z + 1] = cum_prod
            total = np.sum(populations)
            all_fractions[ie] = populations / max(total, 1e-300)

        # Stage 2+3: populations and emissivities
        line_wl = np.asarray(atomic_data.line_wavelengths)
        line_Aki = np.asarray(atomic_data.line_A_ki)
        line_gk = np.asarray(atomic_data.line_g_k)
        line_Ek = np.asarray(atomic_data.line_E_k)
        line_elem = np.asarray(atomic_data.line_element_idx, dtype=int)
        line_stage = np.asarray(atomic_data.line_ion_stage, dtype=int)
        line_stark = np.asarray(atomic_data.line_stark_w)
        line_mass = np.asarray(atomic_data.line_mass_amu)

        # Partition functions
        U_all = np.zeros((n_elem, n_stages))
        for ie in range(n_elem):
            for iz in range(n_stages):
                coeffs = list(pf_coeffs[ie, iz])
                U_all[ie, iz] = max(polynomial_partition_function(T_K, coeffs), 1e-30)

        C_line = concentrations[line_elem]
        f_line = all_fractions[line_elem, line_stage]
        U_line = np.maximum(U_all[line_elem, line_stage], 1e-30)

        boltz = line_gk / U_line * np.exp(-line_Ek / max(T_eV, 1e-10))
        n_k = C_line * n_e * f_line * boltz

        lambda_m = line_wl * 1e-9
        emissivities = _HC_OVER_4PI / np.maximum(lambda_m, 1e-30) * line_Aki * n_k * 1e6

        # Stage 4: broadening
        mass_kg = line_mass * _M_PROTON
        sigma_D = line_wl * np.sqrt(T_eV * _EV_TO_J / (mass_kg * _C_LIGHT**2))
        gamma_S = line_stark * (n_e / 1e16)
        sigma_D = np.maximum(sigma_D, 1e-6)
        gamma_S = np.maximum(gamma_S, 1e-6)

        spectrum = apply_voigt_broadening(wl_grid, line_wl, emissivities, sigma_D, gamma_S)
        return spectrum

    def batch_forward_model(
        T_eV_batch: np.ndarray,
        n_e_batch: np.ndarray,
        concentrations_batch: np.ndarray,
        wl_grid: np.ndarray,
        atomic_data: BatchAtomicData,
    ) -> np.ndarray:
        """NumPy fallback for batch forward model (sequential loop)."""
        B = T_eV_batch.shape[0]
        spectra = []
        for i in range(B):
            s = single_spectrum_forward(
                T_eV_batch[i],
                n_e_batch[i],
                concentrations_batch[i],
                wl_grid,
                atomic_data,
            )
            spectra.append(s)
        return np.stack(spectra, axis=0)
