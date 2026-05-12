"""Unified forward-model kernel (ADR-0001 T1-2).

Single source of truth for the CF-LIBS forward physics:

    plasma_state, atomic_snapshot, instrument, wavelength_grid
                                |
                                v
                       saha_boltzmann_populations
                                |
                                v
                  per-line emissivity epsilon_l
                                |
                                v
            mode-dispatched profile / broadening kernel
                                |
                                v
      optional Planck * (1 - exp(-kappa * L))  (apply_self_absorption)
                                |
                                v
                       I(lambda)  W/m^2/nm/sr

The three historical implementations
(:class:`cflibs.radiation.spectrum_model.SpectrumModel`,
:func:`cflibs.manifold.batch_forward.single_spectrum_forward`,
:meth:`cflibs.inversion.solve.bayesian.BayesianForwardModel._compute_spectrum`)
all dispatch through :func:`forward_model` after T1-2.

Static-vs-traced split (jit cache key):
    static : ``broadening_mode``, ``apply_self_absorption``,
             shape of ``wavelength_grid``, shape of snapshot line/species
             arrays.
    traced : ``plasma_state.T_e_eV``, ``.n_e``, ``.composition_array``;
             all snapshot ``line_*`` arrays; instrument widths;
             ``wavelength_grid``; ``path_length_m``.

Precision is governed by :func:`cflibs.core.jax_runtime.jax_policy().real_dtype`
when JAX is present. On Metal (no fp64) callers must construct
:class:`cflibs.core.jax_runtime.JaxMemoryPolicy(allow_32bit=True)` upstream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from cflibs.core.constants import (
    C_LIGHT,
    EV_TO_J,
    EV_TO_K,
    H_PLANCK,
    KB,
    M_PROTON,
    SAHA_CONST_CM3,
)
from cflibs.core.jax_runtime import jnp
from cflibs.core.logging_config import get_logger
from cflibs.radiation.profiles import BroadeningMode

logger = get_logger("radiation.kernels")

if TYPE_CHECKING:  # pragma: no cover - imports for type checkers only
    from cflibs.core.jax_runtime import AtomicSnapshot
    from cflibs.instrument.model import InstrumentModel
    from cflibs.plasma.state import SingleZoneLTEPlasma

# Constants packed into jnp scalars on the host so they enter the jit cache as
# float-valued traced inputs (no Python literals captured by the cache key).
_HC_OVER_4PI = H_PLANCK * C_LIGHT / (4.0 * np.pi)

# Standard atomic masses (amu) for elements we cannot look up directly from the
# snapshot. The snapshot does not currently carry per-species mass, so we keep
# a host-side table identical to ``SpectrumModel._FALLBACK_MASSES``. T2-2 will
# fold ``mass_amu`` into ``AtomicSnapshot`` and we will drop this dependency.
_FALLBACK_MASSES = {
    "H": 1.008,
    "He": 4.003,
    "Li": 6.941,
    "Be": 9.012,
    "B": 10.81,
    "C": 12.01,
    "N": 14.01,
    "O": 16.00,
    "F": 19.00,
    "Ne": 20.18,
    "Na": 22.99,
    "Mg": 24.31,
    "Al": 26.98,
    "Si": 28.09,
    "P": 30.97,
    "S": 32.07,
    "Cl": 35.45,
    "Ar": 39.95,
    "K": 39.10,
    "Ca": 40.08,
    "Sc": 44.96,
    "Ti": 47.87,
    "V": 50.94,
    "Cr": 52.00,
    "Mn": 54.94,
    "Fe": 55.85,
    "Co": 58.93,
    "Ni": 58.69,
    "Cu": 63.55,
    "Zn": 65.38,
    "Ga": 69.72,
    "Ge": 72.63,
    "As": 74.92,
    "Se": 78.97,
    "Br": 79.90,
    "Kr": 83.80,
    "Rb": 85.47,
    "Sr": 87.62,
    "Y": 88.91,
    "Zr": 91.22,
    "Nb": 92.91,
    "Mo": 95.95,
    "Ag": 107.87,
    "Cd": 112.41,
    "Sn": 118.71,
    "Sb": 121.76,
    "I": 126.90,
    "Cs": 132.91,
    "Ba": 137.33,
    "W": 183.84,
    "Pt": 195.08,
    "Au": 196.97,
    "Hg": 200.59,
    "Pb": 207.20,
    "U": 238.03,
}


def _species_mass_array(snapshot: "AtomicSnapshot") -> np.ndarray:
    """Return a ``(N_species,)`` numpy array of atomic masses (amu).

    Looked up from the ``snapshot.species`` tuple via the host-side fallback
    table. Padded species (empty element string) get mass 1.0 (never reached
    because their line count is zero).
    """
    masses = []
    for element, _stage in snapshot.species:
        if not element:
            masses.append(1.0)
        else:
            masses.append(float(_FALLBACK_MASSES.get(element, 1.0)))
    return np.asarray(masses, dtype=np.float64)


# ---------------------------------------------------------------------------
# Saha-Boltzmann populations (snapshot-based, two-stage I/II)
# ---------------------------------------------------------------------------


def _polynomial_partition_function_jax(T_K, coeffs):
    """log U = sum_n a_n (log10 T_K)^n. Returns U on the active dtype.

    ``coeffs`` has shape ``(N_species, 5)``; row index matches ``snapshot.species``.
    Mirrors :func:`cflibs.plasma.partition.polynomial_partition_function_jax`
    but inlined to avoid extra import overhead inside the jit.
    """
    log_T = jnp.log10(jnp.maximum(T_K, 1.0))
    log_T_powers = jnp.stack([jnp.ones_like(log_T), log_T, log_T**2, log_T**3, log_T**4], axis=-1)
    log_U = jnp.sum(coeffs * log_T_powers, axis=-1)
    return jnp.power(10.0, log_U)


def _saha_two_stage_populations(plasma_state, snapshot):
    """Two-stage (I, II) Saha-Boltzmann populations per snapshot line.

    Returns
    -------
    n_upper_per_line : (N_lines,) jnp.ndarray
        Upper-level population in cm^-3 for each line in the snapshot.

    Notes
    -----
    Mirrors :meth:`BayesianForwardModel._compute_spectrum`'s Saha-Boltzmann
    block. Treats every species as I or II only (``stage in {1, 2}``); higher
    stages contribute negligibly to LIBS at ``T_e <= 2 eV``. Per-element
    concentrations are derived from ``plasma_state.species`` densities --
    we keep the values as jnp scalars so the function stays vmap-clean.

    Heavy-particle density proxy: ``N_total = n_e`` (matches the legacy
    Bayesian/manifold convention). T1-6 will refine this with an optional
    ``total_species_density_cm3`` override.
    """
    T_eV = jnp.asarray(plasma_state.T_e_eV)
    n_e = jnp.asarray(plasma_state.n_e)
    T_K = T_eV * EV_TO_K

    species_list = snapshot.species

    # ---- Host-side index maps (static) ----
    elements_in_snapshot: list[str] = []
    for el, _stage in species_list:
        if el and el not in elements_in_snapshot:
            elements_in_snapshot.append(el)

    # Map each species row -> element-row index (sentinel 0 for padded rows).
    species_to_element_idx = np.array(
        [
            elements_in_snapshot.index(el) if el in elements_in_snapshot else 0
            for el, _ in species_list
        ],
        dtype=np.int32,
    )
    # Map element-row -> species-row index for stage I / stage II (sentinel -1).
    n_elements = len(elements_in_snapshot)
    sp_idx_I = -np.ones(n_elements, dtype=np.int32)
    sp_idx_II = -np.ones(n_elements, dtype=np.int32)
    for sp_idx, (el, stage) in enumerate(species_list):
        if not el:
            continue
        el_idx = elements_in_snapshot.index(el)
        if stage == 1:
            sp_idx_I[el_idx] = sp_idx
        elif stage == 2:
            sp_idx_II[el_idx] = sp_idx
    stage_per_species = np.array([int(s) for _, s in species_list], dtype=np.int32)

    # ---- Per-element density and concentration (traced under vmap) ----
    # Pull each element's density as a jnp scalar via ``species[el]``; this
    # follows the pytree -> dict mapping registered for SingleZoneLTEPlasma.
    dtype = T_eV.dtype if hasattr(T_eV, "dtype") else jnp.float64
    if n_elements == 0:
        # Empty snapshot: nothing to compute.
        return jnp.zeros_like(jnp.asarray(snapshot.line_wavelengths_nm))
    densities = jnp.stack(
        [jnp.asarray(plasma_state.species[el], dtype=dtype) for el in elements_in_snapshot]
    )
    sum_density = jnp.sum(densities)
    concentrations = jnp.where(
        sum_density > 0.0, densities / jnp.maximum(sum_density, 1e-300), jnp.zeros_like(densities)
    )

    # ---- Partition functions for every species row (traced) ----
    pf_all = jnp.asarray(snapshot.partition_coeffs, dtype=dtype)
    ip_all = jnp.asarray(snapshot.ionization_potential_ev, dtype=dtype)
    U_per_species = jnp.maximum(_polynomial_partition_function_jax(T_K, pf_all), 1e-30)

    # Gather stage-I / stage-II values per element. For missing-stage
    # elements (sentinel -1) we fall back to species 0 then zero out the
    # ratio via a mask.
    valid_I_mask = jnp.asarray(sp_idx_I >= 0, dtype=dtype)
    valid_II_mask = jnp.asarray(sp_idx_II >= 0, dtype=dtype)
    safe_I = jnp.where(jnp.asarray(sp_idx_I) >= 0, jnp.asarray(sp_idx_I), 0)
    safe_II = jnp.where(jnp.asarray(sp_idx_II) >= 0, jnp.asarray(sp_idx_II), 0)
    U_I = U_per_species[safe_I]
    U_II = U_per_species[safe_II]
    ip_I = ip_all[safe_I] * valid_I_mask

    saha_factor = (SAHA_CONST_CM3 / jnp.maximum(n_e, 1.0)) * (T_eV**1.5)
    ratio = (
        saha_factor
        * (U_II / U_I)
        * jnp.exp(-ip_I / jnp.maximum(T_eV, 1e-10))
        * valid_I_mask
        * valid_II_mask
    )
    frac_I = 1.0 / (1.0 + ratio)
    frac_II = ratio / (1.0 + ratio)

    # ---- Per-line populations ----
    line_species_index = jnp.asarray(snapshot.line_species_index, dtype=jnp.int32)
    line_g_k = jnp.asarray(snapshot.line_g_k, dtype=dtype)
    line_E_k_ev = jnp.asarray(snapshot.line_E_k_ev, dtype=dtype)
    U_line = U_per_species[line_species_index]
    line_element_idx = jnp.asarray(species_to_element_idx, dtype=jnp.int32)[line_species_index]
    line_stage = jnp.asarray(stage_per_species, dtype=jnp.int32)[line_species_index]

    pop_fraction = jnp.where(
        line_stage == 1,
        frac_I[line_element_idx],
        frac_II[line_element_idx],
    )

    C_per_line = concentrations[line_element_idx]
    N_total = n_e
    boltz = (line_g_k / U_line) * jnp.exp(-line_E_k_ev / jnp.maximum(T_eV, 1e-10))
    return C_per_line * N_total * pop_fraction * boltz


# ---------------------------------------------------------------------------
# Per-line broadening helpers
# ---------------------------------------------------------------------------


def _per_line_doppler_sigma(snapshot, T_eV, line_mass_amu):
    """Per-line Gaussian Doppler sigma (nm). Uses snapshot wavelengths."""
    wl = jnp.asarray(snapshot.line_wavelengths_nm)
    mass_kg = jnp.asarray(line_mass_amu) * M_PROTON
    return wl * jnp.sqrt(T_eV * EV_TO_J / (mass_kg * C_LIGHT**2))


def _per_line_instrument_sigma(snapshot, instrument):
    """Per-line instrumental Gaussian sigma (nm).

    - Resolving-power mode (NIST_PARITY-style): sigma = lambda_i / (R * 2.355).
    - Fixed-FWHM mode: scalar sigma broadcast across the line axis.

    Uses the rounded factor ``2.355`` (NOT the exact ``2 sqrt(2 ln 2)``) to
    match :meth:`InstrumentModel.sigma_at_wavelength` exactly -- pre-T1-2
    callers depend on the rounded constant.
    """
    wl = jnp.asarray(snapshot.line_wavelengths_nm)
    if instrument.is_resolving_power_mode:
        R = float(instrument.resolving_power)
        return wl / (R * 2.355)
    sigma_scalar = float(instrument.resolution_sigma_nm)
    return jnp.full(wl.shape, sigma_scalar, dtype=wl.dtype)


def _per_line_stark_gamma(snapshot, n_e, T_eV):
    """Per-line Lorentzian Stark HWHM (nm).

    Reference scaling: gamma_S(n_e) = stark_w * (n_e / 1e16).
    Temperature dependence is omitted at the kernel level (the existing
    manifold path matches this convention; BayesianForwardModel adds a power-
    law factor that we keep there at the wrapper level if needed).
    """
    stark_w = jnp.asarray(snapshot.line_stark_w)
    return stark_w * (n_e / 1.0e16)


# ---------------------------------------------------------------------------
# Profile kernels (broadening modes)
# ---------------------------------------------------------------------------


def _gaussian_sum_per_line(wavelength_grid, line_centers, line_intensities, line_sigmas):
    """Sum of per-line Gaussians on the wavelength grid (outer-product form)."""
    wl = wavelength_grid[:, None]
    centers = line_centers[None, :]
    sigmas = jnp.maximum(line_sigmas[None, :], 1e-12)
    x = (wl - centers) / sigmas
    profile = jnp.exp(-0.5 * x**2) / (sigmas * jnp.sqrt(2.0 * jnp.pi))
    return jnp.sum(line_intensities[None, :] * profile, axis=1)


def _voigt_sum_per_line(wavelength_grid, line_centers, line_intensities, sigmas, gammas):
    """Sum of per-line Voigt profiles using the Weideman Faddeeva kernel.

    Delegates to :func:`cflibs.radiation.profiles._voigt_profile_kernel_jax`
    so we inherit the Metal-safe real-arithmetic fallback for free.
    """
    from cflibs.radiation.profiles import _voigt_profile_kernel_jax

    diff = wavelength_grid[:, None] - line_centers[None, :]
    profile = _voigt_profile_kernel_jax(diff, sigmas[None, :], gammas[None, :])
    return jnp.sum(line_intensities[None, :] * profile, axis=1)


# ---------------------------------------------------------------------------
# Unified forward model
# ---------------------------------------------------------------------------


def forward_model(
    plasma_state: "SingleZoneLTEPlasma",
    atomic_snapshot: "AtomicSnapshot",
    instrument: "InstrumentModel",
    wavelength_grid,
    sigma_grid=None,
    *,
    broadening_mode: BroadeningMode,
    path_length_m: float,
    apply_self_absorption: bool = False,
    fold_instrument_sigma: bool = True,
    apply_stark: bool = False,
    total_species_density_cm3: Optional[float] = None,
    _precomputed_n_upper_per_line=None,
):
    """Unified forward kernel -- one source of truth for CF-LIBS forward physics.

    Parameters
    ----------
    plasma_state : SingleZoneLTEPlasma
        Plasma state (pytree-registered).
    atomic_snapshot : AtomicSnapshot
        Frozen snapshot built by :meth:`AtomicDatabase.snapshot`.
    instrument : InstrumentModel
        Either fixed-FWHM or resolving-power mode.
    wavelength_grid : array, shape (N_wl,)
        Wavelength grid in nm.
    sigma_grid : array, optional
        Log-sigma grid for ``BroadeningMode.LDM_GAUSSIAN``. Required for that
        mode; ignored otherwise.
    broadening_mode : BroadeningMode
        Static dispatch knob:
          - ``LEGACY`` : scalar sigma = 0.01 * sqrt(T_eV / 0.86) nm. Gaussian.
          - ``NIST_PARITY`` : per-line sigma = lambda_i / (R * 2.355). Gaussian.
          - ``PHYSICAL_DOPPLER`` : per-line physical Doppler sigma; optionally
            folds instrument sigma in quadrature; optional Lorentzian Stark.
          - ``LDM_GAUSSIAN`` : Per-line Gaussian via LDM/DIT.
    path_length_m : float
        Plasma path length (m). Only consulted when ``apply_self_absorption``.
    apply_self_absorption : bool, default False
        If True, apply Planck * (1 - exp(-kappa * L)) radiative transfer.
        ``SpectrumModel.compute_spectrum`` defaults this to True; manifold and
        Bayesian wrappers leave it False (thin plasma).
    fold_instrument_sigma : bool, default True
        If True (Bayesian convention), add instrument Gaussian sigma in
        quadrature to per-line Doppler sigma for ``PHYSICAL_DOPPLER``. If False
        (legacy ``SpectrumModel`` convention), let the host wrapper apply a
        separate downstream scipy convolution.
    apply_stark : bool, default False
        If True, include the per-line Lorentzian Stark width on the Voigt path
        (``PHYSICAL_DOPPLER`` only). Manifold uses True; Bayesian uses True;
        ``SpectrumModel`` legacy path uses False.
    total_species_density_cm3 : float, optional
        Override heavy-particle total density. ``None`` -> legacy proxy
        ``N_total = n_e``.
    _precomputed_n_upper_per_line : array, optional
        Escape hatch for callers (notably ``SpectrumModel.compute_spectrum``)
        that need to inject upper-level populations computed by the legacy
        detailed-levels ``SahaBoltzmannSolver`` to preserve numerical parity
        with pre-T1-2 output. Production callers should leave this ``None``
        and let the kernel run its snapshot-based Saha-Boltzmann path.

    Returns
    -------
    intensity : array, shape (N_wl,)
        Synthetic emissivity per nm. Units: W m^-3 nm^-1 sr^-1 when
        ``apply_self_absorption`` is False; W m^-2 nm^-1 sr^-1 when True.

    Notes
    -----
    ``_precomputed_n_upper_per_line`` is a deviation from spec §2 prompted by
    the stop-condition guidance: the unified Saha-Boltzmann path uses
    polynomial partition functions, while ``SpectrumModel`` historically uses
    direct-summation over the energy-levels table with IPD truncation. To
    preserve rtol=1e-5 parity for ``SpectrumModel`` we let the host inject
    the legacy populations. Manifold and Bayesian callers do NOT use this
    escape hatch -- they consume the canonical snapshot path.
    """
    if broadening_mode == BroadeningMode.LDM_GAUSSIAN and sigma_grid is None:
        raise ValueError("BroadeningMode.LDM_GAUSSIAN requires sigma_grid")

    wl = jnp.asarray(wavelength_grid)

    # ---- Populations (snapshot path) or injected (legacy parity path) ----
    if _precomputed_n_upper_per_line is None:
        n_upper = _saha_two_stage_populations(plasma_state, atomic_snapshot)
    else:
        n_upper = jnp.asarray(_precomputed_n_upper_per_line)

    # ---- Per-line emissivity epsilon_l = (hc / 4pi lambda_l) * A_ki * n_k ----
    line_wl_nm = jnp.asarray(atomic_snapshot.line_wavelengths_nm)
    line_A_ki = jnp.asarray(atomic_snapshot.line_A_ki)
    # Convert n_k from cm^-3 to m^-3 (legacy convention): n_k * 1e6.
    n_upper_m3 = n_upper * 1.0e6
    lambda_m = line_wl_nm * 1.0e-9
    epsilon_line = _HC_OVER_4PI / jnp.maximum(lambda_m, 1e-30) * line_A_ki * n_upper_m3

    # ---- Per-line broadening widths (mode-dispatched) ----
    T_eV = jnp.asarray(plasma_state.T_e_eV)
    n_e = jnp.asarray(plasma_state.n_e)

    if broadening_mode == BroadeningMode.LEGACY:
        # Legacy: single scalar sigma; outer-product Gaussian sum.
        sigma_scalar = 0.01 * jnp.sqrt(jnp.maximum(T_eV, 1e-12) / 0.86)
        sigma_per_line = jnp.full(line_wl_nm.shape, sigma_scalar, dtype=line_wl_nm.dtype)
        emissivity = _gaussian_sum_per_line(wl, line_wl_nm, epsilon_line, sigma_per_line)
    elif broadening_mode == BroadeningMode.NIST_PARITY:
        # Per-line instrument sigma from resolving power, evaluated at each
        # line center inline (spec §5 row "Bayesian approach wins").
        sigma_per_line = _per_line_instrument_sigma(atomic_snapshot, instrument)
        emissivity = _gaussian_sum_per_line(wl, line_wl_nm, epsilon_line, sigma_per_line)
    elif broadening_mode == BroadeningMode.PHYSICAL_DOPPLER:
        line_mass_amu = _species_mass_array(atomic_snapshot)[
            np.asarray(atomic_snapshot.line_species_index)
        ]
        sigma_doppler = _per_line_doppler_sigma(atomic_snapshot, T_eV, line_mass_amu)
        if fold_instrument_sigma:
            sigma_inst = _per_line_instrument_sigma(atomic_snapshot, instrument)
            sigma_per_line = jnp.sqrt(sigma_doppler**2 + sigma_inst**2)
        else:
            sigma_per_line = sigma_doppler
        if apply_stark:
            gamma_per_line = jnp.maximum(_per_line_stark_gamma(atomic_snapshot, n_e, T_eV), 1e-12)
            sigma_per_line = jnp.maximum(sigma_per_line, 1e-12)
            emissivity = _voigt_sum_per_line(
                wl, line_wl_nm, epsilon_line, sigma_per_line, gamma_per_line
            )
        else:
            emissivity = _gaussian_sum_per_line(wl, line_wl_nm, epsilon_line, sigma_per_line)
    elif broadening_mode == BroadeningMode.LDM_GAUSSIAN:
        from cflibs.radiation.ldm import ldm_broaden

        line_mass_amu = _species_mass_array(atomic_snapshot)[
            np.asarray(atomic_snapshot.line_species_index)
        ]
        sigma_per_line = jnp.maximum(
            _per_line_doppler_sigma(atomic_snapshot, T_eV, line_mass_amu), 1e-12
        )
        emissivity = ldm_broaden(
            line_wavelengths=line_wl_nm,
            line_intensities=epsilon_line,
            line_sigmas=sigma_per_line,
            wavelength_grid=wl,
            sigma_grid=jnp.asarray(sigma_grid),
        )
    else:
        raise ValueError(f"Unsupported broadening_mode: {broadening_mode!r}")

    if not apply_self_absorption:
        return emissivity

    # ---- Optical-slab self-absorption: I = B * (1 - exp(-kappa * L)) ----
    wl_m = wl * 1.0e-9
    T_K = T_eV * EV_TO_K
    exponent = (H_PLANCK * C_LIGHT) / (wl_m * KB * T_K)
    exponent = jnp.minimum(exponent, 700.0)
    B_m3 = (2.0 * H_PLANCK * C_LIGHT**2 / (wl_m**5)) / (jnp.exp(exponent) - 1.0)
    B_lambda = B_m3 * 1.0e-9  # W m^-2 nm^-1 sr^-1
    kappa = emissivity / (B_lambda + 1e-100)
    return B_lambda * (-jnp.expm1(-kappa * path_length_m))


__all__ = ["forward_model"]
