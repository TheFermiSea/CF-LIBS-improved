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

import os
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
from cflibs.core.jax_runtime import HAS_JAX, jnp
from cflibs.core.logging_config import get_logger
from cflibs.radiation.profiles import BroadeningMode
from cflibs.radiation.stark import REF_NE as _STARK_REF_NE

logger = get_logger("radiation.kernels")

if TYPE_CHECKING:  # pragma: no cover - imports for type checkers only
    from cflibs.core.jax_runtime import AtomicSnapshot
    from cflibs.instrument.model import InstrumentModel
    from cflibs.plasma.state import SingleZoneLTEPlasma

    # NOTE: ``ChunkPlan`` lives in :mod:`cflibs.radiation.host` and is used
    # only for the ``plan`` forward-ref annotation on
    # :func:`forward_model_chunked`. We deliberately do NOT import it here
    # because :mod:`cflibs.radiation.kernels` must remain free of any
    # ``cflibs.radiation.host`` imports (tests/test_jax_import_hygiene.py
    # ``test_kernels_modules_do_not_import_host``). Type checkers resolve
    # the forward reference via the ``Any`` alias below.

# Loose alias so ``Optional[ChunkPlan]`` resolves under both ruff (no F821)
# and mypy (no name-defined error) without re-introducing a host import.
# The real :class:`cflibs.radiation.host.ChunkPlan` is the only value
# callers pass; the typing payload is documented in the docstring.
ChunkPlan = object  # type: ignore[misc,assignment]

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


def _polynomial_partition_function_jax(T_K, coeffs, t_min=None, t_max=None, g0=None):
    """ln U = sum_n a_n (ln T_K)^n. Returns U on the active dtype.

    The Irwin (1981) partition-function coefficients shipped in the
    canonical atomic DB (see ``scripts/populate_partition_functions.py``)
    are NATURAL-log basis: ``ln U = sum a_n (ln T)^n``. The previous
    version of this helper used log10 + ``power(10, ...)`` which
    produced an ~18-orders-of-magnitude error (Fe I @ 10000 K: true
    U=33.84 vs bad U=1.95e+20). See CF-LIBS-improved-ddwh.

    ``coeffs`` has shape ``(N_species, 5)``; row index matches
    ``snapshot.species``. Mirrors
    :func:`cflibs.plasma.partition.polynomial_partition_function_jax`
    (the canonical implementation, which also uses ``jnp.log``) but
    inlined to avoid extra import overhead inside the jit.

    Optional ``t_min`` / ``t_max`` / ``g0`` arrays broadcast against
    ``coeffs`` and apply the arch-candidate-4 extrapolation guard:
    ``T`` is clamped to ``[t_min, t_max]`` before evaluation and the
    result is floored at ``g0``.  Passing ``None`` for all three
    preserves the legacy un-clamped behaviour.
    """
    T_clamped = T_K
    if t_min is not None and t_max is not None:
        T_clamped = jnp.clip(T_K, t_min, t_max)
    elif t_min is not None:
        T_clamped = jnp.maximum(T_K, t_min)
    elif t_max is not None:
        T_clamped = jnp.minimum(T_K, t_max)
    ln_T = jnp.log(jnp.maximum(T_clamped, 1.0))
    ln_T_powers = jnp.stack([jnp.ones_like(ln_T), ln_T, ln_T**2, ln_T**3, ln_T**4], axis=-1)
    ln_U = jnp.sum(coeffs * ln_T_powers, axis=-1)
    U = jnp.exp(ln_U)
    if g0 is not None:
        U = jnp.maximum(U, g0)
    return U


def _saha_species_index_maps(species_list):
    """Host-side (static) species/element index maps for the Saha block.

    Returns
    -------
    elements_in_snapshot : list[str]
        Distinct element symbols in first-seen order (padded rows excluded).
    species_to_element_idx : np.ndarray (N_species,) int32
        Per-species-row element index (sentinel 0 for padded rows).
    sp_idx_I, sp_idx_II : np.ndarray (N_elements,) int32
        Per-element species-row index for stage I / stage II (sentinel -1).
    stage_per_species : np.ndarray (N_species,) int32
        Ionization stage per species row.
    """
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
    return (
        elements_in_snapshot,
        species_to_element_idx,
        sp_idx_I,
        sp_idx_II,
        stage_per_species,
    )


def _saha_partition_functions(snapshot, T_K, dtype):
    """Per-species partition functions ``U`` (traced), floored at 1e-30.

    Per-species bounds + g0 routed through the encapsulated provider
    (arch candidate 4).  Older snapshots produced before the candidate-4
    rollout leave these fields as ``None`` — fall back to the legacy
    unclamped evaluation so we don't break callers that build their
    own snapshots without the bounds arrays.
    """
    pf_all = jnp.asarray(snapshot.partition_coeffs, dtype=dtype)
    if snapshot.partition_t_min is not None and snapshot.partition_t_max is not None:
        tmin_arr = jnp.asarray(snapshot.partition_t_min, dtype=dtype)
        tmax_arr = jnp.asarray(snapshot.partition_t_max, dtype=dtype)
        g0_arr = (
            jnp.asarray(snapshot.partition_g0, dtype=dtype)
            if snapshot.partition_g0 is not None
            else None
        )
        return jnp.maximum(
            _polynomial_partition_function_jax(T_K, pf_all, tmin_arr, tmax_arr, g0_arr),
            1e-30,
        )
    return jnp.maximum(_polynomial_partition_function_jax(T_K, pf_all), 1e-30)


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
    (
        elements_in_snapshot,
        species_to_element_idx,
        sp_idx_I,
        sp_idx_II,
        stage_per_species,
    ) = _saha_species_index_maps(species_list)
    n_elements = len(elements_in_snapshot)

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
    ip_all = jnp.asarray(snapshot.ionization_potential_ev, dtype=dtype)
    U_per_species = _saha_partition_functions(snapshot, T_K, dtype)

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


def _per_line_stark_gamma(snapshot, n_e, T_eV, disable_t_factor: Optional[bool] = None):
    """Per-line Lorentzian Stark HWHM (nm).

    ``snapshot.line_stark_w`` is the stored electron-impact **FWHM** at
    ``REF_NE = 1e17 cm^-3``, ``T_ref = 10000 K`` (see the convention note in
    ``cflibs/radiation/stark.py``). The Voigt profile wants a Lorentzian
    **HWHM**, so scale and halve::

        gamma_S(n_e, T) = 0.5 * stark_w * (n_e / 1e17) * (T_eV / 0.86173) ** (-alpha)

    ``T_ref = 0.86173 eV = 10000 K`` matches the Griem / NIST tabulation
    convention. Lines without catalogued temperature dependence carry
    ``alpha = 0.0`` in the snapshot (the :meth:`AtomicDatabase.snapshot`
    default for missing DB entries), which collapses ``factor_T`` to 1.0
    automatically — so this kernel needs no special-case for them.

    The earlier ``(n_e / 1e16)`` with no 0.5 over-broadened every Stark line
    by x20 at ps-LIBS densities (A4-CONV-2): x10 wrong reference density and
    x2 from treating an already-FWHM value as a HWHM.

    Bug history (CF-LIBS-improved-vjbh): the original T1-2 kernel omitted
    the temperature factor entirely, which silently dropped the
    BayesianForwardModel's Stark T-dependence after the T1-6 migration.

    Ablation toggle: ``disable_t_factor=True`` collapses ``factor_T`` to
    1.0, reproducing the pre-vjbh kernel for ablation / benchmark
    comparisons (CF-LIBS-improved-4rwe). Off by default. Callers thread it
    from :func:`forward_model` / :func:`forward_model_chunked`
    ``disable_stark_t_factor`` (constructor param on
    ``BayesianForwardModel``). Default ``None`` SEEDS the flag from the
    DEPRECATED ``CFLIBS_DISABLE_STARK_T_FACTOR=1`` env var (with a warning)
    so existing ablation scripts keep working; an explicit ``True``/``False``
    is authoritative. The check is host-side: it resolves at jit-trace time
    so the toggled kernel keeps a separate jit cache key from the default
    one and runtime cost is zero in either branch.
    """
    if disable_t_factor is None:
        disable_t_factor = os.environ.get("CFLIBS_DISABLE_STARK_T_FACTOR", "0") == "1"
        if disable_t_factor:
            logger.warning(
                "CFLIBS_DISABLE_STARK_T_FACTOR=1 (env var) is deprecated; pass "
                "disable_stark_t_factor=True explicitly (e.g. "
                "forward_model(..., disable_stark_t_factor=True) or "
                "BayesianForwardModel(disable_stark_t_factor=True))."
            )
    stark_w = jnp.asarray(snapshot.line_stark_w)
    # stark_w is FWHM at REF_NE=1e17; 0.5 * (n_e/1e17) converts to a HWHM at
    # the live density (single source of truth: cflibs.radiation.stark.REF_NE).
    if disable_t_factor:
        return 0.5 * stark_w * (n_e / _STARK_REF_NE)
    alpha = jnp.asarray(snapshot.line_stark_alpha)
    REF_T_EV = 0.86173  # 10000 K — Griem / NIST reference temperature.
    factor_T = jnp.power(jnp.maximum(T_eV, 0.1) / REF_T_EV, -alpha)
    return 0.5 * stark_w * (n_e / _STARK_REF_NE) * factor_T


def _per_line_stark_shift(snapshot, line_wl_nm, n_e):
    """Per-line Stark-shifted line centers (nm).

    ``snapshot.line_stark_d`` is the stored signed Stark **shift** of the
    line center at ``REF_NE = 1e17 cm^-3`` (nm). The shift scales **linearly**
    with electron density (single source of truth:
    :func:`cflibs.radiation.stark.stark_shift`)::

        delta_lambda = line_stark_d * (n_e / REF_NE)
        lambda_c     = lambda_0 + delta_lambda

    Ion broadening (Griem A-term, R_D) is intentionally **not** applied: at
    ps-LIBS densities (n_e <= 1e17 cm^-3) it is a <2-5% correction and is
    dropped following John et al. 2023 and Stetzler et al. 2020. Only the
    Stark shift — flagged in the corpus as a real line-identification error
    source (Noel 2025) — is carried here.

    Lines without a catalogued shift carry ``line_stark_d = 0.0`` (the
    :meth:`AtomicDatabase.snapshot` default), so they are left unmoved. A
    snapshot with ``line_stark_d is None`` (built before the shift rollout)
    is also treated as "no shift" — the original centers are returned
    unchanged.
    """
    stark_d = getattr(snapshot, "line_stark_d", None)
    if stark_d is None:
        return line_wl_nm
    stark_d = jnp.asarray(stark_d)
    return line_wl_nm + stark_d * (n_e / _STARK_REF_NE)


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


def _apply_self_absorption(emissivity, wl, T_eV, path_length_m):
    """Optical-slab self-absorption: ``I = B * (1 - exp(-kappa * L))``.

    Numerically identical to the inline radiative-transfer block shared by
    :func:`forward_model` and the :func:`forward_model_chunked` scan body.
    """
    wl_m = wl * 1.0e-9
    T_K = T_eV * EV_TO_K
    exponent = (H_PLANCK * C_LIGHT) / (wl_m * KB * T_K)
    exponent = jnp.minimum(exponent, 700.0)
    B_m3 = (2.0 * H_PLANCK * C_LIGHT**2 / (wl_m**5)) / (jnp.exp(exponent) - 1.0)
    B_lambda = B_m3 * 1.0e-9  # W m^-2 nm^-1 sr^-1
    kappa = emissivity / (B_lambda + 1e-100)
    return B_lambda * (-jnp.expm1(-kappa * path_length_m))


def _forward_emissivity(
    wl,
    line_centers,
    epsilon_line,
    atomic_snapshot,
    instrument,
    T_eV,
    n_e,
    sigma_grid,
    *,
    broadening_mode: BroadeningMode,
    fold_instrument_sigma: bool,
    apply_stark: bool,
    disable_stark_t_factor: Optional[bool] = None,
):
    """Mode-dispatched per-line broadening for :func:`forward_model`.

    Returns the broadened emissivity on ``wl``. Splits the static
    broadening-mode dispatch out of :func:`forward_model` verbatim — every
    width formula and conditional matches the pre-extraction inline body.
    """
    if broadening_mode == BroadeningMode.LEGACY:
        # Legacy: single scalar sigma; outer-product Gaussian sum.
        sigma_scalar = 0.01 * jnp.sqrt(jnp.maximum(T_eV, 1e-12) / 0.86)
        line_wl_nm = jnp.asarray(atomic_snapshot.line_wavelengths_nm)
        sigma_per_line = jnp.full(line_wl_nm.shape, sigma_scalar, dtype=line_wl_nm.dtype)
        return _gaussian_sum_per_line(wl, line_centers, epsilon_line, sigma_per_line)
    if broadening_mode == BroadeningMode.NIST_PARITY:
        # Per-line instrument sigma from resolving power, evaluated at each
        # line center inline (spec §5 row "Bayesian approach wins").
        sigma_per_line = _per_line_instrument_sigma(atomic_snapshot, instrument)
        return _gaussian_sum_per_line(wl, line_centers, epsilon_line, sigma_per_line)
    if broadening_mode == BroadeningMode.PHYSICAL_DOPPLER:
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
            gamma_per_line = jnp.maximum(
                _per_line_stark_gamma(atomic_snapshot, n_e, T_eV, disable_stark_t_factor), 1e-12
            )
            sigma_per_line = jnp.maximum(sigma_per_line, 1e-12)
            return _voigt_sum_per_line(
                wl, line_centers, epsilon_line, sigma_per_line, gamma_per_line
            )
        return _gaussian_sum_per_line(wl, line_centers, epsilon_line, sigma_per_line)
    if broadening_mode == BroadeningMode.LDM_GAUSSIAN:
        from cflibs.radiation.ldm import ldm_broaden

        line_mass_amu = _species_mass_array(atomic_snapshot)[
            np.asarray(atomic_snapshot.line_species_index)
        ]
        sigma_per_line = jnp.maximum(
            _per_line_doppler_sigma(atomic_snapshot, T_eV, line_mass_amu), 1e-12
        )
        return ldm_broaden(
            line_wavelengths=line_centers,
            line_intensities=epsilon_line,
            line_sigmas=sigma_per_line,
            wavelength_grid=wl,
            sigma_grid=jnp.asarray(sigma_grid),
        )
    raise ValueError(f"Unsupported broadening_mode: {broadening_mode!r}")


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
    line_mask=None,
    disable_stark_t_factor: Optional[bool] = None,
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
    disable_stark_t_factor : bool, optional
        Ablation toggle: collapse the Stark temperature factor
        ``(T/T_ref)^(-alpha)`` to 1.0 (pre-vjbh kernel; see
        :func:`_per_line_stark_gamma`). Default ``None`` seeds from the
        DEPRECATED ``CFLIBS_DISABLE_STARK_T_FACTOR`` env var; an explicit
        value is authoritative. Host-side static — resolved at jit-trace
        time.
    line_mask : array, optional, shape (N_lines,)
        Per-line activation mask. ``None`` (default) ⇒ all lines contribute
        and the call is bit-identical to the pre-T1-5 behaviour. When
        supplied, the mask is cast to the working dtype and multiplied into
        the per-line emissivity just before broadening dispatch — masked
        lines (mask=0) contribute zero, but the ``(N_lines,)`` shape is
        preserved so this kernel stays trace-shape-stable across the
        :func:`forward_model_chunked` ``lax.scan`` body.
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

    # ---- Optional per-line activation mask (q278) ----
    # The mask is cast to the emissivity dtype so this is a single fused
    # multiply on accelerators. ``None`` is bit-identical to the pre-mask
    # behaviour (zero ops emitted).
    if line_mask is not None:
        mask = jnp.asarray(line_mask).astype(epsilon_line.dtype)
        epsilon_line = epsilon_line * mask

    # ---- Per-line broadening widths (mode-dispatched) ----
    T_eV = jnp.asarray(plasma_state.T_e_eV)
    n_e = jnp.asarray(plasma_state.n_e)

    # Stark line-center shift (linear in n_e). Applied to the profile centers
    # only — the hc/4pi*lambda emissivity prefactor keeps the unshifted
    # transition wavelength. Off when apply_stark is False (returns lambda_0).
    line_centers = (
        _per_line_stark_shift(atomic_snapshot, line_wl_nm, n_e) if apply_stark else line_wl_nm
    )

    emissivity = _forward_emissivity(
        wl,
        line_centers,
        epsilon_line,
        atomic_snapshot,
        instrument,
        T_eV,
        n_e,
        sigma_grid,
        broadening_mode=broadening_mode,
        fold_instrument_sigma=fold_instrument_sigma,
        apply_stark=apply_stark,
        disable_stark_t_factor=disable_stark_t_factor,
    )

    if not apply_self_absorption:
        return emissivity

    # ---- Optical-slab self-absorption: I = B * (1 - exp(-kappa * L)) ----
    return _apply_self_absorption(emissivity, wl, T_eV, path_length_m)


# ---------------------------------------------------------------------------
# Chunked forward model (T1-5) — overlap-and-add over wavelength axis
# ---------------------------------------------------------------------------


def overlap_and_add(
    partials,
    *,
    overlap: int,
    output_length: int,
):
    """Recombine ``nstitch`` per-chunk spectra via overlap-and-add (OLA).

    Mirrors :func:`exojax.signal.ola.overlap_and_add_matrix` (ADR-0001
    §3, §4). Place each chunk at offset ``chunk_idx * div_length - overlap``
    in an output buffer of length ``output_length + 2*overlap``, sum where
    chunks overlap, then trim the wing padding.

    Parameters
    ----------
    partials : array, shape (nstitch, div_length + 2·overlap)
        Per-chunk spectra emitted by the scan body. The first ``overlap``
        and last ``overlap`` samples of each chunk are the wing padding
        that needs to be summed with the neighbouring chunk.
    overlap : int, **static**
        Per-side wing padding in samples; must match the value used by
        :func:`_split_wavelength_grid`.
    output_length : int, **static**
        Length of the original (un-chunked) wavelength axis. Output is
        trimmed to this length.

    Returns
    -------
    spectrum : array, shape (output_length,)
        Stitched spectrum with wing contributions summed.

    Notes
    -----
    Implementation uses :func:`jax.vmap` over
    :func:`jax.lax.dynamic_update_slice` to place each chunk into its own
    zero-padded copy of the output buffer in parallel, then sums across
    the chunk axis (8e2o). On GPU this fuses into a single kernel instead
    of one ``dynamic_update_slice`` launch per chunk in the previous
    ``lax.fori_loop`` form. The ``+ 2*overlap`` padding on the output
    buffer is sliced off before return.
    """
    partials = jnp.asarray(partials)
    nstitch = partials.shape[0]
    chunk_length = partials.shape[1]
    div_length = chunk_length - 2 * overlap

    # Buffer must accommodate every chunk's full footprint, even when the
    # last chunk's interior overshoots ``output_length`` (the pad-and-mask
    # case in ``host._split_wavelength_grid`` when ``N_λ % nstitch != 0``).
    # Total span = (nstitch-1)·div_length + chunk_length, which collapses
    # to ``nstitch·div_length + 2·overlap``. We allocate that many samples
    # and slice the leading ``overlap`` plus the trailing padding at the
    # end.
    buf_length = nstitch * div_length + 2 * overlap

    if not HAS_JAX:  # pragma: no cover — fallback for non-JAX environments
        buf = np.zeros(buf_length, dtype=np.asarray(partials).dtype)
        partials_np = np.asarray(partials)
        for c in range(nstitch):
            start = c * div_length
            buf[start : start + chunk_length] += partials_np[c]
        return buf[overlap : overlap + output_length]

    import jax  # noqa: PLC0415 — JAX is mandatory on this path

    dtype = partials.dtype
    zero_buf = jnp.zeros(buf_length, dtype=dtype)

    # Place each chunk into its own zero-buffer at offset ``c * div_length``
    # via vmap → single fused kernel on accelerators (8e2o). The sum across
    # the leading axis is numerically identical to the previous fori_loop
    # accumulator at default fp64 tolerances; chunk overlaps are summed by
    # the ``.sum(0)``.
    starts = jnp.arange(nstitch, dtype=jnp.int32) * jnp.int32(div_length)

    def _place_chunk(start, chunk):
        return jax.lax.dynamic_update_slice(zero_buf, chunk, (start,))

    placed = jax.vmap(_place_chunk)(starts, partials)
    buf = placed.sum(axis=0)
    return jax.lax.dynamic_slice(buf, (overlap,), (output_length,))


def _broaden_chunk(
    wl,
    line_wl_nm,
    epsilon_line_masked,
    sigma_per_line,
    gamma_per_line,
    *,
    broadening_mode: BroadeningMode,
    sigma_grid,
    apply_stark: bool,
):
    """Broadening dispatch on a single chunk with pre-computed per-line tensors.

    Helper used by :func:`forward_model_chunked` after the chunk-invariant
    Saha-Boltzmann populations and per-line widths have been hoisted out of
    the scan body (q278). The mask multiply on ``epsilon_line`` is assumed
    to have already been applied by the caller.

    All parameters mirror their counterparts in :func:`forward_model`;
    ``gamma_per_line`` is ignored unless
    ``broadening_mode == PHYSICAL_DOPPLER`` with ``apply_stark=True``.
    """
    if broadening_mode == BroadeningMode.LEGACY:
        return _gaussian_sum_per_line(wl, line_wl_nm, epsilon_line_masked, sigma_per_line)
    if broadening_mode == BroadeningMode.NIST_PARITY:
        return _gaussian_sum_per_line(wl, line_wl_nm, epsilon_line_masked, sigma_per_line)
    if broadening_mode == BroadeningMode.PHYSICAL_DOPPLER:
        if apply_stark:
            return _voigt_sum_per_line(
                wl, line_wl_nm, epsilon_line_masked, sigma_per_line, gamma_per_line
            )
        return _gaussian_sum_per_line(wl, line_wl_nm, epsilon_line_masked, sigma_per_line)
    if broadening_mode == BroadeningMode.LDM_GAUSSIAN:
        from cflibs.radiation.ldm import ldm_broaden  # noqa: PLC0415

        return ldm_broaden(
            line_wavelengths=line_wl_nm,
            line_intensities=epsilon_line_masked,
            line_sigmas=sigma_per_line,
            wavelength_grid=wl,
            sigma_grid=jnp.asarray(sigma_grid),
        )
    raise ValueError(f"Unsupported broadening_mode: {broadening_mode!r}")


def _resolve_checkpoint_policy():
    """Return ``jax.checkpoint_policies.dots_with_no_batch_dims_saveable``.

    Falls back to ``everything_saveable`` (slower; correct) when the
    fine-grained policy is missing from the installed JAX (spec §10
    risks). Returns ``None`` when JAX itself is unavailable so the
    chunked path silently degrades to a plain Python loop.
    """
    if not HAS_JAX:  # pragma: no cover
        return None
    import jax  # noqa: PLC0415

    policies = getattr(jax, "checkpoint_policies", None)
    if policies is None:  # very old JAX without checkpoint_policies
        return None
    return (
        getattr(policies, "dots_with_no_batch_dims_saveable", None)
        # NOTE: fallback per spec §10 when the fine-grained policy is
        # missing. ``everything_saveable`` is slower but always correct.
        or getattr(policies, "everything_saveable", None)
    )


def _chunk_per_line_widths(
    atomic_snapshot,
    instrument,
    line_wl_nm,
    T_eV,
    n_e,
    *,
    broadening_mode: BroadeningMode,
    fold_instrument_sigma: bool,
    apply_stark: bool,
    disable_stark_t_factor: Optional[bool] = None,
):
    """Chunk-invariant per-line broadening widths for the chunked scan.

    Hoisted verbatim from :func:`forward_model_chunked`'s pre-scan block
    (q278). Returns ``(sigma_per_line, gamma_per_line)`` where
    ``gamma_per_line`` is ``None`` unless ``PHYSICAL_DOPPLER`` with
    ``apply_stark=True``.
    """
    gamma_per_line = None
    if broadening_mode == BroadeningMode.LEGACY:
        sigma_scalar = 0.01 * jnp.sqrt(jnp.maximum(T_eV, 1e-12) / 0.86)
        sigma_per_line = jnp.full(line_wl_nm.shape, sigma_scalar, dtype=line_wl_nm.dtype)
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
            gamma_per_line = jnp.maximum(
                _per_line_stark_gamma(atomic_snapshot, n_e, T_eV, disable_stark_t_factor), 1e-12
            )
            sigma_per_line = jnp.maximum(sigma_per_line, 1e-12)
    elif broadening_mode == BroadeningMode.LDM_GAUSSIAN:
        line_mass_amu = _species_mass_array(atomic_snapshot)[
            np.asarray(atomic_snapshot.line_species_index)
        ]
        sigma_per_line = jnp.maximum(
            _per_line_doppler_sigma(atomic_snapshot, T_eV, line_mass_amu), 1e-12
        )
    else:  # pragma: no cover - NIST_PARITY rejected above; defensive
        sigma_per_line = _per_line_instrument_sigma(atomic_snapshot, instrument)
    return sigma_per_line, gamma_per_line


def _validate_chunked_inputs(
    broadening_mode,
    chunk_wavelength_grids,
    line_masks,
    output_length,
):
    """Validate the ``nstitch > 1`` chunked-path preconditions.

    Raises the same errors, in the same order, as the inline guards in
    :func:`forward_model_chunked`.
    """
    if broadening_mode == BroadeningMode.NIST_PARITY:
        # Spec §7: lines too sparse for chunked OLA to help.
        raise ValueError(
            "BroadeningMode.NIST_PARITY does not support chunked scan; "
            "lines are sparse and per-line Voigt is already cheap. Pass "
            "nstitch=1 or switch to PHYSICAL_DOPPLER / LDM_GAUSSIAN."
        )

    if chunk_wavelength_grids is None or line_masks is None or output_length is None:
        raise ValueError(
            "forward_model_chunked with nstitch>1 requires "
            "chunk_wavelength_grids, line_masks, and output_length. "
            "Use cflibs.radiation.host.build_chunk_metadata to construct them."
        )

    if not HAS_JAX:
        # forward_model_chunked is JAX-only by design: _forward_model_per_chunk
        # uses jnp ops throughout, which would explode against the numpy
        # fallback in cflibs.core.jax_runtime.jnp. The un-chunked forward_model
        # is similarly JAX-only; nstitch=1 dispatch above is the supported
        # no-JAX path.
        raise RuntimeError(
            "forward_model_chunked requires JAX; nstitch=1 dispatches to "
            "forward_model which is also JAX-required."
        )


def _forward_model_per_chunk(
    plasma_state,
    atomic_snapshot,
    instrument,
    chunk_wavelength_grid,
    sigma_grid,
    line_mask,
    *,
    broadening_mode: BroadeningMode,
    path_length_m: float,
    apply_self_absorption: bool,
    fold_instrument_sigma: bool,
    apply_stark: bool,
):
    """Forward model on one wavelength chunk with a per-line activation mask.

    Thin wrapper around :func:`forward_model` that forwards the per-chunk
    activation mask. Kept as a stable internal entry point for
    :func:`forward_model_chunked` (and any out-of-tree users) — the masking
    behaviour now lives in :func:`forward_model` itself (q278), so this
    function exists only to preserve the historical signature.
    """
    return forward_model(
        plasma_state,
        atomic_snapshot,
        instrument,
        chunk_wavelength_grid,
        sigma_grid=sigma_grid,
        broadening_mode=broadening_mode,
        path_length_m=path_length_m,
        apply_self_absorption=apply_self_absorption,
        fold_instrument_sigma=fold_instrument_sigma,
        apply_stark=apply_stark,
        line_mask=line_mask,
    )


def forward_model_chunked(
    plasma_state: "SingleZoneLTEPlasma",
    atomic_snapshot: "AtomicSnapshot",
    instrument: "InstrumentModel",
    wavelength_grid,
    sigma_grid=None,
    *,
    plan: Optional[ChunkPlan] = None,
    nstitch: int = 1,
    overlap: int = 0,
    chunk_wavelength_grids=None,
    line_masks=None,
    broadening_mode: BroadeningMode = BroadeningMode.PHYSICAL_DOPPLER,
    path_length_m: float = 0.01,
    apply_self_absorption: bool = False,
    fold_instrument_sigma: bool = True,
    apply_stark: bool = False,
    disable_stark_t_factor: Optional[bool] = None,
    output_length: Optional[int] = None,
):
    """Chunked forward model over the wavelength axis (ADR-0001 T1-5).

    Splits the wavelength grid into ``nstitch`` chunks, scans
    :func:`_forward_model_per_chunk` over the chunks under
    :func:`jax.checkpoint`, and recombines the partials via
    :func:`overlap_and_add`. Cuts peak transient memory by a factor of
    ``nstitch`` and cuts backward-pass activation memory roughly in half
    (spec §8 AC#4).

    When ``nstitch == 1`` the function dispatches directly to
    :func:`forward_model` — bit-identical to the un-chunked path so
    existing manifold checksums are preserved (spec §6, §10 rollback).

    Parameters
    ----------
    plasma_state, atomic_snapshot, instrument, wavelength_grid, sigma_grid :
        Same semantics as :func:`forward_model`.
    plan : ChunkPlan, optional
        Frozen :class:`cflibs.radiation.host.ChunkPlan` bundling the five
        always-paired chunk metadata fields (``nstitch``, ``overlap``,
        ``chunk_wavelength_grids``, ``line_masks``, ``output_length``).
        When supplied, takes precedence over the individual keyword
        arguments. When ``None`` (default), the explicit kwargs are used
        — preserved for back-compat with pre-a2m2 callers.
    nstitch : int, **static**, default 1
        Number of wavelength chunks. ``1`` ⇒ direct dispatch to
        :func:`forward_model`.
    overlap : int, **static**, default 0
        Per-side wing padding in samples. Spec §5 recommends
        ``overlap = ceil(overlap_factor · max(σ_grid) / Δλ)`` with
        ``overlap_factor = 4.0``. ``test_overlap_factor`` is the canary.
    chunk_wavelength_grids : array, shape (nstitch, div_length + 2·overlap), optional
        Pre-built padded chunk wavelength grids. When ``None`` and
        ``nstitch > 1`` the caller must pass the metadata explicitly —
        this kernel does not call :func:`host._split_wavelength_grid`
        itself to keep the jit boundary clean.
    line_masks : array of bool/float, shape (nstitch, N_lines), optional
        Per-chunk line activation masks; ``True`` ⇒ line contributes.
        Pair with ``chunk_wavelength_grids``.
    broadening_mode : BroadeningMode, **static**
        Static dispatch knob. :attr:`BroadeningMode.NIST_PARITY` rejects
        ``nstitch > 1`` per spec §7 — lines are sparse, per-line Voigt is
        already cheap, and the spec calls for the un-chunked path.
    path_length_m : float
        Plasma path length (m). Only consulted when
        ``apply_self_absorption``.
    apply_self_absorption : bool, default False
        Self-absorption switch — same as :func:`forward_model`.
    fold_instrument_sigma, apply_stark : bool
        See :func:`forward_model`.
    disable_stark_t_factor : bool, optional
        See :func:`forward_model`.
    output_length : int, optional
        Length of the original wavelength grid. Required when
        ``nstitch > 1``; ignored otherwise. The chunked spectrum is
        trimmed to this length after OLA recombination.

    Returns
    -------
    intensity : array, shape (output_length,) or (N_wl,)
        Stitched spectrum on the original wavelength grid.

    Raises
    ------
    ValueError
        - ``broadening_mode == NIST_PARITY`` with ``nstitch > 1`` (spec §7).
        - ``nstitch > 1`` without ``chunk_wavelength_grids`` /
          ``line_masks`` / ``output_length``.

    Notes
    -----
    The scan body is wrapped in :func:`jax.checkpoint` with policy
    ``dots_with_no_batch_dims_saveable`` (spec §3) — recomputes the
    cheap per-line scalar work on the backward pass while saving the
    expensive matrix dots. Falls back to ``everything_saveable`` on
    older JAX builds (see :func:`_resolve_checkpoint_policy`).
    """
    if plan is not None:
        # ``plan`` takes precedence over the individual kwargs (a2m2). We
        # do not silently merge so that callers can spot accidental
        # mixed-mode invocations (an explicit pre-a2m2 kwarg shadowing a
        # plan field is almost always a bug).
        nstitch = plan.nstitch
        overlap = plan.overlap
        chunk_wavelength_grids = plan.chunk_wavelength_grids
        line_masks = plan.line_masks
        output_length = plan.output_length

    if nstitch == 1:
        # Zero-cost dispatch — caller pays nothing for the chunked path
        # when they have not opted in. Bit-identical to ``forward_model``.
        return forward_model(
            plasma_state,
            atomic_snapshot,
            instrument,
            wavelength_grid,
            sigma_grid=sigma_grid,
            broadening_mode=broadening_mode,
            path_length_m=path_length_m,
            apply_self_absorption=apply_self_absorption,
            fold_instrument_sigma=fold_instrument_sigma,
            apply_stark=apply_stark,
            disable_stark_t_factor=disable_stark_t_factor,
        )

    _validate_chunked_inputs(broadening_mode, chunk_wavelength_grids, line_masks, output_length)

    import jax  # noqa: PLC0415

    chunks = jnp.asarray(chunk_wavelength_grids)
    masks_dtype = chunks.dtype
    masks = jnp.asarray(line_masks).astype(masks_dtype)

    # ---- q278: hoist chunk-invariant work out of the scan body ----
    # Saha-Boltzmann populations, the un-masked per-line emissivity, and
    # the per-line broadening widths (Doppler + instrument; optional Stark
    # gamma) depend only on plasma_state / snapshot / instrument — NOT on
    # the chunk. Computing them once before ``lax.scan`` removes a 20–40%
    # wall-time tax that the previous implementation paid per chunk
    # iteration. The scan body now only multiplies in the chunk mask and
    # dispatches the broadening kernel against the chunk's wavelengths.
    n_upper = _saha_two_stage_populations(plasma_state, atomic_snapshot)
    line_wl_nm = jnp.asarray(atomic_snapshot.line_wavelengths_nm)
    line_A_ki = jnp.asarray(atomic_snapshot.line_A_ki)
    n_upper_m3 = n_upper * 1.0e6
    lambda_m = line_wl_nm * 1.0e-9
    epsilon_line_base = _HC_OVER_4PI / jnp.maximum(lambda_m, 1e-30) * line_A_ki * n_upper_m3

    T_eV = jnp.asarray(plasma_state.T_e_eV)
    n_e = jnp.asarray(plasma_state.n_e)

    # Stark line-center shift (linear in n_e), profile-centers only. Hoisted
    # out of the scan body since it depends only on snapshot / n_e, not the
    # chunk. Off (== lambda_0) when apply_stark is False.
    line_centers = (
        _per_line_stark_shift(atomic_snapshot, line_wl_nm, n_e) if apply_stark else line_wl_nm
    )

    sigma_per_line, gamma_per_line = _chunk_per_line_widths(
        atomic_snapshot,
        instrument,
        line_wl_nm,
        T_eV,
        n_e,
        broadening_mode=broadening_mode,
        fold_instrument_sigma=fold_instrument_sigma,
        apply_stark=apply_stark,
        disable_stark_t_factor=disable_stark_t_factor,
    )

    policy = _resolve_checkpoint_policy()

    def _body_uncheckpointed(chunk_wl, chunk_mask):
        # Mask only — every other tensor is closed over from the hoist.
        epsilon_masked = epsilon_line_base * chunk_mask.astype(epsilon_line_base.dtype)
        emissivity = _broaden_chunk(
            chunk_wl,
            line_centers,
            epsilon_masked,
            sigma_per_line,
            gamma_per_line,
            broadening_mode=broadening_mode,
            sigma_grid=sigma_grid,
            apply_stark=apply_stark,
        )
        if not apply_self_absorption:
            return emissivity
        return _apply_self_absorption(emissivity, chunk_wl, T_eV, path_length_m)

    if policy is not None:
        body = jax.checkpoint(_body_uncheckpointed, policy=policy)
    else:  # pragma: no cover - very old JAX
        body = _body_uncheckpointed

    def _scan_step(carry, inputs):
        chunk_wl, chunk_mask = inputs
        partial = body(chunk_wl, chunk_mask)
        return carry, partial

    _, partials = jax.lax.scan(_scan_step, None, (chunks, masks))
    return overlap_and_add(partials, overlap=overlap, output_length=output_length)


__all__ = ["forward_model", "forward_model_chunked", "overlap_and_add"]
