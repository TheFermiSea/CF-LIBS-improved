"""Atomic-data carriers and database helpers for the Bayesian forward model (T1-6).

Hosts the JAX-array carrier :class:`AtomicDataArrays` (with the
``from_snapshot`` bridge to T1-1 :class:`AtomicSnapshot`), the SQLite query
helpers (:func:`_query_atomic_data`, :func:`_format_atomic_data_arrays`,
:func:`load_atomic_data`), the guarded :func:`partition_function` delegator
(routes to the ONE shared ``polynomial_partition_function_jax``), the
:func:`mcwhirter_log_penalty` soft LTE penalty, and a handful of JAX-real
casting helpers shared across the Bayesian sub-package.

The ``[t_min, t_max]`` validity window and ground-state ``g0`` floor for every
species' partition function are loaded from the ONE factory
(:meth:`cflibs.atomic.AtomicDatabase.partition_spec_for`) and baked into
:class:`AtomicDataArrays` as static ``(n_elements, n_stages)`` guard arrays, so
the guarded evaluator clamps/floors ``U(T)`` inside the NUTS jit trace without
any Python provider call (PF-3/PF-4 / the 2026-06-03 partition-provider
unification).

Separating this from :mod:`forward` keeps each file under the 800-LOC limit
required by ADR-0001 / T1-6 spec section 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cflibs.core.constants import (
    C_LIGHT,
    EV_TO_J,
    EV_TO_K,
    E_CHARGE,
    H_PLANCK,
    M_E,
    M_PROTON,
    MCWHIRTER_CONST,
    SAHA_CONST_CM3,
)
from cflibs.core.jax_runtime import jax_default_real_dtype
from cflibs.core.logging_config import get_logger

from .priors import HAS_JAX

logger = get_logger("inversion.bayesian.atomic")


# ---------------------------------------------------------------------------
# Optional JAX gate (mirrors legacy module-level imports)
# ---------------------------------------------------------------------------

if HAS_JAX:
    import jax.numpy as jnp
    from jax import jit  # noqa: F401  (re-exported)
else:  # pragma: no cover - JAX not installed
    jnp = None  # type: ignore[assignment]

    def jit(f):  # type: ignore[no-redef]
        return f


# ---------------------------------------------------------------------------
# JAX-real casting helpers
# ---------------------------------------------------------------------------

if HAS_JAX:
    _JAX_REAL_DTYPE = jax_default_real_dtype()

    def _as_jax_real(value: Any) -> Any:
        """Cast scalars and arrays to the active JAX real dtype."""
        return jnp.asarray(value, dtype=_JAX_REAL_DTYPE)

    _JAX_SAHA_CONST_CM3 = _as_jax_real(SAHA_CONST_CM3)
    _JAX_C_LIGHT = _as_jax_real(C_LIGHT)
    _JAX_EV_TO_K = _as_jax_real(EV_TO_K)
    _JAX_EV_TO_J = _as_jax_real(EV_TO_J)
    _JAX_MCWHIRTER_CONST = _as_jax_real(MCWHIRTER_CONST)
    _JAX_H_PLANCK = _as_jax_real(H_PLANCK)
    _JAX_E_CHARGE = _as_jax_real(E_CHARGE)
    _JAX_M_E = _as_jax_real(M_E)
    _JAX_M_PROTON = _as_jax_real(M_PROTON)
else:  # pragma: no cover - JAX not installed
    _JAX_REAL_DTYPE = None

    def _as_jax_real(value: Any) -> Any:  # type: ignore[no-redef]
        return value

    _JAX_SAHA_CONST_CM3 = SAHA_CONST_CM3
    _JAX_C_LIGHT = C_LIGHT
    _JAX_EV_TO_K = EV_TO_K
    _JAX_EV_TO_J = EV_TO_J
    _JAX_MCWHIRTER_CONST = MCWHIRTER_CONST
    _JAX_H_PLANCK = H_PLANCK
    _JAX_E_CHARGE = E_CHARGE
    _JAX_M_E = M_E
    _JAX_M_PROTON = M_PROTON


# Standard atomic masses for fallback [amu]
STANDARD_MASSES: Dict[str, float] = {
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
}


def _resolve_total_species_density_cm3(
    n_e: float,
    total_species_density_cm3: Optional[float],
) -> float:
    """Resolve heavy-particle density for forward models.

    When no explicit heavy-particle density is provided, preserve the legacy
    behaviour that approximates it with ``n_e``.
    """
    if total_species_density_cm3 is None:
        return n_e
    if np.isscalar(total_species_density_cm3):
        resolved = float(total_species_density_cm3)
        if resolved <= 0.0:
            raise ValueError("total_species_density_cm3 must be positive")
        return resolved
    if HAS_JAX:
        resolved = _as_jax_real(total_species_density_cm3)
        return jnp.where(resolved > 0.0, resolved, jnp.nan)
    raise ValueError("total_species_density_cm3 must be a positive scalar")


def _compute_instrument_sigma(
    line_wavelengths_nm,
    instrument_fwhm_nm: float,
    resolving_power: Optional[float],
):
    """Compute per-line instrumental Gaussian sigma.

    Two modes:
    - Constant FWHM (Czerny-Turner): ``sigma = FWHM / 2.355`` (scalar).
    - Constant resolving power (Echelle): ``sigma(lambda) = lambda / (R * 2.355)``
      (per-line array).
    """
    if resolving_power is not None:
        if resolving_power <= 0:
            raise ValueError(f"resolving_power must be positive, got {resolving_power}")
        return line_wavelengths_nm / (resolving_power * 2.355)
    return instrument_fwhm_nm / 2.355


# ---------------------------------------------------------------------------
# Atomic database query helpers
# ---------------------------------------------------------------------------


def _build_lines_dataframe(
    conn: Any,
    elements: List[str],
    wavelength_range: Tuple[float, float],
    placeholders: str,
) -> Tuple[Any, Dict[str, int]]:
    """Read the per-line atomic dataframe and attach element/mass columns.

    Returns the populated dataframe and the ``element -> index`` map.
    """
    import pandas as pd

    query = f"""
            SELECT
                l.id, l.element, l.sp_num, l.wavelength_nm, l.aki, l.ek_ev, l.gk,
                sp.ip_ev, l.stark_w, l.stark_alpha, l.ei_ev
            FROM lines l
            JOIN species_physics sp ON l.element = sp.element AND l.sp_num = sp.sp_num
            WHERE l.wavelength_nm BETWEEN ? AND ?
            AND l.element IN ({placeholders})
            ORDER BY l.wavelength_nm, l.id
        """
    params = [wavelength_range[0], wavelength_range[1]] + list(elements)
    df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        raise ValueError(f"No atomic data for elements {elements} in range {wavelength_range}")

    el_map = {el: i for i, el in enumerate(elements)}
    df["el_idx"] = df["element"].map(el_map)

    element_masses = {}
    for el in elements:
        if el in STANDARD_MASSES:
            element_masses[el] = STANDARD_MASSES[el]
        else:
            element_masses[el] = 50.0
            logger.warning(f"No mass for {el}, using fallback 50 amu")
    df["mass_amu"] = df["element"].map(element_masses)

    return df, el_map


def _init_partition_arrays(
    n_elements: int,
    max_stages: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the default partition coeff/ip and ``[t_min, t_max]`` + ``g0`` guards.

    The validity window matches the canonical ``[2000, 25000] K`` fit band; ``g0``
    defaults to the stage's placeholder ground-state weight.  These are the
    *static* arrays the guarded JAX evaluator clamps / floors with, so no Python
    provider call is needed inside the trace.
    """
    coeffs = np.zeros((n_elements, max_stages, 5), dtype=np.float32)
    ips = np.zeros((n_elements, max_stages), dtype=np.float32)

    coeffs[:, 0, 0] = np.log(25.0)
    coeffs[:, 1, 0] = np.log(15.0)
    coeffs[:, 2, 0] = np.log(10.0)

    t_min = np.full((n_elements, max_stages), 2000.0, dtype=np.float32)
    t_max = np.full((n_elements, max_stages), 25000.0, dtype=np.float32)
    g0 = np.exp(coeffs[:, :, 0]).astype(np.float32)
    return coeffs, ips, t_min, t_max, g0


def _load_ionization_potentials(
    cursor: Any,
    elements: List[str],
    placeholders: str,
    el_map: Dict[str, int],
    ips: np.ndarray,
    max_stages: int,
) -> None:
    """Fill ``ips`` in place from the ``species_physics`` table."""
    cursor.execute(
        f"SELECT element, sp_num, ip_ev FROM species_physics " f"WHERE element IN ({placeholders})",
        elements,
    )
    for row in cursor.fetchall():
        el, sp_num, ip_ev = row
        if el in el_map and ip_ev is not None:
            el_idx = el_map[el]
            stage_idx = sp_num - 1
            if 0 <= stage_idx < max_stages:
                ips[el_idx, stage_idx] = ip_ev


def _apply_partition_factory_overrides(
    db_path: str,
    el_map: Dict[str, int],
    max_stages: int,
    coeffs: np.ndarray,
    t_min: np.ndarray,
    t_max: np.ndarray,
    g0: np.ndarray,
) -> None:
    """Override coeffs/bounds/g0 in place from the ONE partition-function factory.

    ``partition_spec_for`` applies the locked policy in a single place — *prefer
    the direct-sum FIT over energy levels when present; otherwise the stored
    polynomial fallback; and always carry ``[t_min, t_max]`` + ``g0``* — and
    caches the (expensive) per-species fit so it is compute-once (Invariant #5).
    The Bayesian JAX adapter therefore bakes the SAME static arrays the manifold
    snapshot bakes; the guarded JAX evaluator in the forward model clamps/floors
    with the bounds, so jit / vmap are unaffected (the fit is a build-time
    concern).
    """
    from cflibs.atomic.database import AtomicDatabase

    try:
        factory_db = AtomicDatabase(db_path)
        for el, el_idx in el_map.items():
            for stage in (1, 2, 3):
                stage_idx = stage - 1
                if stage_idx >= max_stages:
                    continue
                spec = factory_db.partition_spec_for(el, stage)
                if spec is None:
                    continue
                spec_coeffs = (list(spec.coefficients) + [0.0] * 5)[:5]
                coeffs[el_idx, stage_idx] = spec_coeffs
                t_min[el_idx, stage_idx] = spec.t_min
                t_max[el_idx, stage_idx] = spec.t_max
                g0[el_idx, stage_idx] = spec.g0
    except Exception as exc:  # pragma: no cover - DB shape fallback
        logger.debug(f"Partition factory override skipped: {exc}")


def _load_physics_arrays(
    db_path: str,
    conn: Any,
    elements: List[str],
    placeholders: str,
    el_map: Dict[str, int],
    max_stages: int,
    ips: np.ndarray,
    coeffs: np.ndarray,
    t_min: np.ndarray,
    t_max: np.ndarray,
    g0: np.ndarray,
) -> None:
    """Load IPs + factory partition overrides in place, guarded as a unit."""
    try:
        cursor = conn.cursor()
        _load_ionization_potentials(cursor, elements, placeholders, el_map, ips, max_stages)
        _apply_partition_factory_overrides(db_path, el_map, max_stages, coeffs, t_min, t_max, g0)
    except Exception as e:
        logger.warning(f"Failed to load physics data: {e}")


def _query_atomic_data(
    db_path: str,
    elements: List[str],
    wavelength_range: Tuple[float, float],
) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Query the database to retrieve atomic lines and species physics.

    Returns
    -------
    df, coeffs, ips, t_min, t_max, g0
        ``df`` carries the per-line atomic data; ``coeffs`` is the
        ``(n_elements, n_stages, 5)`` partition-function polynomial table; and
        ``t_min`` / ``t_max`` / ``g0`` are the ``(n_elements, n_stages)`` guard
        arrays (validity window + ground-state floor) that the guarded JAX
        evaluator clamps/floors with.  Coefficients + bounds come from the ONE
        factory (:meth:`AtomicDatabase.partition_spec_for`) so the Bayesian JAX
        adapter consumes the *same* direct-sum-fit-preferred, guarded source as
        the manifold snapshot and the CPU scalar provider (PF-3/PF-4 / the
        2026-06-03 partition-provider unification).
    """
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        placeholders = ",".join(["?"] * len(elements))
        df, el_map = _build_lines_dataframe(conn, elements, wavelength_range, placeholders)

        max_stages = 3
        n_elements = len(elements)
        coeffs, ips, t_min, t_max, g0 = _init_partition_arrays(n_elements, max_stages)

        _load_physics_arrays(
            db_path,
            conn,
            elements,
            placeholders,
            el_map,
            max_stages,
            ips,
            coeffs,
            t_min,
            t_max,
            g0,
        )

    return df, coeffs, ips, t_min, t_max, g0


def _format_atomic_data_arrays(
    df: Any,
    coeffs: np.ndarray,
    ips: np.ndarray,
    elements: List[str],
    t_min: Optional[np.ndarray] = None,
    t_max: Optional[np.ndarray] = None,
    g0: Optional[np.ndarray] = None,
) -> "AtomicDataArrays":
    """Format dataframe and physics arrays into JAX :class:`AtomicDataArrays`.

    The ``t_min`` / ``t_max`` / ``g0`` guard arrays (shape
    ``(n_elements, n_stages)``) are baked into the carrier so the guarded JAX
    partition evaluator can clamp ``T`` to ``[t_min, t_max]`` and floor at
    ``g0`` inside the NUTS jit trace without any Python provider call.  When
    omitted (legacy callers) they default to the canonical
    ``[2000, 25000] K`` window and a ``g0`` of 1.
    """
    stark_w_raw = df["stark_w"].fillna(float("nan")).values
    stark_alpha_raw = df["stark_alpha"].fillna(0.5).values

    wavelength_m = df["wavelength_nm"].values * 1e-9
    ek_ev_vals = df["ek_ev"].values
    if "ei_ev" in df.columns and not df["ei_ev"].isnull().all():
        ei_ev_vals = df["ei_ev"].values
    else:
        # Delta E = hc / lambda in eV
        delta_e_ev = (H_PLANCK * C_LIGHT / wavelength_m) / EV_TO_J
        ei_ev_vals = np.maximum(ek_ev_vals - delta_e_ev, 0.0)

    gk_vals = df["gk"].values.astype(float)
    aki_vals = df["aki"].values.astype(float)
    # Oscillator strength: the Einstein-A ↔ f_ik relation is
    #   f_ik = (m_e c) / (8 π² e²) × λ² × (g_k / g_i) × A_ki
    # which in nm units evaluates to 1.499e-16 × λ² × (g_k/g_i) × A_ki.
    # The atomic DB's `lines` table doesn't carry g_i (lower-level
    # degeneracy) directly; the `energy_levels` join on
    # (element, sp_num, ei_ev) would multi-match in practice because the
    # same g can appear at the same energy across multiple ek_ev rows.
    # Until that lookup is wired, we keep the g_k ≈ g_i approximation
    # but CLAMP the result to physically plausible bounds
    # (f_osc ∈ [1e-6, 2.0]) so the worst-case 2–10× error on Tier-2
    # transitions doesn't propagate into a TwoZone optical-depth blowup.
    # Bug surfaced 2026-05-19 by AI physics review; full g_i lookup is a
    # documented follow-up.
    f_osc_raw = 1.499e-16 * df["wavelength_nm"].values ** 2 * aki_vals
    f_osc = np.clip(f_osc_raw, 1e-6, 2.0)

    n_stages = coeffs.shape[1]
    if t_min is None:
        t_min = np.full((len(elements), n_stages), 2000.0, dtype=np.float32)
    if t_max is None:
        t_max = np.full((len(elements), n_stages), 25000.0, dtype=np.float32)
    if g0 is None:
        g0 = np.ones((len(elements), n_stages), dtype=np.float32)

    return AtomicDataArrays(
        wavelength_nm=jnp.array(df["wavelength_nm"].values, dtype=jnp.float32),
        aki=jnp.array(aki_vals, dtype=jnp.float32),
        ek_ev=jnp.array(ek_ev_vals, dtype=jnp.float32),
        gk=jnp.array(gk_vals, dtype=jnp.float32),
        ip_ev=jnp.array(df["ip_ev"].values, dtype=jnp.float32),
        ion_stage=jnp.array(df["sp_num"].values - 1, dtype=jnp.int32),
        element_idx=jnp.array(df["el_idx"].values, dtype=jnp.int32),
        stark_w=jnp.array(stark_w_raw, dtype=jnp.float32),
        stark_alpha=jnp.array(stark_alpha_raw, dtype=jnp.float32),
        mass_amu=jnp.array(df["mass_amu"].values, dtype=jnp.float32),
        partition_coeffs=jnp.array(coeffs, dtype=jnp.float32),
        partition_t_min=jnp.array(t_min, dtype=jnp.float32),
        partition_t_max=jnp.array(t_max, dtype=jnp.float32),
        partition_g0=jnp.array(g0, dtype=jnp.float32),
        ionization_potentials=jnp.array(ips, dtype=jnp.float32),
        elements=list(elements),
        ei_ev=jnp.array(ei_ev_vals, dtype=jnp.float32),
        f_osc=jnp.array(f_osc, dtype=jnp.float32),
    )


def load_atomic_data(
    db_path: str,
    elements: List[str],
    wavelength_range: Tuple[float, float],
) -> "AtomicDataArrays":
    """Load atomic data from the SQLite database into JAX arrays."""
    if not HAS_JAX:
        raise ImportError("JAX required. Install with: pip install jax jaxlib")

    df, coeffs, ips, t_min, t_max, g0 = _query_atomic_data(db_path, elements, wavelength_range)
    return _format_atomic_data_arrays(df, coeffs, ips, elements, t_min, t_max, g0)


def partition_function(
    T_K: Any,
    coeffs: Any,
    *,
    t_min: Any = None,
    t_max: Any = None,
    g0: Any = None,
) -> Any:
    """Evaluate the polynomial partition function ``U(T)`` (JAX-compatible).

    Thin guarded delegator to the ONE shared evaluator
    :func:`cflibs.plasma.partition.polynomial_partition_function_jax` — the
    SAME one the manifold batch forward model uses.  This is no longer a
    standalone re-implementation of ``ln U = Σ aₙ (ln T)ⁿ``: the duplicate that
    lived here (no clamp, no ``g0`` floor, no ``[t_min, t_max]`` window,
    consumed raw in the NUTS jit trace) is deleted in favour of routing every
    Bayesian ``U(T)`` through the guarded evaluator.

    When ``t_min`` / ``t_max`` / ``g0`` are supplied (the production path threads
    them from :class:`AtomicDataArrays`), ``T`` is clamped to ``[t_min, t_max]``
    and the result floored at ``g0`` before return.  Omitting them preserves the
    legacy bounds-free behaviour for callers that still want a raw evaluation.
    """
    if HAS_JAX:
        from cflibs.plasma.partition import polynomial_partition_function_jax

        return polynomial_partition_function_jax(T_K, coeffs, t_min=t_min, t_max=t_max, g0=g0)

    # JAX-free fallback: same guard (clamp T to [t_min, t_max], floor at g0)
    # applied with NumPy, supporting both a single (5,) coefficient vector and a
    # stacked (N, 5) table.  This mirrors the shared guarded evaluator's policy;
    # it is NOT a second polynomial implementation of the unclamped form.
    T_eval = np.asarray(T_K, dtype=np.float64)
    if t_min is not None:
        T_eval = np.maximum(T_eval, np.asarray(t_min, dtype=np.float64))
    if t_max is not None:
        T_eval = np.minimum(T_eval, np.asarray(t_max, dtype=np.float64))
    log_T = np.log(np.maximum(T_eval, 1.0))
    powers = np.stack([np.ones_like(log_T), log_T, log_T**2, log_T**3, log_T**4], axis=-1)
    log_U = np.sum(np.asarray(coeffs, dtype=np.float64) * powers, axis=-1)
    U = np.exp(log_U)
    if g0 is not None:
        U = np.maximum(U, np.asarray(g0, dtype=np.float64))
    return U


def mcwhirter_log_penalty(
    T_eV: float,
    log_ne: float,
    max_delta_E_eV: float = 3.0,
    scale: float = 10.0,
) -> float:
    """Soft McWhirter criterion penalty for LTE validity.

    The McWhirter criterion requires
    ``n_e >= 1.6e12 * T^{1/2} * (Delta E)^3`` with ``T`` in K and
    ``Delta E`` in eV. The returned log-penalty is zero when the criterion
    holds and becomes increasingly negative when it is violated.
    """
    if HAS_JAX:
        T_K = _as_jax_real(T_eV) * _JAX_EV_TO_K
        log10_threshold = (
            jnp.log10(_JAX_MCWHIRTER_CONST)
            + 0.5 * jnp.log10(T_K)
            + 3.0 * jnp.log10(_as_jax_real(max_delta_E_eV))
        )
        deficit = jnp.maximum(0.0, log10_threshold - log_ne)
        return -scale * deficit**2
    T_K = T_eV * EV_TO_K
    log10_threshold = (
        np.log10(MCWHIRTER_CONST) + 0.5 * np.log10(T_K) + 3.0 * np.log10(max_delta_E_eV)
    )
    deficit = max(0.0, log10_threshold - log_ne)
    return -scale * deficit**2


# ---------------------------------------------------------------------------
# AtomicDataArrays carrier (with from_snapshot bridge for T1-1 callers)
# ---------------------------------------------------------------------------


@dataclass
class AtomicDataArrays:
    """Atomic data stored as JAX arrays for efficient computation.

    All arrays are indexed by line number (``n_lines,``).

    Notes
    -----
    The legacy schema (created by :func:`load_atomic_data`) carries a
    three-stage polynomial partition table (``partition_coeffs`` shaped
    ``(n_elements, n_stages, 5)``) and per-element Stark-temperature exponents.
    :meth:`from_snapshot` is a one-way bridge that constructs a minimal
    :class:`AtomicDataArrays` from a T1-1 :class:`AtomicSnapshot`;
    it preserves the partition coefficients but defaults ``stark_alpha`` to 0
    and ``f_osc`` to ``None``.
    """

    wavelength_nm: Any  # Line wavelengths [nm]
    aki: Any  # Einstein A coefficients [s^-1]
    ek_ev: Any  # Upper level energy [eV]
    gk: Any  # Upper level degeneracy
    ip_ev: Any  # Ionization potential of parent species [eV]
    ion_stage: Any  # Ionization stage (0 = neutral, 1 = singly ionized)
    element_idx: Any  # Element index (per line)
    stark_w: Any  # Stark width reference [nm]
    stark_alpha: Any  # Stark temperature exponent
    mass_amu: Any  # Atomic mass [amu]
    partition_coeffs: Any  # (n_elements, n_stages, 5)
    ionization_potentials: Any  # (n_elements, n_stages)
    elements: List[str] = field(default_factory=list)
    ei_ev: Any = None  # Lower level energy [eV]
    f_osc: Any = None  # Oscillator strength
    # Partition-function guard arrays (n_elements, n_stages): the static
    # [t_min, t_max] validity window + ground-state floor g0 the guarded JAX
    # evaluator clamps/floors with inside the NUTS jit trace.  ``None`` only on
    # the legacy ``from_snapshot`` bridge (whose snapshot already bakes guarded
    # coeffs); ``load_atomic_data`` always populates them from the factory.
    partition_t_min: Any = None
    partition_t_max: Any = None
    partition_g0: Any = None

    @classmethod
    def from_snapshot(cls, snapshot, elements: List[str]) -> "AtomicDataArrays":
        """Build a minimal :class:`AtomicDataArrays` from an :class:`AtomicSnapshot`.

        The bridge is intentionally narrow: only per-line fields used by the
        kernel-equivalent forward path are populated. ``stark_alpha`` defaults
        to 0 and ``f_osc`` to ``None``. The partition coefficients carried by
        the snapshot are propagated unchanged.

        Notes
        -----
        The unified kernel (T1-2) drives Saha-Boltzmann populations from the
        snapshot's polynomial partition coefficients while the legacy
        :class:`BayesianForwardModel` uses the same polynomial coefficients
        carried inside :class:`AtomicDataArrays`. The bridge therefore matches
        the partition path; it does not introduce intensity drift. The
        ``ion_stage`` field is filled from ``snapshot.species`` (mapping
        stage 1/2 -> ion_stage 0/1).
        """
        n_lines = int(np.asarray(snapshot.line_wavelengths_nm).shape[0])
        element_to_idx = {el: i for i, el in enumerate(elements)}
        line_element_idx = np.zeros(n_lines, dtype=np.int32)
        line_ion_stage = np.zeros(n_lines, dtype=np.int32)
        line_mass_amu = np.zeros(n_lines, dtype=np.float64)
        line_ip_ev = np.zeros(n_lines, dtype=np.float64)
        sp_idx = np.asarray(snapshot.line_species_index)
        ip_arr = np.asarray(snapshot.ionization_potential_ev)
        for li in range(n_lines):
            el, stage = snapshot.species[int(sp_idx[li])]
            line_element_idx[li] = element_to_idx.get(el, 0)
            line_ion_stage[li] = max(int(stage) - 1, 0)
            line_mass_amu[li] = STANDARD_MASSES.get(el, 50.0)
            line_ip_ev[li] = float(ip_arr[int(sp_idx[li])])

        if HAS_JAX:

            def _as(v):
                return jnp.asarray(v)

            stark_alpha_default = jnp.zeros(n_lines)
        else:

            def _as(v):
                return np.asarray(v)

            stark_alpha_default = np.zeros(n_lines)

        return cls(
            wavelength_nm=_as(snapshot.line_wavelengths_nm),
            aki=_as(snapshot.line_A_ki),
            ek_ev=_as(snapshot.line_E_k_ev),
            gk=_as(snapshot.line_g_k),
            ip_ev=_as(line_ip_ev),
            ion_stage=_as(line_ion_stage),
            element_idx=_as(line_element_idx),
            stark_w=_as(snapshot.line_stark_w),
            stark_alpha=stark_alpha_default,
            mass_amu=_as(line_mass_amu),
            partition_coeffs=_as(snapshot.partition_coeffs),
            ionization_potentials=_as(snapshot.ionization_potential_ev),
            elements=list(elements),
            ei_ev=_as(snapshot.line_E_i_ev),
            f_osc=None,
        )


def _atomic_data_arrays_from_snapshot(snapshot, elements):  # noqa: D401
    """Deprecated function form of :meth:`AtomicDataArrays.from_snapshot`."""
    return AtomicDataArrays.from_snapshot(snapshot, elements)


__all__ = [
    "STANDARD_MASSES",
    "AtomicDataArrays",
    "load_atomic_data",
    "partition_function",
    "mcwhirter_log_penalty",
    "_atomic_data_arrays_from_snapshot",
    "_query_atomic_data",
    "_format_atomic_data_arrays",
    "_resolve_total_species_density_cm3",
    "_compute_instrument_sigma",
    "_as_jax_real",
    "_JAX_REAL_DTYPE",
    "_JAX_SAHA_CONST_CM3",
    "_JAX_C_LIGHT",
    "_JAX_EV_TO_K",
    "_JAX_EV_TO_J",
    "_JAX_MCWHIRTER_CONST",
    "_JAX_H_PLANCK",
    "_JAX_E_CHARGE",
    "_JAX_M_E",
    "_JAX_M_PROTON",
    "logger",
]
