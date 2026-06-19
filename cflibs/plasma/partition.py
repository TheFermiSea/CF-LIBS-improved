"""
Partition function evaluation logic.

Two computation methods are provided:

1. **Direct summation** (recommended): U(T) = Σ gᵢ exp(-Eᵢ / kT) over energy
   levels from the atomic database, with plasma-truncated cutoff at
   E_max = IP - Δχ(nₑ, T).  Based on Alimohamadi & Ferland (2022, PASP 134).

2. **Polynomial** (Irwin 1981 form, NATURAL-LOG basis): the implementation
   evaluates::

       ln U(T) = a0 + a1·ln T + a2·(ln T)² + a3·(ln T)³ + a4·(ln T)⁴

   i.e. a 4th-order polynomial in ``ln T`` whose value is ``ln U``.  This is
   mathematically equivalent to Irwin (1981 ApJS 45 621) once a basis change
   ``log10 ↔ ln`` is applied to the coefficients — Irwin tabulated his fits
   in ``log10 T``/``log10 U``, but the present code stores natural-log
   coefficients (per the historical NIST-ASD-fit convention used by this
   project).  When ingesting Irwin's published Table II coefficients, run
   them through :func:`irwin_log10_to_ln_coeffs` first.

   Stored in ``partition_functions`` (a0..a4, t_min, t_max, source).  Less
   accurate than direct summation when energy levels are available
   (errors up to 66 % for some species) — use only as a fallback when the
   ``energy_levels`` table is missing rows for a species.

Historical note (2026-05-09 fix): the previous docstring ambiguously said
"log U = Σ aₙ (log T)ⁿ" without specifying base.  The implementation has
always used natural log (``np.log``), and the 13 partition_functions rows
in the production DB at the time of audit were fit to that convention.
The 30–60 % poly-vs-direct-sum discrepancy noted in the audit was caused
by *stale fit data* (the polynomial coefficients were fit against an
older energy_levels snapshot than the one currently in the DB), not by a
math/convention mismatch.  Re-fitting from the current EL table — which
is what :mod:`scripts.archive.migrations.populate_partition_functions` does — restores the
consistency.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

try:  # Python 3.8+ — Protocol lives in typing on modern interpreters.
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover - fallback for old runtimes
    from typing_extensions import Protocol, runtime_checkable  # type: ignore

import numpy as np

from cflibs.core.constants import C_LIGHT, E_CHARGE, J_TO_EV, KB, KB_EV
from cflibs.core.jax_runtime import HAS_JAX, jit_if_available, jnp
from cflibs.core.logging_config import get_logger

jit = jit_if_available

logger = get_logger("plasma.partition")


# ---------------------------------------------------------------------------
# Canonical Debye-Hückel ionization potential depression (single source)
# ---------------------------------------------------------------------------
#
# THE one Debye-Hückel IPD implementation for the whole package.  Previously
# two divergent formulas coexisted (audit Family J): this module used the
# approximate ``3e-8·Z·√n_e/√T`` form (~0.0949 eV at n_e=1e17, T=1e4 K) while
# ``cflibs.plasma.saha_boltzmann`` used the canonical Gaussian-CGS form
# (~0.0660 eV) — a 1.44× discrepancy.  Both call sites now route through
# :func:`ionization_potential_depression`, so the Saha exponent and the
# partition-sum truncation see the *same* Δχ.
#
# Derived CGS helpers built from the canonical SI constants in
# ``cflibs.core.constants`` (mirrors ``saha_boltzmann.py``).
_E_ESU = E_CHARGE * C_LIGHT * 10.0  # electron charge [esu = statcoulomb]
_KB_ERG = KB * 1.0e7  # Boltzmann constant [erg/K]
_ERG_TO_EV = J_TO_EV * 1.0e-7  # conversion factor erg -> eV


def ionization_potential_depression(n_e: float, T_K: float, Z: int = 1) -> float:
    """Debye-Hückel ionization potential depression (canonical Gaussian-CGS).

    .. math::

        \\Delta\\chi = Z \\, \\frac{e^2}{\\lambda_D},
        \\qquad
        \\lambda_D = \\sqrt{\\frac{k_B T}{4\\pi n_e e^2}}

    in Gaussian-CGS units, converted to eV.  At ``n_e = 1e17 cm⁻³``,
    ``T = 1e4 K`` this gives ≈ 0.066 eV for ``Z = 1`` (Mihalas 1978
    Eq. 9-106; Alimohamadi & Ferland 2022).  This is the SAME computation as
    :func:`cflibs.plasma.saha_boltzmann.ionization_potential_lowering`, which
    now delegates here so the Saha exponent and the direct-sum partition
    cutoff use one consistent Δχ.

    Parameters
    ----------
    n_e : float
        Electron density in cm⁻³.
    T_K : float
        Temperature in Kelvin.
    Z : int
        Ionic charge of perturbers (default 1 for singly-ionized).

    Returns
    -------
    float
        IPD in eV.  Returns 0 if n_e <= 0 or T_K <= 0.
    """
    if n_e <= 0 or T_K <= 0:
        return 0.0
    # Debye length: lambda_D = sqrt(kT / (4*pi*n_e*e^2))  [cm]
    lambda_D = np.sqrt(_KB_ERG * T_K / (4.0 * np.pi * n_e * _E_ESU**2))
    # IPD = Z * e^2 / lambda_D  [erg], converted to eV.
    delta_chi_erg = Z * _E_ESU**2 / lambda_D
    return float(delta_chi_erg * _ERG_TO_EV)


def ionization_potential_depression_jax(n_e: Any, T_K: Any, Z: int = 1) -> Any:
    """JAX twin of :func:`ionization_potential_depression` (traced inputs).

    Identical Gaussian-CGS Debye-Hückel math — ``Δχ = Z·e²/λ_D`` with
    ``λ_D = sqrt(k_B T / (4π n_e e²))`` — expressed in ``jnp`` so jit/vmap
    kernels (the snapshot Saha in :mod:`cflibs.radiation.kernels` and the
    manifold generator's Saha-Eggert solver) can apply the SAME Δχ the CPU
    solver uses (audit 01-F4, bead CF-LIBS-improved-rs7e).  Non-physical
    inputs (``n_e <= 0`` or ``T_K <= 0``) yield 0, mirroring the scalar
    helper, via ``jnp.where`` (no Python branching on traced values).

    Parameters
    ----------
    n_e : float or traced scalar/array
        Electron density in cm⁻³.
    T_K : float or traced scalar/array
        Temperature in Kelvin.
    Z : int
        Ionic charge of perturbers (default 1, static).

    Returns
    -------
    jnp.ndarray
        IPD in eV, broadcast over the inputs.
    """
    safe_ne = jnp.maximum(n_e, 1e-300)
    safe_T = jnp.maximum(T_K, 1e-300)
    lambda_D = jnp.sqrt(_KB_ERG * safe_T / (4.0 * jnp.pi * safe_ne * _E_ESU**2))
    delta_chi = Z * _E_ESU**2 / lambda_D * _ERG_TO_EV
    return jnp.where((n_e > 0.0) & (T_K > 0.0), delta_chi, 0.0)


def direct_sum_partition_function(
    T_K: float,
    g_levels: np.ndarray,
    E_levels_ev: np.ndarray,
    ip_ev: float,
    n_e: Optional[float] = None,
) -> float:
    """Compute partition function by direct summation over energy levels.

    U(T) = Σ gᵢ exp(-Eᵢ / kT)   for  Eᵢ < IP - Δχ

    This is the standard method recommended by Alimohamadi & Ferland (2022)
    and used by NIST ASD.  The sum is truncated at the plasma-lowered
    ionization potential to exclude dissolved Rydberg states.

    Parameters
    ----------
    T_K : float
        Temperature in Kelvin.
    g_levels : np.ndarray
        Statistical weights (degeneracies) for each energy level.
    E_levels_ev : np.ndarray
        Energy of each level in eV (measured from ground state).
    ip_ev : float
        Ionization potential in eV (from species_physics table).
    n_e : float, optional
        Electron density in cm⁻³.  If provided, applies Debye-Hückel
        IPD to further lower the cutoff.  If None, sharp IP cutoff only.

    Returns
    -------
    float
        Partition function U(T).  Guaranteed >= 1.0 (at least the
        ground state contributes).
    """
    if T_K <= 1.0:
        return max(float(g_levels[0]) if len(g_levels) > 0 else 1.0, 1.0)

    # Plasma-truncated cutoff
    delta_chi = ionization_potential_depression(n_e, T_K) if n_e is not None else 0.0
    e_max = ip_ev - delta_chi

    # Mask: include only levels below the effective ionization limit
    mask = E_levels_ev < e_max
    if not np.any(mask):
        return max(float(g_levels[0]) if len(g_levels) > 0 else 1.0, 1.0)

    kT_ev = KB_EV * T_K
    U = float(np.sum(g_levels[mask] * np.exp(-E_levels_ev[mask] / kT_ev)))
    return max(U, 1.0)


# ---------------------------------------------------------------------------
# Direct-sum-preferred polynomial coefficient fit (manifold/Bayesian fallback)
# ---------------------------------------------------------------------------

# Fit grid + LIBS validation band for :func:`direct_sum_fit_coeffs`.  These
# mirror the standalone regeneration recipe in
# ``scripts/archive/migrations/regenerate_partition_functions.py`` so the load-time fallback and
# the persisted DB rows agree by construction.
_DSFIT_T_MIN = 2000.0
_DSFIT_T_MAX = 25000.0
_DSFIT_N_POINTS = 60
_DSFIT_KEY_TEMPS = (5000.0, 10000.0, 15000.0, 20000.0)
_DSFIT_BAND_LO = 6000.0
_DSFIT_BAND_HI = 12000.0


def direct_sum_fit_coeffs(
    g_levels: np.ndarray,
    E_levels_ev: np.ndarray,
    ip_ev: float,
) -> Optional[Tuple[List[float], float, float]]:
    """Fit ``ln U = Σ aₙ (ln T)ⁿ`` to the direct-sum U(T) over energy levels.

    This is the *direct-sum-preferred* partition-function provider used by the
    JAX manifold / Bayesian batch paths.  The default CPU solver already
    prefers direct summation (``saha_boltzmann.py``); the JAX kernels can only
    consume a polynomial (they ``vmap`` over plasma parameters and cannot hold
    a per-species variable-length level sum), so this helper converts the
    direct-sum into an equivalent polynomial at snapshot-construction time.
    The kernel then evaluates the *same* polynomial form it always has — only
    the coefficients change — so jit / vmap are completely unaffected.

    The fit is constrained so the resulting polynomial does not undershoot the
    direct-sum (a physically-impossible regime) across the ps-LIBS band
    (6000–12000 K): a small positive ln-offset is added to ``a0`` to lift the
    minimum band ratio to 1.0.  When the fit turns over inside the default
    ``[2000, 25000] K`` window the validity bounds are tightened past the
    cold/hot turnover (the evaluator clamps to ``[t_min, t_max]``), but the
    LIBS band is never clipped.

    Parameters
    ----------
    g_levels, E_levels_ev : np.ndarray
        Statistical weights and level energies (eV) from ``energy_levels``.
    ip_ev : float
        Ionization potential (eV) — direct-sum cutoff.

    Returns
    -------
    tuple[list[float], float, float] or None
        ``(coeffs[0..4], t_min, t_max)`` in the natural-log basis, or ``None``
        when there are too few levels to fit (caller keeps the stored
        polynomial — never fabricate).
    """
    g = np.asarray(g_levels, dtype=np.float64)
    e = np.asarray(E_levels_ev, dtype=np.float64)
    if g.size < 2:
        return None

    def _ds(T_K: float) -> float:
        mask = e < ip_ev
        if not np.any(mask):
            return 1.0
        return max(float(np.sum(g[mask] * np.exp(-e[mask] / (KB_EV * T_K)))), 1.0)

    def _poly(coeffs: List[float], T_K: float) -> float:
        ln_T = np.log(T_K)
        return float(np.exp(sum(a * ln_T**i for i, a in enumerate(coeffs))))

    T_grid = np.unique(
        np.concatenate(
            [np.linspace(_DSFIT_T_MIN, _DSFIT_T_MAX, _DSFIT_N_POINTS), np.array(_DSFIT_KEY_TEMPS)]
        )
    )
    Q_grid = np.array([_ds(T) for T in T_grid])
    ln_T = np.log(T_grid)
    ln_Q = np.log(np.maximum(Q_grid, 1e-12))
    w = np.ones_like(T_grid)
    for kt in _DSFIT_KEY_TEMPS:
        w[np.isclose(T_grid, kt)] = 10.0
    coeffs = list(np.polynomial.polynomial.polyfit(ln_T, ln_Q, 4, w=w))
    coeffs = (coeffs + [0.0] * 5)[:5]

    # Monotonicity guard: tighten the validity window past any turnover so the
    # evaluator's clamp holds U flat in the (physically-flat) tails.
    t_min, t_max = _DSFIT_T_MIN, _DSFIT_T_MAX
    Tm = np.linspace(t_min, t_max, 600)
    Um = np.array([_poly(coeffs, T) for T in Tm])
    dec = np.where(np.diff(Um) < 0)[0]
    if dec.size:
        mid = len(Tm) // 2
        cold = [i for i in dec if i < mid]
        hot = [i for i in dec if i >= mid]
        if cold:
            t_min = min(float(Tm[max(cold) + 1]), _DSFIT_BAND_LO)
        if hot:
            t_max = max(float(Tm[min(hot)]), _DSFIT_BAND_HI)

    # Lift so U_poly >= U_directsum across the LIBS band (a pure ln-offset on
    # a0 preserves shape + monotonicity).
    T_band = np.linspace(_DSFIT_BAND_LO, _DSFIT_BAND_HI, 25)
    ratios = np.array([_poly(coeffs, T) / _ds(T) for T in T_band])
    coeffs[0] += max(0.0, float(np.log(1.0 / float(ratios.min()))))
    return [float(c) for c in coeffs], float(t_min), float(t_max)


# ---------------------------------------------------------------------------
# The single derived partition-function spec (one policy, two adapters)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PartitionFunctionSpec:
    """Coefficients + bounds + g0 for ONE species, derived by the single policy.

    This is the load-bearing seam between the policy (*prefer the direct-sum
    fit over energy levels when present; otherwise the stored polynomial; always
    carry ``[t_min, t_max]`` and ``g0``*) and the two adapters that consume it:

    * the **CPU scalar adapter** — :class:`PolynomialPartitionFunctionProvider`,
      built from this spec via :meth:`to_provider`, whose ``.at()`` applies the
      clamp + ``g0`` floor at Python runtime; and
    * the **JAX batched adapter** — the same ``coefficients`` / ``t_min`` /
      ``t_max`` / ``g0`` baked as plain floats into the static snapshot arrays
      and evaluated by the guarded ``polynomial_partition_function_jax``.

    Producing both adapters from the *same* :class:`PartitionFunctionSpec`
    instance is what makes "direct-sum-preferred, always guarded" provable in a
    single place rather than re-implemented at the ~9 historical call sites
    (see ``docs/architecture/2026-06-03-composition-pipeline-diagnosis.md`` § 2.1
    and ``CONTEXT.md`` § "The partition-function provider").

    Attributes
    ----------
    element : str
        Element symbol — informational only.
    ionization_stage : int
        Ionization stage — informational only.
    coefficients : tuple[float, ...]
        Natural-log polynomial coefficients ``(a0, …, a4)``.
    t_min, t_max : float
        Validity window; both adapters clamp ``T`` into this interval.
    g0 : float
        Ground-state statistical weight (strict physical lower bound on U).
    source : str
        Provenance: ``"direct_sum_fit"`` when fit live from energy levels,
        otherwise the stored ``partition_functions.source`` (or ``"default"``).
    from_direct_sum : bool
        ``True`` when ``coefficients`` were fit from the species' energy
        levels; ``False`` when they came from the stored polynomial fallback.
    """

    element: str
    ionization_stage: int
    coefficients: Tuple[float, ...]
    t_min: float
    t_max: float
    g0: float
    source: str
    from_direct_sum: bool
    # The IP-filtered, energy-sorted ``(g, E_ev, ip_ev)`` level arrays the
    # direct-sum FIT was derived from, retained ONLY so the CPU scalar adapter
    # can evaluate the EXACT direct sum (not the fit) at Python runtime — see
    # :meth:`to_provider`.  ``None`` when ``from_direct_sum`` is False (the
    # stored-polynomial fallback has no levels).  These arrays are NOT used by
    # the JAX batched adapter, which bakes ``coefficients`` (the fit) into its
    # static snapshot because ``vmap`` needs fixed-shape arrays.
    levels: Optional[Tuple[np.ndarray, np.ndarray, float]] = None

    def to_provider(self) -> "PartitionFunctionProvider":
        """Build the CPU scalar adapter.

        When the species has tabulated energy levels (``from_direct_sum``) the
        provider evaluates the **exact** direct sum ``U(T) = Σ gᵢ exp(-Eᵢ/kT)``
        at runtime — *not* the ln-poly fit.  This is what keeps the default
        CPU ``invert`` / ``analyze`` / iterative path bit-for-bit unchanged
        (Invariant #2): the fit is a build-time concern for the JAX snapshot,
        where ``vmap`` needs static fixed-shape arrays, but a Python-runtime
        scalar caller can afford the full sum and the diagnosis's "prefer the
        direct sum over energy levels" is taken literally for it.

        When no levels exist the provider falls back to the guarded stored
        polynomial (clamp ``T`` to ``[t_min, t_max]``, floor at ``g0``).
        """
        if self.from_direct_sum and self.levels is not None:
            g_arr, e_arr, ip_ev = self.levels
            return DirectSumPartitionFunctionProvider(
                element=self.element,
                ionization_stage=self.ionization_stage,
                _g_levels=tuple(float(g) for g in np.asarray(g_arr, dtype=np.float64)),
                _e_levels_ev=tuple(float(e) for e in np.asarray(e_arr, dtype=np.float64)),
                ip_ev=float(ip_ev),
                t_min=self.t_min,
                t_max=self.t_max,
                _g0=self.g0,
                source=self.source,
            )
        return PolynomialPartitionFunctionProvider(
            element=self.element,
            ionization_stage=self.ionization_stage,
            coefficients=self.coefficients,
            t_min=self.t_min,
            t_max=self.t_max,
            _g0=self.g0,
            source=self.source,
        )


# Module-level spec cache so the per-species direct-sum FIT (the expensive part
# — a 4th-order ln-poly regression over hundreds of energy levels) runs
# compute-once per (db, element, stage). The snapshot builder fits ~83 elements
# × 2-3 stages at build time; without this cache it would refit on every
# snapshot/spectrum. Keyed on the same (db_path, element, stage) tuple as
# ``_level_cache``.
_spec_cache: Dict[Tuple[str, str, int], "PartitionFunctionSpec"] = {}


#: Stored ``partition_functions.source`` values that outrank the direct-sum
#: fit over our (possibly incomplete) ``energy_levels`` table.  Barklem &
#: Collet 2016 (A&A 588, A96) is computed from complete NIST level lists,
#: whereas the in-DB level scrape drops high-Rydberg levels (Ca I −25 %,
#: Na I −30 %, K I −40 % at 1e4 K — audit 2026-06-09, finding 01-F3).  The
#: shipped production DB carries no rows with these sources, so this
#: preference is inert until a patched DB (see
#: ``scripts/archive/migrations/patch_partition_functions_bc2016.py``) is explicitly wired in.
AUTHORITATIVE_PF_SOURCES = frozenset({"BarklemCollet2016"})


def _stored_polynomial_spec(
    atomic_db: Any, element: str, ionization_stage: int, g0: float
) -> Optional["PartitionFunctionSpec"]:
    """Build a :class:`PartitionFunctionSpec` from the stored polynomial row.

    Returns ``None`` when the ``partition_functions`` table has no row for the
    species.  ``t_min``/``t_max`` default to ``[2000, 25000] K`` when NULL.
    """
    try:
        pf = atomic_db.get_partition_coefficients(element, ionization_stage)
    except Exception:  # pragma: no cover - defensive
        pf = None
    if pf is None:
        return None
    coeffs_list = list(pf.coefficients)
    coeffs_list += [0.0] * (5 - len(coeffs_list))
    return PartitionFunctionSpec(
        element=element,
        ionization_stage=int(ionization_stage),
        coefficients=tuple(float(c) for c in coeffs_list[:5]),
        t_min=float(pf.t_min) if pf.t_min is not None else 2000.0,
        t_max=float(pf.t_max) if pf.t_max is not None else 25000.0,
        g0=g0,
        source=str(pf.source or ""),
        from_direct_sum=False,
    )


def derive_partition_spec(
    atomic_db: Any,
    element: str,
    ionization_stage: int,
) -> Optional["PartitionFunctionSpec"]:
    """Derive the single :class:`PartitionFunctionSpec` for a species.

    This is THE one place the partition-function policy lives:

    0. If the stored ``partition_functions`` row carries an AUTHORITATIVE
       source (:data:`AUTHORITATIVE_PF_SOURCES`, e.g. a Barklem & Collet 2016
       refit), it wins outright — such rows derive from complete level lists,
       which beat a direct-sum over our incomplete ``energy_levels`` scrape.
       Inert for the shipped DB (no authoritative rows yet).
    1. If the species has tabulated ``energy_levels`` *and* a fittable
       (``>= 2``) level count, fit the direct-sum to a guarded ln-polynomial
       via :func:`direct_sum_fit_coeffs` (``source="direct_sum_fit"``).
    2. Otherwise fall back to the stored ``partition_functions`` polynomial
       (its own ``t_min``/``t_max``, defaulting to ``[2000, 25000] K`` when the
       columns are NULL).
    3. If neither exists, return ``None`` — the caller decides the default
       (e.g. ``ln U = ln 2`` placeholder for the snapshot pad rows); this
       helper never fabricates coefficients.

    The ground-state ``g0`` is always taken from :func:`get_ground_state_g`
    (lowest-energy level's degeneracy, default 1.0).

    The result is cached per ``(db_path, element, stage)`` so the fit is
    compute-once (invariant 5).

    Returns
    -------
    PartitionFunctionSpec or None
    """
    cache_key = (
        str(getattr(atomic_db, "db_path", id(atomic_db))),
        element,
        int(ionization_stage),
    )
    cached = _spec_cache.get(cache_key)
    if cached is not None:
        return cached

    g0 = float(get_ground_state_g(atomic_db, element, ionization_stage))

    # (0) Authoritative stored rows (complete-level-list provenance) win.
    stored = _stored_polynomial_spec(atomic_db, element, ionization_stage, g0)
    if stored is not None and stored.source in AUTHORITATIVE_PF_SOURCES:
        _spec_cache[cache_key] = stored
        return stored

    # (1) Prefer the direct-sum fit when energy levels are present.
    ip_ev: Optional[float] = None
    try:
        levels = atomic_db.get_energy_levels(element, ionization_stage)
    except Exception:  # pragma: no cover - defensive DB-shape guard
        levels = None
    if levels:
        try:
            ip_ev = atomic_db.get_ionization_potential(element, ionization_stage)
        except Exception:  # pragma: no cover - defensive
            ip_ev = None
        if ip_ev is None:
            ip_ev = max((lev.energy_ev for lev in levels), default=0.0) + 1.0
        g_arr = np.array([lev.g for lev in levels], dtype=np.float64)
        e_arr = np.array([lev.energy_ev for lev in levels], dtype=np.float64)
        fit = direct_sum_fit_coeffs(g_arr, e_arr, float(ip_ev))
        if fit is not None:
            coeffs, t_min, t_max = fit
            # Retain the IP-filtered, energy-sorted level arrays the CPU scalar
            # adapter sums directly — the SAME arrays ``evaluate_direct`` would
            # consume via ``get_levels_for_species`` — so the CPU path stays
            # bit-for-bit on the exact direct sum (Invariant #2).  The JAX
            # snapshot still bakes ``coefficients`` (the fit) below.
            provider_levels = get_levels_for_species(atomic_db, element, ionization_stage)
            spec = PartitionFunctionSpec(
                element=element,
                ionization_stage=int(ionization_stage),
                coefficients=tuple(float(c) for c in coeffs),
                t_min=float(t_min),
                t_max=float(t_max),
                g0=g0,
                source="direct_sum_fit",
                from_direct_sum=True,
                levels=provider_levels,
            )
            _spec_cache[cache_key] = spec
            return spec

    # (2) Fall back to the stored polynomial (prefetched in step 0).
    if stored is None:
        return None
    _spec_cache[cache_key] = stored
    return stored


# ---------------------------------------------------------------------------
# Energy level cache for partition function evaluation
# ---------------------------------------------------------------------------

# Module-level cache: {(db_path, element, stage): (g_array, E_array, ip_ev)}
_level_cache: Dict[Tuple[str, str, int], Tuple[np.ndarray, np.ndarray, float]] = {}


def get_levels_for_species(
    atomic_db: Any,
    element: str,
    ionization_stage: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Load and cache energy level data for a species.

    Returns (g_array, E_array, ip_ev) or None if data is unavailable.
    Levels above the ionization potential are pre-filtered.
    """
    cache_key = (str(getattr(atomic_db, "db_path", id(atomic_db))), element, ionization_stage)
    if cache_key in _level_cache:
        return _level_cache[cache_key]

    # Query energy levels via public API
    try:
        levels = atomic_db.get_energy_levels(element, ionization_stage)
    except Exception:
        return None

    if not levels:
        return None

    # Get ionization potential
    ip = atomic_db.get_ionization_potential(element, ionization_stage)
    if ip is None:
        # Fallback: use max level energy + 1 eV as rough IP
        ip = max(lev.energy_ev for lev in levels) + 1.0

    g_arr = np.array([lev.g for lev in levels], dtype=np.float64)
    E_arr = np.array([lev.energy_ev for lev in levels], dtype=np.float64)

    # Sort by energy
    sort_idx = np.argsort(E_arr)
    g_arr = g_arr[sort_idx]
    E_arr = E_arr[sort_idx]

    # Pre-filter autoionizing levels (belt-and-suspenders with DB cleanup)
    below_ip = E_arr < ip
    g_arr = g_arr[below_ip]
    E_arr = E_arr[below_ip]

    result = (g_arr, E_arr, ip)
    _level_cache[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Legacy polynomial partition functions
# ---------------------------------------------------------------------------


def get_ground_state_g(
    atomic_db: Any,
    element: str,
    ionization_stage: int,
    default: float = 1.0,
) -> float:
    """Look up the ground-state statistical weight g0 from the energy_levels table.

    Used as the physical lower bound for polynomial-fallback partition function
    evaluation (a partition function U(T) cannot be less than the ground state's
    degeneracy because the ground state always contributes a Boltzmann weight of
    1).  Falls back to ``default`` when the DB lookup fails for any reason
    (missing species, no levels, exception).

    Parameters
    ----------
    atomic_db : Any
        Atomic data source exposing ``get_energy_levels``.
    element : str
        Element symbol.
    ionization_stage : int
        Ionization stage (1 = neutral).
    default : float
        Value returned when no levels are available (default 1.0, since
        every level has g >= 1).

    Returns
    -------
    float
        Ground-state statistical weight, or ``default`` if unavailable.
    """
    try:
        levels = atomic_db.get_energy_levels(element, ionization_stage)
    except Exception:
        return default
    if not levels:
        return default
    # Lowest-energy level — sort by energy_ev just in case the DB iteration
    # order is not energy-sorted.
    try:
        g0 = min(levels, key=lambda lev: lev.energy_ev).g
    except Exception:
        return default
    return float(g0) if g0 and g0 > 0 else default


# ---------------------------------------------------------------------------
# Canonical last-resort fallback ladder (ONE helper for the whole package)
# ---------------------------------------------------------------------------
#
# Historically every consumer carried its own hardcoded U(T) fallback
# (25.0 / 15.0 / 2.0 — and 10.0, ln 2, per-species dicts elsewhere) for
# species the provider factory cannot resolve.  That was silently and
# catastrophically wrong for closed-shell ions: the production DB has ZERO
# energy levels and ZERO stored polynomials for Na II / Li II / H II, so the
# stage-II constant 15.0 was used for Na II whose true U is ~1.00 (Ne-like
# closed shell, first excited level ≈ 33 eV) — a ~15× error feeding straight
# into the Saha multiplier of a basalt major (audit 2026-06-09, finding
# 02-F1; bead CF-LIBS-improved-16m7).
#
# The ladder below replaces all of those sites:
#
#   (a) EXACT values from isoelectronic structure: bare nuclei (U = 1),
#       hydrogen-like ions (ground ²S₁/₂, g = 2; first excited ≥ 10.2 eV so
#       U = 2 across the LIBS band), and noble-gas-like closed shells
#       (¹S₀ ground, first excited ≳ 10 eV, U ≈ 1).
#   (b) the species' ground-level degeneracy g₀ from ``energy_levels`` when
#       a database handle is available (U ≥ g₀ is a strict lower bound and
#       the dominant term for low-lying-sparse species) — logged, because it
#       can still be a large underestimate for open-shell species;
#   (c) only then the legacy generic constants, with a WARNING naming the
#       species — silent wrong physics is the bug, not the constant itself.

# Symbols indexed by atomic number (Z = index + 1), H..U.  Self-contained so
# this module keeps zero non-core imports.
_ELEMENT_SYMBOLS: Tuple[str, ...] = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U",
)  # fmt: skip
_ATOMIC_NUMBERS: Dict[str, int] = {sym: z + 1 for z, sym in enumerate(_ELEMENT_SYMBOLS)}

#: Electron counts with a noble-gas (closed-shell ¹S₀) configuration.
_NOBLE_GAS_ELECTRON_COUNTS = frozenset({2, 10, 18, 36, 54, 86})

#: Legacy generic constants — tier (c) of the ladder, by ionization stage.
_GENERIC_PARTITION_FALLBACK: Dict[int, float] = {1: 25.0, 2: 15.0}
_GENERIC_PARTITION_FALLBACK_DEFAULT = 2.0

# Warn-once registry so per-iteration solver loops do not spam the log.
_FALLBACK_WARNED: set = set()


def _closed_shell_partition_value(element: str, ionization_stage: int) -> Optional[float]:
    """Tier (a): exact U for bare / hydrogen-like / noble-gas-like species.

    Electron counting: a stage-``s`` ion of element ``Z`` carries
    ``Z - (s - 1)`` electrons.  Returns ``None`` when the species is not one
    of the exactly-known configurations (or the element symbol is unknown).

    Examples: H II (0 e⁻) → 1; H I / He II (1 e⁻) → 2; He I / Li II (2 e⁻,
    He-like) → 1; Na II / Mg III (10 e⁻, Ne-like) → 1; K II / Ca III (18 e⁻,
    Ar-like) → 1.  First excited levels of all of these lie ≥ 10 eV above
    ground, so the ground term is exact to <0.5 % across the LIBS band
    (T ≤ ~2 eV).
    """
    z = _ATOMIC_NUMBERS.get(element)
    if z is None or ionization_stage < 1:
        return None
    n_electrons = z - (ionization_stage - 1)
    if n_electrons == 0:
        return 1.0  # bare nucleus
    if n_electrons == 1:
        return 2.0  # hydrogen-like: ²S₁/₂ ground, g = 2
    if n_electrons in _NOBLE_GAS_ELECTRON_COUNTS:
        return 1.0  # noble-gas-like closed shell: ¹S₀ ground, g = 1
    return None


def _ground_degeneracy_partition_value(
    atomic_db: Any, element: str, ionization_stage: int
) -> Optional[float]:
    """Tier (b): the species' ground-level degeneracy g₀ from ``energy_levels``.

    Returns ``None`` when no database handle is supplied or the species has
    no tabulated levels (so the caller can distinguish "no data" from the
    sentinel ``default`` that :func:`get_ground_state_g` would return).
    """
    if atomic_db is None:
        return None
    try:
        levels = atomic_db.get_energy_levels(element, ionization_stage)
    except Exception:
        return None
    if not levels:
        return None
    g0 = get_ground_state_g(atomic_db, element, ionization_stage)
    return float(g0)


def _warn_partition_fallback(element: str, ionization_stage: int, tier: str, value: float) -> None:
    """Warn once per (species, tier) that a partition fallback engaged."""
    key = (element, int(ionization_stage), tier)
    if key in _FALLBACK_WARNED:
        return
    _FALLBACK_WARNED.add(key)
    if tier == "g0":
        logger.warning(
            "Partition function for %s %s unresolved (no provider); using "
            "ground-level degeneracy g0=%.1f as U(T). This is a strict lower "
            "bound and may underestimate U for open-shell species.",
            element,
            _roman(ionization_stage),
            value,
        )
    else:
        logger.warning(
            "Partition function for %s %s unresolved (no energy levels, no "
            "stored polynomial); falling back to the GENERIC constant U=%.1f. "
            "This is NOT a physical value — ingest atomic data for this "
            "species (e.g. scripts/archive/migrations/patch_partition_functions_bc2016.py).",
            element,
            _roman(ionization_stage),
            value,
        )


def _roman(stage: int) -> str:
    """Spectroscopic stage notation for log messages (1 → 'I', 2 → 'II', …)."""
    numerals = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}
    return numerals.get(int(stage), str(stage))


def canonical_partition_fallback(
    element: str,
    ionization_stage: int,
    atomic_db: Any = None,
    *,
    warn: bool = True,
) -> float:
    """THE canonical last-resort U(T) estimate for unresolvable species.

    Physically-grounded fallback ladder (see the section comment above):

    1. Exact isoelectronic values — bare nuclei (1), hydrogen-like (2),
       noble-gas-like closed shells (≈1).  These are temperature-independent
       to <0.5 % across the LIBS band because the first excited levels lie
       ≥ 10 eV above ground.
    2. The ground-level degeneracy ``g0`` from ``energy_levels`` when a
       database handle is available (strict lower bound on U).
    3. The legacy generic constants 25.0 / 15.0 / 2.0 by stage, with a
       warning naming the species — never silently.

    Parameters
    ----------
    element : str
        Element symbol.
    ionization_stage : int
        Ionization stage (1 = neutral).
    atomic_db : optional
        Atomic data source for the tier-(b) ``g0`` lookup.  ``None`` skips
        straight from tier (a) to tier (c).
    warn : bool, default True
        Emit the (warn-once) log message for tiers (b)/(c).  Pass ``False``
        from pre-seeding contexts that fill default arrays which may never be
        read (e.g. snapshot scaffolding), to avoid misleading warnings.

    Returns
    -------
    float
        Fallback partition function estimate.
    """
    exact = _closed_shell_partition_value(element, ionization_stage)
    if exact is not None:
        return exact

    g0_value = _ground_degeneracy_partition_value(atomic_db, element, ionization_stage)
    if g0_value is not None:
        if warn:
            _warn_partition_fallback(element, ionization_stage, "g0", g0_value)
        return g0_value

    generic = _GENERIC_PARTITION_FALLBACK.get(
        int(ionization_stage), _GENERIC_PARTITION_FALLBACK_DEFAULT
    )
    if warn:
        _warn_partition_fallback(element, ionization_stage, "generic", generic)
    return generic


def lookup_partition_function(
    partition_funcs: Dict[str, float],
    element: str,
    ionization_stage: int,
    atomic_db: Any = None,
) -> float:
    """Dict lookup with the canonical physics-grounded fallback.

    Replaces the historical ``partition_funcs.get(el, 25.0)`` /
    ``.get(el, 15.0)`` pattern scattered across the inversion package: a
    missing entry now routes through :func:`canonical_partition_fallback`
    (exact closed-shell values first, then ``g0``, then a *warned* generic
    constant) instead of a silent hardcoded number.
    """
    value = partition_funcs.get(element)
    if value is not None:
        return float(value)
    return canonical_partition_fallback(element, ionization_stage, atomic_db)


def polynomial_partition_function(
    T_K: float,
    coefficients: List[float],
    *,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    g0: Optional[float] = None,
) -> float:
    """
    Evaluate partition function using the Irwin (1981) polynomial form,
    *natural-log basis*.

    The function evaluates::

        ln U(T) = Σ_{n=0..4} a_n · (ln T)^n

    Note: the polynomial argument is the **natural** logarithm of T, and
    the polynomial value is the **natural** logarithm of U.  Irwin's
    published Table II uses base-10 logs; convert his coefficients with
    :func:`irwin_log10_to_ln_coeffs` before storing them as ``a_n`` here.

    Extrapolation guard (bead CF-LIBS-improved-s1qr.1, 2026-05-25): outside
    the fit domain ``[t_min, t_max]`` the polynomial is unconstrained and
    can blow up exponentially (e.g. Ca I U(100 000 K) = 1.14e5 vs the
    direct-sum truth of ~200, 560× wrong) or fall below the ground-state
    degeneracy (e.g. Nb I U(500 K) = 0.31 < g0=1).  When ``t_min``/``t_max``
    are supplied, the input temperature is clamped to that interval before
    evaluation; when ``g0`` is supplied, the result is floored at ``g0``.
    All three are keyword-only so legacy callers retain bit-identical
    behaviour.

    Parameters
    ----------
    T_K : float
        Temperature in Kelvin
    coefficients : List[float]
        Polynomial coefficients [a0, a1, a2, a3, a4], natural-log basis.
    t_min, t_max : float, optional
        Validity range of the polynomial fit (from the ``partition_functions``
        table).  When both are supplied, ``T_K`` is clamped to
        ``[t_min, t_max]`` prior to polynomial evaluation.  Passing one
        alone clamps on that side only.
    g0 : float, optional
        Ground-state degeneracy (statistical weight of the lowest energy
        level).  The returned value is floored at ``g0`` so the partition
        function cannot drop below the ground-state contribution, which is
        a strict physical lower bound.  Defaults to no floor when omitted
        (legacy behaviour).

    Returns
    -------
    float
        Partition function value U(T) = exp(Σ a_n (ln T)^n).

    See Also
    --------
    irwin_log10_to_ln_coeffs : Convert log10-basis Irwin coefficients
        to the natural-log basis used here.
    """
    if T_K <= 1.0:
        return 1.0 if g0 is None else max(1.0, float(g0))

    # Clamp T to the polynomial's validity window (if supplied) before
    # evaluating — extrapolating exp(quartic in ln T) outside [t_min, t_max]
    # is the root cause of the 560× Ca I error documented above.
    T_eval = float(T_K)
    if t_min is not None:
        T_eval = max(T_eval, float(t_min))
    if t_max is not None:
        T_eval = min(T_eval, float(t_max))

    ln_T = np.log(T_eval)
    ln_U = 0.0

    for i, a in enumerate(coefficients):
        ln_U += a * (ln_T**i)

    U = float(np.exp(ln_U))
    if g0 is not None:
        U = max(U, float(g0))
    return U


def irwin_log10_to_ln_coeffs(b: List[float]) -> List[float]:
    """Convert Irwin-style (log10/log10) coefficients to natural-log basis.

    If Irwin tabulates::

        log10 U(T) = Σ_{n=0..4} b_n · (log10 T)^n

    then because ln x = ln(10)·log10 x, the equivalent natural-log
    expansion ``ln U = Σ a_n (ln T)^n`` has::

        a_n = ln(10)^(1-n) · b_n

    Apply this once per published Irwin row before storing into the
    ``partition_functions`` table (whose stored a_n are natural-log).

    Parameters
    ----------
    b : list of float
        Irwin (1981) Table II coefficients [b0, b1, b2, b3, b4]
        in the log10 basis.

    Returns
    -------
    list of float
        Equivalent coefficients [a0, a1, a2, a3, a4] in the natural-log
        basis, suitable for direct insertion into ``partition_functions``.
    """
    ln10 = float(np.log(10.0))
    return [bn * ln10 ** (1 - n) for n, bn in enumerate(b)]


if HAS_JAX:

    @jit
    def _polynomial_partition_function_jax_core(
        T_K: Union[float, jnp.ndarray], coefficients: jnp.ndarray
    ) -> jnp.ndarray:
        """Inner JIT-compiled polynomial evaluation (no clamping)."""
        ln_T = jnp.log(jnp.maximum(T_K, 1.0))

        ln_U = (
            coefficients[..., 0]
            + coefficients[..., 1] * ln_T
            + coefficients[..., 2] * (ln_T**2)
            + coefficients[..., 3] * (ln_T**3)
            + coefficients[..., 4] * (ln_T**4)
        )

        return jnp.exp(ln_U)

    def polynomial_partition_function_jax(
        T_K: Union[float, jnp.ndarray],
        coefficients: jnp.ndarray,
        *,
        t_min: Optional[Union[float, jnp.ndarray]] = None,
        t_max: Optional[Union[float, jnp.ndarray]] = None,
        g0: Optional[Union[float, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        """
        JAX-compatible evaluation of partition function.

        Parameters
        ----------
        T_K : float or array
            Temperature in Kelvin
        coefficients : array
            Coefficients [a0, a1, a2, a3, a4]
            Can be shape (5,) or (N, 5)
        t_min, t_max : float or array, optional
            Validity range of the polynomial fit.  When supplied, ``T_K`` is
            clamped to ``[t_min, t_max]`` before evaluation (broadcasts
            against ``T_K``).  See the NumPy twin for the rationale.
        g0 : float or array, optional
            Ground-state degeneracy.  Result is floored at ``g0`` (broadcasts
            against the output).

        Returns
        -------
        array
            Partition function value U(T)
        """
        # Clamp T to validity window (if provided).  Kwargs are evaluated
        # eagerly in Python so this wrapper stays jit-friendly without
        # needing static_argnames.
        T_clamped = T_K
        if t_min is not None and t_max is not None:
            T_clamped = jnp.clip(T_K, t_min, t_max)
        elif t_min is not None:
            T_clamped = jnp.maximum(T_K, t_min)
        elif t_max is not None:
            T_clamped = jnp.minimum(T_K, t_max)

        U = _polynomial_partition_function_jax_core(T_clamped, coefficients)
        if g0 is not None:
            U = jnp.maximum(U, g0)
        return U

else:

    def polynomial_partition_function_jax(*args, **kwargs):
        raise ImportError("JAX not installed")


class PartitionFunctionEvaluator:
    """Partition function evaluation with direct summation (preferred) and polynomial fallback.

    The recommended workflow is:

    1. Call :meth:`evaluate_direct` with energy-level arrays from the database.
       This gives exact results with plasma-aware truncation.
    2. If energy levels are unavailable, fall back to :meth:`evaluate` with
       polynomial coefficients (legacy path, less accurate).
    """

    @staticmethod
    def evaluate_direct(
        T_K: float,
        g_levels: np.ndarray,
        E_levels_ev: np.ndarray,
        ip_ev: float,
        n_e: Optional[float] = None,
    ) -> float:
        """Evaluate via direct summation over energy levels (recommended).

        Parameters
        ----------
        T_K : float
            Temperature in Kelvin.
        g_levels, E_levels_ev : np.ndarray
            Statistical weights and energies from energy_levels table.
        ip_ev : float
            Ionization potential in eV.
        n_e : float, optional
            Electron density for Debye-Hückel IPD truncation.
        """
        return direct_sum_partition_function(T_K, g_levels, E_levels_ev, ip_ev, n_e)

    @staticmethod
    def evaluate(
        T_K: float,
        coefficients: List[float],
        *,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
        g0: Optional[float] = None,
    ) -> float:
        """Evaluate using polynomial coefficients (legacy fallback).

        Forwards optional ``t_min``/``t_max``/``g0`` extrapolation guards to
        :func:`polynomial_partition_function`.
        """
        return polynomial_partition_function(T_K, coefficients, t_min=t_min, t_max=t_max, g0=g0)

    @staticmethod
    def evaluate_batch(
        coefficients: np.ndarray,
        temperatures: np.ndarray,
    ) -> np.ndarray:
        """Evaluate partition functions using polynomial coefficients (legacy batch).

        Uses Rust acceleration when available, falls back to NumPy.

        Parameters
        ----------
        coefficients : np.ndarray
            Shape (N_species, N_coeffs) partition function coefficients.
        temperatures : np.ndarray
            Shape (N_temps,) temperatures in Kelvin.

        Returns
        -------
        np.ndarray
            Shape (N_species, N_temps) partition function values.
        """
        try:
            from cflibs._core import batch_partition_functions
        except ImportError:
            try:
                from _core import batch_partition_functions
            except ImportError:
                batch_partition_functions = None  # type: ignore[assignment]

        if batch_partition_functions is not None:
            return np.asarray(
                batch_partition_functions(
                    np.ascontiguousarray(coefficients, dtype=np.float64),
                    np.ascontiguousarray(temperatures, dtype=np.float64),
                )
            )

        # NumPy fallback
        n_species = coefficients.shape[0]
        n_temps = temperatures.shape[0]
        result = np.zeros((n_species, n_temps))
        for s in range(n_species):
            coeffs_s = list(coefficients[s])
            for t in range(n_temps):
                result[s, t] = polynomial_partition_function(float(temperatures[t]), coeffs_s)
        return result


# ---------------------------------------------------------------------------
# Provider protocol + concrete implementations  (arch candidate 4)
# ---------------------------------------------------------------------------
#
# Motivation: the legacy contract was ``polynomial_partition_function(T, coeffs,
# t_min=…, t_max=…, g0=…)``, but only half the call sites threaded the
# ``t_min``/``t_max``/``g0`` kwargs through.  The remaining call sites
# happily extrapolated the quartic-in-ln-T polynomial outside the fit window,
# producing 560× errors at high T (Ca I, U(100 000 K) = 1.14e5 vs ~200) and
# sub-unity values at low T (Nb I, U(500 K) = 0.31 vs g0 = 1).
#
# Encapsulating the bounds + floor inside a provider object centralises the
# guard logic: the coefficients never leave the provider, and every caller
# automatically picks up the validity-window clamp and the ground-state
# floor.  See ``docs/architecture/2026-05-26-architecture-review.md``
# § Candidate 4, ADR-0001 B-P7.


@runtime_checkable
class PartitionFunctionProvider(Protocol):
    """Per-species partition function with encapsulated validity bounds.

    The provider abstraction is the canonical contract for partition-function
    evaluation throughout CF-LIBS.  Implementations encapsulate

    * the polynomial coefficients (or whatever underlying form they use),
    * the validity window ``[t_min, t_max]`` outside which the underlying
      fit is unreliable, and
    * the ground-state statistical weight ``g0`` which acts as a strict
      physical lower bound on the returned value.

    Callers obtain providers via :meth:`AtomicDatabase.partition_function_for`
    and never touch the underlying coefficients.  This concentrates the
    "clamp T to [t_min, t_max], floor at g0" logic in one place — previously
    that logic was smeared across half the call sites and silently absent
    from the other half (see arch candidate 4 § Problem).
    """

    def at(self, T_K: Any, n_e: Optional[float] = None) -> Any:
        """Evaluate U(T) with bounds clamping and g0 floor applied.

        ``n_e`` (electron density, cm⁻³) optionally lowers the partition-sum
        cutoff via Debye-Hückel IPD for direct-sum providers; the polynomial
        fallback ignores it (no per-level cutoff exists for a fitted poly).
        """
        ...

    def valid_range(self) -> Tuple[float, float]:
        """Return ``(t_min, t_max)`` for the underlying fit."""
        ...

    @property
    def g0(self) -> float:
        """Ground-state statistical weight (strict physical lower bound)."""
        ...


@dataclass(frozen=True)
class DirectSumPartitionFunctionProvider:
    """CPU scalar adapter that evaluates the EXACT direct sum over energy levels.

    ``U(T) = Σ gᵢ exp(-Eᵢ / kT)`` over the species' IP-filtered, energy-sorted
    levels — i.e. the same computation :func:`direct_sum_partition_function`
    (via :meth:`PartitionFunctionEvaluator.evaluate_direct`) performed at the
    historical CPU call sites, now behind the single provider seam.

    This is the provider :meth:`AtomicDatabase.partition_function_for` vends for
    species that HAVE tabulated energy levels.  It deliberately does NOT use the
    ln-poly fit: the fit (``PartitionFunctionSpec.coefficients``) is a build-time
    artefact for the JAX batched adapter (``vmap`` needs static fixed-shape
    arrays), whereas a Python-runtime scalar caller can afford the full sum.
    Keeping the CPU path on the exact direct sum is what makes the default
    ``invert`` / ``analyze`` / iterative U(T) bit-for-bit unchanged after the
    provider unification (composition-pipeline diagnosis 2026-06-03 §2.1,
    Invariant #2).

    Like :class:`PolynomialPartitionFunctionProvider` it honours the provider
    contract: ``T`` is clamped to ``[t_min, t_max]`` (so a temperature above the
    fit window evaluates the sum at ``t_max`` rather than runaway) and the
    result is floored at ``g0``.  In the LIBS band (8000-12000 K) the clamp is a
    no-op for every workhorse species, so ``.at(T)`` reproduces ``evaluate_direct``
    to floating-point precision there.

    Attributes
    ----------
    element, ionization_stage : informational only.
    _g_levels, _e_levels_ev : tuple[float, ...]
        IP-filtered, energy-sorted statistical weights and level energies (eV).
        Stored as tuples to keep the dataclass frozen/hashable.
    ip_ev : float
        Ionization potential (eV) — the sum cutoff (already applied to the
        stored level arrays, but retained for parity with ``evaluate_direct``).
    t_min, t_max : float
        Validity window inherited from the direct-sum fit; used only for the
        clamp guard.
    _g0 : float
        Ground-state statistical weight (lower bound on U).
    source : str
        Provenance (``"direct_sum_fit"``).
    """

    element: str
    ionization_stage: int
    _g_levels: Tuple[float, ...]
    _e_levels_ev: Tuple[float, ...]
    ip_ev: float
    t_min: float
    t_max: float
    _g0: float
    source: str = ""

    @property
    def g0(self) -> float:
        return float(self._g0)

    def valid_range(self) -> Tuple[float, float]:
        return float(self.t_min), float(self.t_max)

    def at(self, T_K: Any, n_e: Optional[float] = None) -> Any:
        """Evaluate the exact direct sum with clamp + g0 floor.

        Accepts a Python float / numpy scalar / numpy ndarray.  For a scalar
        this delegates to :func:`direct_sum_partition_function`, reproducing the
        historical ``evaluate_direct`` call bit-for-bit inside the clamp window
        when ``n_e is None``.

        Parameters
        ----------
        T_K : float or np.ndarray
            Temperature(s) in Kelvin.
        n_e : float, optional
            Electron density in cm⁻³.  When provided, the direct-sum cutoff
            is lowered to ``e_max = ip - Δχ(n_e, T)`` (Debye-Hückel IPD),
            making the partition-sum truncation CONSISTENT with the
            IPD-lowered Saha ionization exponent (audit Family J).  When
            ``None`` (default) the sharp ``ip`` cutoff is used, preserving
            the historical Boltzmann-plot behaviour bit-for-bit.
        """
        g_arr = np.asarray(self._g_levels, dtype=np.float64)
        e_arr = np.asarray(self._e_levels_ev, dtype=np.float64)
        if isinstance(T_K, np.ndarray) and T_K.ndim > 0:
            T_clamped = np.clip(
                np.asarray(T_K, dtype=np.float64), float(self.t_min), float(self.t_max)
            )
            out = np.array(
                [
                    direct_sum_partition_function(
                        float(t), g_arr, e_arr, float(self.ip_ev), n_e=n_e
                    )
                    for t in T_clamped.ravel()
                ],
                dtype=np.float64,
            ).reshape(T_clamped.shape)
            return np.maximum(out, float(self._g0))
        T_clamped = float(np.clip(float(T_K), float(self.t_min), float(self.t_max)))
        u = direct_sum_partition_function(T_clamped, g_arr, e_arr, float(self.ip_ev), n_e=n_e)
        return max(float(u), float(self._g0))


@dataclass(frozen=True)
class PolynomialPartitionFunctionProvider:
    """Provider backed by a 4th-order natural-log polynomial fit (Irwin form).

    Concretely::

        ln U(clamp(T, t_min, t_max)) = a0 + a1·ln T + a2·(ln T)² + …

    with the result floored at ``g0``.  This is the production fallback used
    by every CF-LIBS call site once :meth:`AtomicDatabase.partition_function_for`
    is wired in.

    Attributes
    ----------
    element : str
        Element symbol — informational only.
    ionization_stage : int
        Ionization stage — informational only.
    coefficients : tuple[float, ...]
        Polynomial coefficients ``(a0, …, a4)`` in natural-log basis.
        Stored as a tuple to keep the dataclass frozen/hashable.
    t_min, t_max : float
        Validity range of the fit.  Outside this interval :meth:`at`
        evaluates the polynomial at the boundary.
    _g0 : float
        Ground-state statistical weight (lower bound on U).  Stored under
        a private name so the public :attr:`g0` matches the Protocol
        property contract.
    source : str
        Provenance string from the ``partition_functions`` table.
    """

    element: str
    ionization_stage: int
    coefficients: Tuple[float, ...]
    t_min: float
    t_max: float
    _g0: float
    source: str = ""

    @property
    def g0(self) -> float:
        return float(self._g0)

    def valid_range(self) -> Tuple[float, float]:
        return float(self.t_min), float(self.t_max)

    def at(self, T_K: Any, n_e: Optional[float] = None) -> Any:
        """Evaluate U(T) with clamp + g0 floor.

        Accepts a Python float / numpy scalar / numpy ndarray.  Use
        :func:`polynomial_partition_function_jax` directly (or wrap a
        :class:`BatchedPartitionFunctionProvider`) for jit-traced array
        inputs.

        ``n_e`` is accepted for provider-contract uniformity with
        :class:`DirectSumPartitionFunctionProvider` but ignored: the fitted
        polynomial has no per-level energies, so there is no IPD cutoff to
        lower.
        """
        del n_e  # not applicable to the fitted-polynomial fallback
        coeffs = list(self.coefficients)
        if isinstance(T_K, np.ndarray) and T_K.ndim > 0:
            T_arr = np.asarray(T_K, dtype=np.float64)
            T_clamped = np.clip(T_arr, float(self.t_min), float(self.t_max))
            ln_T = np.log(np.maximum(T_clamped, 1.0))
            ln_U = np.zeros_like(ln_T)
            for i, a in enumerate(coeffs):
                ln_U = ln_U + a * (ln_T**i)
            U = np.exp(ln_U)
            U = np.maximum(U, float(self._g0))
            # Match the scalar-path low-T sentinel.
            U = np.where(T_arr <= 1.0, np.maximum(1.0, float(self._g0)), U)
            return U
        return polynomial_partition_function(
            float(T_K),
            coeffs,
            t_min=float(self.t_min),
            t_max=float(self.t_max),
            g0=float(self._g0),
        )


@dataclass(frozen=True)
class BatchedPartitionFunctionProvider:
    """Vectorised provider for per-species batched evaluation under jit.

    Holds parallel arrays carrying every species' polynomial coefficients
    plus validity window and ground-state degeneracy.  Indexing by
    ``species_indices`` returns the per-species U at the requested
    temperature(s) with bounds clamping and g0 floor applied — all using
    ``jnp`` operations, so the method is jit/vmap compatible.

    This is the carrier the manifold path and the kernels' Saha-Boltzmann
    populations needed: previously they unpacked
    :class:`AtomicSnapshot.partition_coeffs` directly and called
    :func:`polynomial_partition_function_jax` without bounds, silently
    extrapolating the polynomial outside the fit window.

    Attributes
    ----------
    coefficients : array, shape (N_species, 5) or (N_species, …, 5)
        Polynomial coefficients (natural-log basis).
    t_min, t_max : array, shape (N_species,)
        Per-species validity bounds.
    g0 : array, shape (N_species,)
        Per-species ground-state degeneracies.
    """

    coefficients: Any
    t_min: Any
    t_max: Any
    g0: Any  # type: ignore[assignment]

    def valid_range(self) -> Tuple[Any, Any]:
        return self.t_min, self.t_max

    def at_all(self, T_K: Any) -> Any:
        """Evaluate U(T_K) for every species row in parallel."""
        if HAS_JAX:
            return polynomial_partition_function_jax(
                T_K,
                self.coefficients,
                t_min=self.t_min,
                t_max=self.t_max,
                g0=self.g0,
            )
        coeffs = np.asarray(self.coefficients, dtype=np.float64)
        t_min = np.asarray(self.t_min, dtype=np.float64)
        t_max = np.asarray(self.t_max, dtype=np.float64)
        g0_arr = np.asarray(self.g0, dtype=np.float64)
        T_clamped = np.clip(np.asarray(T_K, dtype=np.float64), t_min, t_max)
        ln_T = np.log(np.maximum(T_clamped, 1.0))
        powers = np.stack([np.ones_like(ln_T), ln_T, ln_T**2, ln_T**3, ln_T**4], axis=-1)
        ln_U = np.sum(coeffs * powers, axis=-1)
        return np.maximum(np.exp(ln_U), g0_arr)

    def at_batch(self, T_K: Any, species_indices: Any) -> Any:
        """Gather-and-evaluate U(T) for a subset of species rows.

        Parameters
        ----------
        T_K : scalar or array
            Temperature(s) in Kelvin.
        species_indices : array of int
            Indices into the species axis.

        Returns
        -------
        array
            ``U`` evaluated for the selected species at ``T_K``.  Same
            shape as ``species_indices``.
        """
        if HAS_JAX:
            idx = jnp.asarray(species_indices, dtype=jnp.int32)
            return polynomial_partition_function_jax(
                T_K,
                self.coefficients[idx],
                t_min=self.t_min[idx],
                t_max=self.t_max[idx],
                g0=self.g0[idx],
            )
        idx = np.asarray(species_indices, dtype=np.int64)
        coeffs = np.asarray(self.coefficients, dtype=np.float64)[idx]
        t_min = np.asarray(self.t_min, dtype=np.float64)[idx]
        t_max = np.asarray(self.t_max, dtype=np.float64)[idx]
        g0_arr = np.asarray(self.g0, dtype=np.float64)[idx]
        T_clamped = np.clip(np.asarray(T_K, dtype=np.float64), t_min, t_max)
        ln_T = np.log(np.maximum(T_clamped, 1.0))
        powers = np.stack([np.ones_like(ln_T), ln_T, ln_T**2, ln_T**3, ln_T**4], axis=-1)
        ln_U = np.sum(coeffs * powers, axis=-1)
        return np.maximum(np.exp(ln_U), g0_arr)
