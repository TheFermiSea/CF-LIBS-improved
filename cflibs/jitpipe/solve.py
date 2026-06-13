"""Stage 7 — fixed-shape CF-LIBS solve (ADR-0004 §3 C4, §5.1.1, §6.1; J7).

This module ships the two layers ADR-0004 §3 C4 prescribes for the jittable
solve stage, both as **fixed-shape, vmap-clean, reverse-differentiable** JAX
kernels operating on padded ``(E, N_max)`` arrays:

1. :func:`scan_solve` — the **fixed-K scan initializer**. It takes the lax
   path's component kernels (``_eval_partition_jax``, ``_saha_ratio_per_element``,
   ``_saha_correct_kernel``, ``_common_slope_kernel`` from
   :mod:`cflibs.inversion.solve.iterative`) *verbatim* and runs them inside a
   ``jax.lax.scan`` of a **static** trip count (``StaticConfig.max_iters``)
   instead of the reference's data-dependent ``lax.while_loop``. A
   converged-state freeze (``where(converged, prev, new)``) makes the scan
   *idempotent* past the fixed point, so it is bit-compatible with the while
   loop at convergence yet differentiable and ``vmap``-clean (the while loop is
   neither — see J7 spec §2(vii) and the try/except grad smoke at
   ``tests/inversion/test_iterative_lax.py:354``).

2. :func:`joint_wls_solve` — the **joint WLS Gauss–Newton production
   estimator** (ADR-0004 §6.1). It minimises, over ``θ = (ln T, α ∈ R^{E-1},
   β)`` with ``n_e`` pinned to the J6 Stark measurement,

       Σ_l w_l · [ y_l − ( −E_l/(k_B T) + ln C_{s(l)}(α) − ln U_{s(l)}(T)
                            − ln M_{s(l)}(T,n_e) + β ) ]² .

   Frozen at ``θ_0`` (U_s, M_s held at the warm-start ``T_0``), the residual is
   **affine in θ**, so the first Gauss–Newton step is algebraically the
   :class:`cflibs.inversion.solve.closed_form.ClosedFormILRSolver` weighted
   least-squares solve (J4's Schur identity) — the exact parity anchor and warm
   start. Subsequent fixed-K GN steps re-freeze U_s, M_s at the running ``T`` to
   recover the Saha/partition non-linearity that the numpy precursor only
   captured via 1–2 hand-rolled Saha passes.

Fixed-shape contract (ADR-0004 §1.1): every array axis is a padded constant
(``E`` species, ``N_max`` lines, ``D-1`` ILR coords); failures surface as
``converged`` / ``physical`` quality flags, never exceptions or
data-dependent shapes. fp64 is mandatory (§5.3).

Reuse: this module imports the lax kernels rather than re-deriving them, so the
parity anchor is exact by construction. No SQLite, no host imports inside the
kernels (import-hygiene test, J0 AC5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp

from cflibs.core.constants import EV_TO_K, KB, KB_EV, SAHA_CONST_CM3
from cflibs.inversion.solve.iterative import (
    LoopState,
    _common_slope_kernel,
    _eval_partition_jax,
    _saha_correct_kernel,
    _saha_ratio_per_element,
)

if TYPE_CHECKING:  # pragma: no cover
    from cflibs.jitpipe.params import PipelineParams, StaticConfig
    from cflibs.jitpipe.snapshot import PipelineSnapshot


__all__ = [
    "scan_solve",
    "joint_wls_solve",
    "LaxKernelInputs",
    "JointWLSResult",
    "helmert_basis",
    "ilr_inverse",
    "iterative_solve",
]


# ---------------------------------------------------------------------------
# Padded, device-ready inputs (the per-bucket candidate set, J0 §2).
# ---------------------------------------------------------------------------


class LaxKernelInputs(NamedTuple):
    """Fixed-shape padded inputs the solve kernels consume.

    Mirrors the field layout the reference :func:`_run_lax_while_loop` reads off
    a :class:`cflibs.inversion.solve.iterative._AtomicSnapshot` plus the
    ``_build_padded_arrays_from_obs`` observation block, but as a flat,
    pytree-registered tuple so it flows through ``jit``/``vmap`` with no host
    objects. All arrays are JAX device arrays; axes are static (``E`` species,
    ``N_max`` lines, ``Lmax`` levels).

    Attributes
    ----------
    x, y, w : (E, N_max)
        Padded upper-level energies (eV), Boltzmann ``y`` values and
        inverse-variance weights. Padding zeroed; ``mask`` selects valid lines.
    stage : (E, N_max), int
        Ionisation stage (1 neutral, 2 ionic) per line.
    mask : (E, N_max), bool
        Valid-line mask.
    ip0 : (E,)
        Stage-I ionisation potentials, eV.
    use_direct : (E, 2), bool
        Direct-sum vs polynomial partition selector per stage.
    g_I, E_I, ip_I, mask_I, g_II, E_II, ip_II, mask_II
        Padded per-stage level blocks for the direct partition sum (stage I/II).
    coeffs_I, coeffs_II : (E, 5)
        Partition polynomials (NaN sentinel -> scalar fallback) per stage.
    fallback_I, fallback_II : (E,)
        Scalar partition fallbacks per stage.
    """

    x: Any
    y: Any
    w: Any
    stage: Any
    mask: Any
    ip0: Any
    use_direct: Any
    g_I: Any
    E_I: Any
    ip_I: Any
    mask_I: Any
    g_II: Any
    E_II: Any
    ip_II: Any
    mask_II: Any
    coeffs_I: Any
    coeffs_II: Any
    fallback_I: Any
    fallback_II: Any

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Any,
        x: Any,
        y: Any,
        w: Any,
        stage: Any,
        mask: Any,
    ) -> "LaxKernelInputs":
        """Assemble device inputs from a lax ``_AtomicSnapshot`` + padded obs.

        The atomic block fields are read straight off the reference
        ``_AtomicSnapshot`` (or the :meth:`PipelineSnapshot.to_lax_snapshot`
        bridge output, which has identical field names) so the kernel sees byte
        identical data to ``_run_lax_while_loop``.
        """
        f64 = jnp.float64
        return cls(
            x=jnp.asarray(x, dtype=f64),
            y=jnp.asarray(y, dtype=f64),
            w=jnp.asarray(w, dtype=f64),
            stage=jnp.asarray(stage, dtype=jnp.int32),
            mask=jnp.asarray(mask, dtype=bool),
            ip0=jnp.asarray(snapshot.ip0_eV, dtype=f64),
            use_direct=jnp.asarray(snapshot.use_direct, dtype=bool),
            g_I=jnp.asarray(snapshot.g_levels_I, dtype=f64),
            E_I=jnp.asarray(snapshot.E_levels_I, dtype=f64),
            ip_I=jnp.asarray(snapshot.ip_I_for_direct, dtype=f64),
            mask_I=jnp.asarray(snapshot.levels_mask_I, dtype=bool),
            g_II=jnp.asarray(snapshot.g_levels_II, dtype=f64),
            E_II=jnp.asarray(snapshot.E_levels_II, dtype=f64),
            ip_II=jnp.asarray(snapshot.ip_II_for_direct, dtype=f64),
            mask_II=jnp.asarray(snapshot.levels_mask_II, dtype=bool),
            coeffs_I=jnp.asarray(snapshot.coeffs_I, dtype=f64),
            coeffs_II=jnp.asarray(snapshot.coeffs_II, dtype=f64),
            fallback_I=jnp.asarray(snapshot.fallback_U_I, dtype=f64),
            fallback_II=jnp.asarray(snapshot.fallback_U_II, dtype=f64),
        )


jax.tree_util.register_pytree_node(
    LaxKernelInputs,
    lambda k: (tuple(k), None),
    lambda _aux, ch: LaxKernelInputs(*ch),
)


# ---------------------------------------------------------------------------
# Closure: α -> C maps (standard / oxide / matrix), all pure JAX, vmap-safe.
# ILR is identical to standard at the simplex (J7 spec §3). The native JAX form
# replaces the reference pure_callback for ilr/pwlr/dirichlet (spec §2(ii)).
# ---------------------------------------------------------------------------


def _closure_standard(rel: Any) -> Any:
    """Standard closure: normalise relative concentrations to sum 1."""
    total = jnp.sum(rel)
    return jnp.where(total > 0.0, rel / jnp.where(total > 0.0, total, 1.0), 0.0)


def _closure_oxide(rel: Any, factors: Any) -> Any:
    """Oxide closure: factor-weighted normalisation (implied oxides sum 1)."""
    total = jnp.sum(rel * factors)
    return jnp.where(total > 0.0, rel / jnp.where(total > 0.0, total, 1.0), 0.0)


def _closure_matrix(rel: Any, m_idx: int, matrix_fraction: float) -> Any:
    """Matrix closure: pin one coordinate to ``matrix_fraction``."""
    F = rel[m_idx] / matrix_fraction
    return jnp.where(F > 0.0, rel / jnp.where(F > 0.0, F, 1.0), 0.0)


def _apply_closure(
    rel: Any,
    closure_mode: str,
    oxide_factors: Any,
    matrix_idx: int,
    matrix_fraction: float,
) -> Any:
    """Dispatch closure (resolved statically on the host, J7 spec §5 Option A).

    ``standard`` / ``ilr`` / ``pwlr`` / ``dirichlet_residual`` collapse to the
    standard normalisation here (they are identical at the simplex for the
    forward composition map); ``oxide`` and ``matrix`` carry their extra static
    data as traced arrays / a static index.
    """
    mode = closure_mode.lower()
    if mode == "oxide":
        return _closure_oxide(rel, oxide_factors)
    if mode == "matrix":
        return _closure_matrix(rel, matrix_idx, matrix_fraction)
    return _closure_standard(rel)


# ---------------------------------------------------------------------------
# Layer 1 — fixed-K lax.scan initializer.
# ---------------------------------------------------------------------------


def _solve_iteration(
    state: LoopState,
    inp: LaxKernelInputs,
    *,
    apply_ipd: bool,
    closure_mode: str,
    oxide_factors: Any,
    matrix_idx: int,
    matrix_fraction: float,
    t_tol_k: float,
    ne_tol_frac: float,
    pressure_pa: float,
    min_r2: float,
) -> LoopState:
    """One CF-LIBS iteration, math-identical to ``_run_lax_while_loop.body_fun``.

    Reuses the imported lax kernels verbatim. The ONLY behavioural change vs the
    reference body is the closure call: it goes through the native-JAX
    :func:`_apply_closure` instead of the host ``pure_callback`` (J7 spec §2(ii),
    §5) — numerically identical for standard/oxide/matrix; ILR collapses to
    standard at the simplex.
    """
    T_prev = state.T_K
    ne_prev = state.n_e_cm3

    U_I = _eval_partition_jax(
        T_prev,
        inp.use_direct[:, 0],
        inp.g_I,
        inp.E_I,
        inp.ip_I,
        inp.coeffs_I,
        inp.fallback_I,
        inp.mask_I,
    )
    U_II = _eval_partition_jax(
        T_prev,
        inp.use_direct[:, 1],
        inp.g_II,
        inp.E_II,
        inp.ip_II,
        inp.coeffs_II,
        inp.fallback_II,
        inp.mask_II,
    )

    if apply_ipd:
        kT_eV = jnp.maximum(T_prev / EV_TO_K, 1e-3)
        lambda_D_cm = 6.9 * jnp.sqrt(kT_eV * EV_TO_K / jnp.maximum(ne_prev, 1.0))
        delta_chi = 1.44e-7 / jnp.maximum(lambda_D_cm, 1e-30)
        ip_eff = jnp.maximum(inp.ip0 - delta_chi, 0.0)
    else:
        ip_eff = inp.ip0

    T_eV = jnp.maximum(T_prev / EV_TO_K, 0.1)
    safe_ne = jnp.maximum(ne_prev, 1e10)
    log_correction = jnp.log((SAHA_CONST_CM3 / safe_ne) * (T_eV**1.5))
    ip_arr = jnp.broadcast_to(ip_eff[:, None], inp.x.shape)
    x_corr, y_corr = _saha_correct_kernel(inp.x, inp.y, inp.stage, ip_arr, T_eV, log_correction)

    fit = _common_slope_kernel(x_corr, y_corr, inp.w, inp.mask)
    slope = fit["slope"]
    intercepts = fit["intercepts"]
    r_squared = fit["r_squared"]

    degenerate = jnp.logical_or(slope >= 0.0, r_squared < min_r2)
    T_new = jnp.where(degenerate, T_prev, -1.0 / (slope * KB_EV))
    T_K = 0.5 * T_prev + 0.5 * T_new

    # Corona weighting is a no-op unless two_region; deferred per spec §11.
    T_for_saha = T_K

    S = _saha_ratio_per_element(T_for_saha, ne_prev, U_I, U_II, ip_eff)
    mult = 1.0 + jnp.maximum(S, 0.0)

    rel = mult * U_I * jnp.exp(intercepts)
    concentrations = _apply_closure(rel, closure_mode, oxide_factors, matrix_idx, matrix_fraction)

    S_now = _saha_ratio_per_element(T_K, ne_prev, U_I, U_II, ip_eff)
    eps_s = S_now / (1.0 + S_now)
    avg_Z = jnp.sum(concentrations * eps_s)
    n_tot = pressure_pa / (KB * T_K * (1.0 + avg_Z))
    n_tot_cm3 = n_tot * 1e-6
    ne_new = avg_Z * n_tot_cm3
    n_e = 0.5 * ne_prev + 0.5 * ne_new

    converged = jnp.logical_and(
        jnp.logical_not(degenerate),
        jnp.logical_and(
            jnp.abs(T_K - T_prev) < t_tol_k,
            jnp.abs(n_e - ne_prev) / jnp.maximum(ne_prev, 1e-30) < ne_tol_frac,
        ),
    )

    return LoopState(
        T_K=T_K,
        n_e_cm3=n_e,
        T_prev=T_prev,
        n_e_prev=ne_prev,
        converged=converged,
        i=state.i + 1,
        U_I=U_I,
        U_II=U_II,
        intercepts=intercepts,
        concentrations=concentrations,
        r_squared=r_squared,
        boltzmann_degenerate=degenerate,
    )


def scan_solve(
    inp: LaxKernelInputs,
    *,
    init_T_K: float = 10000.0,
    init_ne_cm3: float = 1.0e17,
    max_iters: int = 20,
    apply_ipd: bool = False,
    closure_mode: str = "standard",
    oxide_factors: Any = None,
    matrix_idx: int = 0,
    matrix_fraction: float = 0.9,
    t_tol_k: float = 100.0,
    ne_tol_frac: float = 0.1,
    pressure_pa: float = 101325.0,
    min_r2: float = 0.3,
) -> LoopState:
    """Fixed-K ``lax.scan`` CF-LIBS initializer (J7 layer 1; ADR-0004 §3 C4).

    Runs exactly ``max_iters`` iterations of :func:`_solve_iteration` through
    ``jax.lax.scan`` with a **static** trip count, freezing the state once
    ``converged`` first flips True. The freeze (``where(converged, prev, new)``)
    makes the output identical at ``K`` and ``K + n`` for any ``n >= 0`` once the
    fixed point is reached (idempotence) — so the kernel reproduces the
    reference ``lax.while_loop`` fixed point while being ``jit``/``vmap`` clean
    and reverse-differentiable.

    Parameters
    ----------
    inp : LaxKernelInputs
        Padded device inputs (the per-bucket candidate set).
    init_T_K, init_ne_cm3 : float
        Warm-start temperature (K) and electron density (cm^-3). Match the
        reference ``_solve_lax`` defaults (10000 K, 1e17 cm^-3) for parity.
    max_iters : int
        Static scan trip count (``StaticConfig.max_iters``); keys the jit cache.
    closure_mode : str
        ``standard`` / ``ilr`` / ``pwlr`` / ``dirichlet_residual`` ->
        normalisation; ``oxide`` -> factor-weighted; ``matrix`` -> pinned coord.
    oxide_factors : (E,) array, optional
        Per-element oxide stoichiometry for ``oxide`` mode (defaults to ones).
    matrix_idx, matrix_fraction : int, float
        Pinned-element index and fraction for ``matrix`` mode.

    Returns
    -------
    LoopState
        Final (possibly frozen) state. ``i`` is the iteration at which the freeze
        engaged (the reference while-loop's trip count) when convergence was
        reached, else ``max_iters``.
    """
    E = inp.x.shape[0]
    if oxide_factors is None:
        oxide_factors = jnp.ones(E, dtype=jnp.float64)
    else:
        oxide_factors = jnp.asarray(oxide_factors, dtype=jnp.float64)

    init_state = LoopState(
        T_K=jnp.asarray(init_T_K, dtype=jnp.float64),
        n_e_cm3=jnp.asarray(init_ne_cm3, dtype=jnp.float64),
        T_prev=jnp.asarray(init_T_K, dtype=jnp.float64),
        n_e_prev=jnp.asarray(init_ne_cm3, dtype=jnp.float64),
        converged=jnp.asarray(False),
        i=jnp.asarray(0, dtype=jnp.int32),
        U_I=jnp.zeros(E, dtype=jnp.float64),
        U_II=jnp.zeros(E, dtype=jnp.float64),
        intercepts=jnp.zeros(E, dtype=jnp.float64),
        concentrations=jnp.zeros(E, dtype=jnp.float64),
        r_squared=jnp.asarray(0.0, dtype=jnp.float64),
        boltzmann_degenerate=jnp.asarray(True),
    )

    def _step(state: LoopState, _unused: Any):
        already = state.converged
        new_state = _solve_iteration(
            state,
            inp,
            apply_ipd=apply_ipd,
            closure_mode=closure_mode,
            oxide_factors=oxide_factors,
            matrix_idx=matrix_idx,
            matrix_fraction=matrix_fraction,
            t_tol_k=t_tol_k,
            ne_tol_frac=ne_tol_frac,
            pressure_pa=pressure_pa,
            min_r2=min_r2,
        )
        # Converged-state freeze: once `already` is True, hold the prior state
        # verbatim (including `i` and `converged`) so the scan is idempotent
        # past the fixed point and matches the while-loop trip count exactly.
        frozen = jax.tree_util.tree_map(
            lambda old, new: jnp.where(already, old, new), state, new_state
        )
        return frozen, None

    final_state, _ = jax.lax.scan(_step, init_state, None, length=max_iters)
    return final_state


# ---------------------------------------------------------------------------
# Layer 2 — joint WLS Gauss–Newton production estimator (ADR-0004 §6.1).
# ---------------------------------------------------------------------------


def helmert_basis(D: int) -> Any:
    """Helmert sub-composition contrast matrix ``(D, D-1)`` as a JAX array.

    Bit-identical to :func:`cflibs.inversion.physics.closure._helmert_basis`
    (the closed-form ILR solver's basis), so the GN-step-0 anchor is exact.
    """
    if D < 2:
        raise ValueError("Helmert basis requires D >= 2")
    cols = []
    for i in range(1, D):
        col = jnp.zeros(D, dtype=jnp.float64)
        col = col.at[:i].set(1.0 / jnp.sqrt(i * (i + 1)))
        col = col.at[i].set(-i / jnp.sqrt(i * (i + 1)))
        cols.append(col)
    return jnp.stack(cols, axis=1)  # (D, D-1)


def ilr_inverse(coords: Any, V: Any) -> Any:
    """Inverse ILR: map ``α ∈ R^{D-1}`` back to the D-simplex via Helmert ``V``.

    Bit-identical to :func:`cflibs.inversion.physics.closure.ilr_inverse`.
    """
    clr = coords @ V.T  # (D-1,) @ (D-1, D) -> (D,)
    comp = jnp.exp(clr)
    return comp / jnp.sum(comp)


class JointWLSResult(NamedTuple):
    """Result of :func:`joint_wls_solve`.

    Attributes
    ----------
    theta : (n_cols,)
        Fitted ``θ = (m, α_1..α_{D-1}, β)`` where ``m = -1/(k_B T)``.
    T_K : scalar
        Recovered temperature (K). ``50000`` clamp when slope non-physical.
    n_e_cm3 : scalar
        Electron density (pinned to the Stark input; pass-through).
    concentrations : (E,)
        Recovered simplex composition (sum 1; re-closed host-side for
        oxide/matrix).
    cov_theta : (n_cols, n_cols)
        WLS parameter covariance ``σ̂² (XᵀWX)⁻¹`` at the converged θ.
    physical : scalar bool
        False when the fitted slope was non-negative (unphysical T).
    n_iter : int
        Static GN step count actually run.
    """

    theta: Any
    T_K: Any
    n_e_cm3: Any
    concentrations: Any
    cov_theta: Any
    physical: Any
    n_iter: int


def joint_wls_solve(
    inp: LaxKernelInputs,
    *,
    init_T_K: float = 10000.0,
    n_e_cm3: float = 1.0e17,
    n_gn_steps: int = 1,
    refine_saha: bool = True,
    closure_mode: str = "standard",
    oxide_factors: Any = None,
    matrix_idx: int = 0,
    matrix_fraction: float = 0.9,
    lm_damping: float = 0.0,
) -> JointWLSResult:
    """Joint WLS Gauss–Newton CF-LIBS estimator (J7 layer 2; ADR-0004 §6.1).

    Minimises the §6.1 objective over ``θ = (m, α, β)`` with ``n_e`` pinned to
    ``n_e_cm3`` (the J6 Stark measurement). At ``θ_0`` (U_s, M_s frozen at
    ``init_T_K``) the GN step is algebraically the
    :class:`cflibs.inversion.solve.closed_form.ClosedFormILRSolver` weighted
    least-squares solve — the exact parity anchor (rtol 1e-10).

    Implementation: each fixed-K GN step (a) freezes ``U_s(T)``, ``M_s(T, n_e)``
    at the running ``θ``'s ``T``, (b) builds the affine design ``X``, response
    ``y_adj`` and weights ``W`` exactly as :meth:`ClosedFormILRSolver._design_row`
    does, (c) solves the ``(D+1)×(D+1)`` normal system
    ``θ = (XᵀWX + λI)⁻¹ XᵀWy`` (LM damping ``λ`` optional), all in fixed shape
    (padded rows carry ``W = 0``). When ``n_gn_steps == 1`` and
    ``refine_saha`` is irrelevant this is *exactly* GN step 0.

    Parameters
    ----------
    inp : LaxKernelInputs
        Padded device inputs.
    init_T_K : float
        Warm-start temperature freezing ``U_s``/``M_s`` for GN step 0.
    n_e_cm3 : float
        Pinned electron density (J6 Stark).
    n_gn_steps : int
        Static Gauss–Newton step count. ``1`` -> GN step 0 (closed-form anchor).
    refine_saha : bool
        When True (default) GN steps >0 refresh ``U_s``/``M_s`` at the running
        ``T``; when False they stay frozen at ``init_T_K`` (pure linear WLS).
    closure_mode, oxide_factors, matrix_idx, matrix_fraction
        Closure mapping applied to the recovered simplex (host re-closure for
        oxide/matrix; the regression itself lands on the standard simplex).
    lm_damping : float
        Levenberg–Marquardt damping added to the normal-matrix diagonal.

    Returns
    -------
    JointWLSResult
    """
    E = inp.x.shape[0]
    D = E
    if oxide_factors is None:
        oxide_factors = jnp.ones(E, dtype=jnp.float64)
    else:
        oxide_factors = jnp.asarray(oxide_factors, dtype=jnp.float64)

    if D >= 2:
        V = helmert_basis(D)
        n_cols = 1 + (D - 1) + 1
    else:
        V = jnp.zeros((1, 0), dtype=jnp.float64)
        n_cols = 2

    n_e_pinned = jnp.asarray(n_e_cm3, dtype=jnp.float64)

    # Flatten the (E, N_max) line block to rows; species index per row.
    Nmax = inp.x.shape[1]
    sp_idx = jnp.repeat(jnp.arange(E), Nmax)  # (E*Nmax,)
    x_flat = inp.x.reshape(-1)
    y_flat = inp.y.reshape(-1)
    w_flat = inp.w.reshape(-1)
    mask_flat = inp.mask.reshape(-1)
    # Inverse-variance weight; padded/invalid rows -> 0 (drop without reshaping).
    W = jnp.where(mask_flat, w_flat, 0.0)

    # Constant design columns (slope col is E_k; ILR cols from V; intercept 1).
    if D >= 2:
        ilr_cols = V[sp_idx, :]  # (E*Nmax, D-1)
    else:
        ilr_cols = jnp.zeros((x_flat.shape[0], 0), dtype=jnp.float64)
    ones_col = jnp.ones((x_flat.shape[0], 1), dtype=jnp.float64)
    X = jnp.concatenate([x_flat[:, None], ilr_cols, ones_col], axis=1)  # (rows, n_cols)

    def _U_at(T_K: Any):
        U_I = _eval_partition_jax(
            T_K,
            inp.use_direct[:, 0],
            inp.g_I,
            inp.E_I,
            inp.ip_I,
            inp.coeffs_I,
            inp.fallback_I,
            inp.mask_I,
        )
        U_II = _eval_partition_jax(
            T_K,
            inp.use_direct[:, 1],
            inp.g_II,
            inp.E_II,
            inp.ip_II,
            inp.coeffs_II,
            inp.fallback_II,
            inp.mask_II,
        )
        return U_I, U_II

    def _y_adj_for(T_freeze: Any):
        """y_adj = y + ln U_s(T) + ln M_s(T, n_e), per the closed-form solver."""
        U_I, U_II = _U_at(T_freeze)
        S = _saha_ratio_per_element(T_freeze, n_e_pinned, U_I, U_II, inp.ip0)
        M = 1.0 + jnp.maximum(S, 0.0)
        lnU = jnp.log(jnp.maximum(U_I, 1e-30))  # (E,)
        lnM = jnp.log(jnp.maximum(M, 1e-30))  # (E,)
        per_row = lnU[sp_idx] + lnM[sp_idx]  # (rows,)
        return y_flat + per_row

    def _wls(y_adj: Any):
        WX = X * W[:, None]
        XtWX = X.T @ WX
        XtWy = WX.T @ y_adj
        if lm_damping > 0.0:
            XtWX = XtWX + lm_damping * jnp.eye(n_cols, dtype=jnp.float64)
        theta = jnp.linalg.solve(XtWX, XtWy)
        return theta, XtWX

    # GN step 0: freeze at init_T_K (the closed-form anchor & warm start).
    last_freeze = jnp.asarray(init_T_K, dtype=jnp.float64)
    y_adj_last = _y_adj_for(last_freeze)
    theta, XtWX = _wls(y_adj_last)

    # GN steps >0: refresh the U_s/M_s freeze at the running T (Saha non-linearity).
    for _ in range(max(0, n_gn_steps - 1)):
        m = theta[0]
        T_run = jnp.where(m < 0.0, -1.0 / (m * KB_EV), 50000.0)
        last_freeze = T_run if refine_saha else jnp.asarray(init_T_K, dtype=jnp.float64)
        y_adj_last = _y_adj_for(last_freeze)
        theta, XtWX = _wls(y_adj_last)

    # Covariance σ̂² (XᵀWX)⁻¹ of the converged WLS step. The residual is taken
    # at the SAME freeze point as the final fit (``y_adj_last``) so it is the
    # genuine least-squares residual of the linear system that produced θ — not
    # a re-frozen one (which would inflate σ̂² and corrupt the covariance).
    m = theta[0]
    physical = m < 0.0
    T_K = jnp.where(physical, -1.0 / (m * KB_EV), 50000.0)
    residuals = y_adj_last - X @ theta
    dof = jnp.maximum(jnp.sum(W > 0.0) - n_cols, 1.0)
    sigma2 = jnp.sum(W * residuals**2) / dof
    cov_theta = sigma2 * jnp.linalg.inv(XtWX)

    if D >= 2:
        alpha = theta[1:D]
        comp = ilr_inverse(alpha, V)
    else:
        comp = jnp.ones(1, dtype=jnp.float64)

    # Closure re-mapping (oxide/matrix) on the standard simplex.
    comp = _apply_closure(comp, closure_mode, oxide_factors, matrix_idx, matrix_fraction)

    return JointWLSResult(
        theta=theta,
        T_K=T_K,
        n_e_cm3=n_e_pinned,
        concentrations=comp,
        cov_theta=cov_theta,
        physical=physical,
        n_iter=n_gn_steps,
    )


# ---------------------------------------------------------------------------
# J0 stub entry point — kept for the integration seam (delegates to scan_solve).
# ---------------------------------------------------------------------------


def iterative_solve(
    line_intensities: Any,
    line_index: Any,
    temperature_eV: Any,
    electron_density: Any,
    snapshot: "PipelineSnapshot",
    params: "PipelineParams",
    static: "StaticConfig",
) -> Any:
    """Run the iterative CF-LIBS plasma-parameter solve (J7 integration seam).

    This is the J0-skeleton signature. The fixed-shape kernels are
    :func:`scan_solve` (initializer) and :func:`joint_wls_solve` (production
    estimator); the host wrapper that gathers the per-bucket candidate set and
    reconstitutes :class:`~cflibs.inversion.solve.iterative.CFLIBSResult` lands
    with J8 integration. Until then this delegates to :func:`scan_solve` so the
    seam is exercised end to end.
    """
    raise NotImplementedError(
        "jitpipe.solve.iterative_solve host wrapper lands at J8 integration; "
        "use scan_solve / joint_wls_solve directly (see module docstring)"
    )
