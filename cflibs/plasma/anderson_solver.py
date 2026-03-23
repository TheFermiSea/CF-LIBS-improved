"""
Anderson-accelerated Saha-Boltzmann charge-balance solver in JAX.

Implements Anderson acceleration (Walker & Ni 2011, unconstrained formulation)
for the fixed-point iteration g(log n_e) -> log n_e arising from charge
neutrality in an LTE plasma.  Falls back to Picard iteration when the
Anderson depth m=0.

# ASSERT_CONVENTION: n_e [cm^-3], T_eV [eV], C_i dimensionless sum-to-1,
#   SAHA_CONST_CM3 from cflibs.core.constants

References
----------
Walker, H.F. & Ni, P. (2011). "Anderson Acceleration for Fixed-Point
    Iterations." SIAM J. Numer. Anal. 49(4), 1715-1735.
Evans, C. et al. (2018). "A Proof That Anderson Acceleration Improves the
    Convergence Rate in Linearly Converging Fixed-Point Methods."
    arXiv:1810.08455.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional, Tuple

import numpy as np

from cflibs.core.constants import EV_TO_K, SAHA_CONST_CM3
from cflibs.core.jax_runtime import HAS_JAX

if HAS_JAX:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap

    from cflibs.plasma.partition import polynomial_partition_function_jax
else:
    raise ImportError("anderson_solver requires JAX. Install with: pip install jax jaxlib")


# ---------------------------------------------------------------------------
# Physical constants in JAX
# ---------------------------------------------------------------------------
_SAHA_CONST = jnp.float64(SAHA_CONST_CM3)
_EV_TO_K = jnp.float64(EV_TO_K)

# Safeguarding clamps on log(n_e)
_LOG_NE_MIN = jnp.log(jnp.float64(1e12))  # n_e = 1e12 cm^-3
_LOG_NE_MAX = jnp.log(jnp.float64(1e20))  # n_e = 1e20 cm^-3


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
class AndersonSolverResult(NamedTuple):
    """Result of Anderson-accelerated charge-balance iteration.

    Attributes
    ----------
    n_e : jnp.ndarray
        Converged electron density [cm^-3].
    iterations : jnp.ndarray
        Number of iterations performed (int-valued).
    residual : jnp.ndarray
        Final absolute residual |g(x) - x| in log-space.
    converged : jnp.ndarray
        Whether the solver converged (bool-valued).
    residual_history : jnp.ndarray
        Residual at each iteration (padded with zeros beyond final iteration).
    """

    n_e: jnp.ndarray
    iterations: jnp.ndarray
    residual: jnp.ndarray
    converged: jnp.ndarray
    residual_history: jnp.ndarray


# ---------------------------------------------------------------------------
# Atomic data helper
# ---------------------------------------------------------------------------
class AtomicDataJAX(NamedTuple):
    """JAX-compatible atomic data for the Anderson solver.

    All arrays are padded to uniform shape across elements.

    Attributes
    ----------
    ionization_potentials : jnp.ndarray
        Shape (N_elements, max_transitions).  IP in eV for each ionization
        transition.  Index 0 = neutral->singly ionized, etc.
    partition_coefficients : jnp.ndarray
        Shape (N_elements, max_species, 5).  Irwin polynomial coefficients
        for each species (neutral, singly ionized, ...).
    n_stages : jnp.ndarray
        Shape (N_elements,).  Number of *species* per element (e.g., 3 means
        neutral + singly + doubly ionized).
    """

    ionization_potentials: jnp.ndarray
    partition_coefficients: jnp.ndarray
    n_stages: jnp.ndarray


def prepare_atomic_data_jax(
    elements: list[str],
    atomic_db: Any,
    max_stages: int = 3,
) -> AtomicDataJAX:
    """Extract atomic data from the database into JAX arrays.

    Parameters
    ----------
    elements : list of str
        Element symbols (e.g., ["Fe", "Cu"]).
    atomic_db : AtomicDataSource
        An atomic database with ``get_ionization_potential`` and
        ``get_partition_coefficients`` methods.
    max_stages : int
        Maximum number of *species* per element (default 3: neutral,
        singly ionized, doubly ionized).

    Returns
    -------
    AtomicDataJAX
        Packed JAX arrays ready for the Anderson solver.
    """
    n_elem = len(elements)
    max_transitions = max_stages - 1
    ip_arr = np.zeros((n_elem, max_transitions), dtype=np.float64)
    pf_arr = np.zeros((n_elem, max_stages, 5), dtype=np.float64)
    ns_arr = np.zeros(n_elem, dtype=np.int32)

    for i, elem in enumerate(elements):
        n_species = 0
        for stage in range(1, max_stages + 1):
            # Get partition function for this species
            pf = None
            if hasattr(atomic_db, "get_partition_coefficients"):
                pf = atomic_db.get_partition_coefficients(elem, stage)
            if pf is not None:
                coeffs = list(pf.coefficients)
                while len(coeffs) < 5:
                    coeffs.append(0.0)
                pf_arr[i, stage - 1, :] = coeffs[:5]
            else:
                # Default: log(U) ~ log(2) (ground state degeneracy ~ 2)
                pf_arr[i, stage - 1, 0] = np.log(2.0)

            n_species = stage

            # Get IP for transition stage -> stage+1
            if stage < max_stages:
                ip = atomic_db.get_ionization_potential(elem, stage)
                if ip is not None and ip > 0:
                    ip_arr[i, stage - 1] = ip
                else:
                    break

        ns_arr[i] = n_species

    return AtomicDataJAX(
        ionization_potentials=jnp.array(ip_arr),
        partition_coefficients=jnp.array(pf_arr),
        n_stages=jnp.array(ns_arr),
    )


# ---------------------------------------------------------------------------
# Saha fixed-point map: compute mean charge z_bar(T, n_e)
# ---------------------------------------------------------------------------


def _compute_mean_charge(
    log_ne: jnp.ndarray,
    T_eV: jnp.ndarray,
    compositions: jnp.ndarray,
    atomic_data: AtomicDataJAX,
) -> jnp.ndarray:
    """Compute composition-weighted mean ionic charge <z>.

    For each element, compute ionization stage populations from the Saha
    equation, then mean charge z_bar = sum_z z * f_z.  Weight by
    compositions: <z> = sum_i C_i * z_bar_i.

    Parameters
    ----------
    log_ne : scalar
        log(n_e) with n_e in cm^-3.
    T_eV : scalar
        Temperature in eV.
    compositions : array (N_elements,)
        Number fractions C_i (sum = 1).
    atomic_data : AtomicDataJAX
        Packed atomic data.

    Returns
    -------
    scalar
        Composition-weighted mean charge <z>.
    """
    n_e = jnp.exp(log_ne)
    T_K = T_eV * _EV_TO_K

    ip = atomic_data.ionization_potentials  # (N_elem, max_transitions)
    pf_coeffs = atomic_data.partition_coefficients  # (N_elem, max_species, 5)
    n_stages_arr = atomic_data.n_stages  # (N_elem,)

    n_elem = compositions.shape[0]
    max_species = pf_coeffs.shape[1]
    max_transitions = ip.shape[1]

    # Evaluate all partition functions at once
    pf_flat = pf_coeffs.reshape(-1, 5)
    U_flat = polynomial_partition_function_jax(T_K, pf_flat)
    U_all = U_flat.reshape(n_elem, max_species)

    # Saha prefactor (same for all elements/transitions)
    saha_prefactor = _SAHA_CONST / n_e * T_eV**1.5

    def _element_mean_charge(elem_idx: jnp.ndarray) -> jnp.ndarray:
        elem_ip = ip[elem_idx]  # (max_transitions,)
        elem_U = U_all[elem_idx]  # (max_species,)
        elem_n_species = n_stages_arr[elem_idx]

        # Compute cumulative Saha products via scan
        # P[0]=1 (neutral), P[z] = S[0]*...*S[z-1] (relative population of stage z)
        def _scan_prod(carry: jnp.ndarray, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            # Saha ratio for transition t -> t+1
            u_low = jnp.maximum(elem_U[t], 1e-30)
            u_high = elem_U[t + 1]
            ip_val = elem_ip[t]
            s = saha_prefactor * u_high / u_low * jnp.exp(-ip_val / jnp.maximum(T_eV, 1e-10))
            valid = (t < (elem_n_species - 1)) & (ip_val > 0)
            new_carry = jnp.where(valid, carry * s, 0.0)
            return new_carry, new_carry

        _, P_tail = jax.lax.scan(_scan_prod, jnp.float64(1.0), jnp.arange(max_transitions))
        # P_tail[t] is relative population of stage t+1
        populations = jnp.concatenate([jnp.array([1.0]), P_tail])[:max_species]

        # Mask invalid species
        species_valid = jnp.arange(max_species) < elem_n_species
        populations = jnp.where(species_valid, populations, 0.0)

        # Normalize
        total = jnp.sum(populations)
        fractions = populations / jnp.maximum(total, 1e-300)

        # Mean charge
        charges = jnp.arange(max_species, dtype=jnp.float64)
        return jnp.sum(charges * fractions)

    z_bars = vmap(_element_mean_charge)(jnp.arange(n_elem))
    mean_z = jnp.sum(compositions * z_bars)
    return jnp.maximum(mean_z, 1e-30)


# ---------------------------------------------------------------------------
# Core Anderson iteration
# ---------------------------------------------------------------------------


def _make_anderson_solver(m: int, max_iter: int):  # noqa: C901
    """Create a JIT-compiled Anderson solver for given (m, max_iter).

    The Anderson acceleration follows Walker & Ni (2011), Eq. 2.2
    (unconstrained formulation).  For the scalar (1D) fixed-point problem
    x = g(x), the update at step k with history depth mk is:

        x_{k+1} = g(x_k) - sum_{j=1}^{mk} gamma_j * (Delta_G_j + Delta_R_j)

    where Delta_R_j = r_k - r_{k-j}, Delta_G_j = g_k - g_{k-j},
    and gamma solves the regularized LS problem:

        min_gamma  ||r_k - Delta_R gamma||^2 + lambda ||gamma||^2

    Parameters
    ----------
    m : int
        Anderson depth (0 = Picard).
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    callable
        JIT-compiled solver function.
    """
    buf_size = max(m + 1, 2)  # Need at least 2 for any history

    @jit
    def _solve(
        T_eV: jnp.ndarray,
        compositions: jnp.ndarray,
        atomic_data: AtomicDataJAX,
        log_n_total_ion: jnp.ndarray,
        log_ne_init: jnp.ndarray,
        tol: jnp.ndarray,
        tikhonov_lambda: jnp.ndarray,
    ) -> AndersonSolverResult:

        def g(log_ne: jnp.ndarray) -> jnp.ndarray:
            """Fixed-point map: log_ne -> log(n_total_ion * <z>(T, n_e))."""
            mean_z = _compute_mean_charge(log_ne, T_eV, compositions, atomic_data)
            return jnp.clip(log_n_total_ion + jnp.log(mean_z), _LOG_NE_MIN, _LOG_NE_MAX)

        # Step 0: initial Picard step
        x0 = jnp.clip(log_ne_init, _LOG_NE_MIN, _LOG_NE_MAX)
        gx0 = g(x0)
        r0 = gx0 - x0

        # Circular buffers: store (x_k, g(x_k), r_k) at each step
        X_buf = jnp.zeros(buf_size, dtype=jnp.float64).at[0].set(x0)
        G_buf = jnp.zeros(buf_size, dtype=jnp.float64).at[0].set(gx0)
        R_buf = jnp.zeros(buf_size, dtype=jnp.float64).at[0].set(r0)

        res_history = jnp.zeros(max_iter, dtype=jnp.float64).at[0].set(jnp.abs(r0))

        # State: (k, x_k, X_buf, G_buf, R_buf, converged, res_hist, prev_res)
        init_state = (
            jnp.int32(1),
            gx0,  # first iterate is g(x0) = Picard step
            X_buf,
            G_buf,
            R_buf,
            jnp.bool_(False),
            res_history,
            jnp.abs(r0),
        )

        def cond_fn(state: tuple) -> jnp.ndarray:
            k, _, _, _, _, converged, _, _ = state
            return (k < max_iter) & (~converged)

        def body_fn(state: tuple) -> tuple:
            k, x_k, X_buf, G_buf, R_buf, _, res_hist, prev_res = state

            gx_k = g(x_k)
            r_k = gx_k - x_k

            # Write current step to circular buffer
            buf_idx = k % buf_size
            X_buf = X_buf.at[buf_idx].set(x_k)
            G_buf = G_buf.at[buf_idx].set(gx_k)
            R_buf = R_buf.at[buf_idx].set(r_k)

            abs_res = jnp.abs(r_k)
            rel_res = abs_res / jnp.maximum(jnp.abs(x_k), 1.0)
            converged = rel_res < tol

            # Effective history depth mk
            mk = jnp.minimum(k, jnp.int32(m))
            # Safeguarding: reset if residual blew up by 10x
            mk = jnp.where(abs_res > 10.0 * prev_res, jnp.int32(0), mk)

            if m == 0:
                # Pure Picard
                x_next = gx_k
            else:
                # Anderson acceleration (scalar 1D case)
                # Build delta vectors of length m, masking unused entries
                def _get_delta(j: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
                    # j is 1-indexed: Delta_j uses history point at step k-j
                    hist_idx = (buf_idx - j + buf_size) % buf_size
                    dr = r_k - R_buf[hist_idx]
                    dg = gx_k - G_buf[hist_idx]
                    return dr, dg

                js = jnp.arange(1, m + 1)
                drs, dgs = vmap(_get_delta)(js)  # each shape (m,)
                mask = js <= mk  # (m,) bool

                drs = jnp.where(mask, drs, 0.0)
                dgs = jnp.where(mask, dgs, 0.0)

                # Solve regularized LS: min ||r_k - drs @ gamma||^2 + lam ||gamma||^2
                # Normal equations: (drs^T drs + lam I) gamma = drs^T r_k
                # For scalar x, drs is a row vector; the Gram matrix is m x m
                ATA = jnp.outer(drs, drs)  # (m, m)
                ATb = drs * r_k  # (m,)

                # Regularization
                reg = tikhonov_lambda * jnp.diag(jnp.ones(m, dtype=jnp.float64))
                mask_2d = jnp.outer(mask, mask)
                eye = jnp.eye(m, dtype=jnp.float64)

                # Zero out invalid rows/cols, replace with identity
                ATA_reg = jnp.where(mask_2d, ATA + reg, eye)
                ATb = jnp.where(mask, ATb, 0.0)

                gamma = jnp.linalg.solve(ATA_reg, ATb)
                gamma = jnp.where(mask, gamma, 0.0)

                # Anderson update (Type I, Walker & Ni 2011):
                # x_{k+1} = g(x_k) - sum_j gamma_j * Delta_G_j
                # where Delta_G_j = g(x_k) - g(x_{k-j})
                x_aa = gx_k - jnp.sum(gamma * dgs)

                x_next = jnp.where(mk > 0, x_aa, gx_k)

            x_next = jnp.clip(x_next, _LOG_NE_MIN, _LOG_NE_MAX)
            res_hist = res_hist.at[k].set(abs_res)

            return (
                k + 1,
                x_next,
                X_buf,
                G_buf,
                R_buf,
                converged,
                res_hist,
                abs_res,
            )

        final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
        k_final, x_final, _, _, _, converged, res_hist, final_res = final_state

        return AndersonSolverResult(
            n_e=jnp.exp(x_final),
            iterations=k_final,
            residual=final_res,
            converged=converged,
            residual_history=res_hist,
        )

    return _solve


# Cache compiled solvers
_SOLVER_CACHE: dict[Tuple[int, int], Any] = {}


def _get_solver(m: int, max_iter: int):
    """Get or create a JIT-compiled solver for the given (m, max_iter)."""
    key = (m, max_iter)
    if key not in _SOLVER_CACHE:
        _SOLVER_CACHE[key] = _make_anderson_solver(m, max_iter)
    return _SOLVER_CACHE[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def anderson_solve(
    T_eV: float,
    compositions: jnp.ndarray,
    atomic_data: AtomicDataJAX,
    n_e_init: float = 1e16,
    n_total_ion: Optional[float] = None,
    m: int = 3,
    tol: float = 1e-10,
    max_iter: int = 50,
    tikhonov_lambda: float = 1e-6,
) -> AndersonSolverResult:
    """Solve Saha charge balance using Anderson acceleration.

    Parameters
    ----------
    T_eV : float
        Electron temperature in eV.
    compositions : array-like, shape (N_elements,)
        Number fractions C_i (must sum to 1).
    atomic_data : AtomicDataJAX
        Packed atomic data from ``prepare_atomic_data_jax``.
    n_e_init : float
        Initial guess for electron density [cm^-3].
    n_total_ion : float or None
        Total ion number density [cm^-3].  If None, estimated from n_e_init
        assuming mean charge ~1 (typical for LIBS).
    m : int
        Anderson acceleration depth (0 = Picard iteration).
    tol : float
        Convergence tolerance on relative residual in log-space.
    max_iter : int
        Maximum iterations.
    tikhonov_lambda : float
        Tikhonov regularization parameter for the LS subproblem.

    Returns
    -------
    AndersonSolverResult
        Contains converged n_e [cm^-3], iteration count, residual,
        convergence flag.
    """
    compositions = jnp.asarray(compositions, dtype=jnp.float64)
    T_eV_jax = jnp.float64(T_eV)

    if n_total_ion is None:
        n_total_ion = float(n_e_init)

    log_ne_init = jnp.log(jnp.float64(max(n_e_init, 1e12)))
    log_n_total_ion = jnp.log(jnp.float64(n_total_ion))
    tol_jax = jnp.float64(tol)
    lam = jnp.float64(tikhonov_lambda)

    solver = _get_solver(m, max_iter)
    return solver(
        T_eV_jax,
        compositions,
        atomic_data,
        log_n_total_ion,
        log_ne_init,
        tol_jax,
        lam,
    )


def picard_solve(
    T_eV: float,
    compositions: jnp.ndarray,
    atomic_data: AtomicDataJAX,
    n_e_init: float = 1e16,
    n_total_ion: Optional[float] = None,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> AndersonSolverResult:
    """Solve Saha charge balance using Picard (simple) iteration.

    This is ``anderson_solve`` with ``m=0`` -- the m=0 limiting case
    from DERV-03 that recovers standard Picard iteration.

    Parameters
    ----------
    T_eV : float
        Electron temperature in eV.
    compositions : array-like, shape (N_elements,)
        Number fractions C_i (must sum to 1).
    atomic_data : AtomicDataJAX
        Packed atomic data.
    n_e_init : float
        Initial guess for electron density [cm^-3].
    n_total_ion : float or None
        Total ion number density [cm^-3].
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    AndersonSolverResult
    """
    return anderson_solve(
        T_eV=T_eV,
        compositions=compositions,
        atomic_data=atomic_data,
        n_e_init=n_e_init,
        n_total_ion=n_total_ion,
        m=0,
        tol=tol,
        max_iter=max_iter,
    )
