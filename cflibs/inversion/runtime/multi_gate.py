"""
Joint multi-gate Saha-Boltzmann fit for time-resolved CF-LIBS.

When a LIBS plasma is sampled with several gate timings, the elemental
composition is invariant in time while the plasma state ``(T, n_e)`` evolves.
A *joint* multi-gate fit exploits this by sharing a single composition vector
across gates while letting each gate carry its own temperature, electron
density, and emission scale. Pooling per-gate residuals against one common
composition reduces composition uncertainty roughly as ``1/sqrt(N_gates)``
relative to averaging per-gate fits.

Algorithm
---------
For each line ``i`` of element ``s`` (ion stage ``z``) observed in gate ``g``,
the predicted log-intensity in the neutral Saha-Boltzmann plane is

    log I_pred(s, g, i) = alpha_g + log C_s + log(g_k A_ki / lambda)
                         - log U_{s,1}(T_g) - E_k / (k T_g)
                         + log S_{s}(T_g, n_e,g)        if z == 2

where ``alpha_g`` absorbs the per-gate emission scale (number density and
geometric factors), ``C_s`` is the shared composition (number fraction), and
``S_s`` is the Saha ratio ``n_II/n_I``. The objective is

    chi2 = sum_g sum_i  (log I_obs(s,g,i) - log I_pred(s,g,i))^2 / sigma_i^2

with ``sigma_i`` propagated from the measured intensity uncertainty
(``sigma_log = sigma_I / I``). Composition is parameterized via the ILR
transform so the simplex constraint ``sum_s C_s = 1`` is exact and the
optimization is unconstrained in the ILR coordinates.

Optimization uses ``scipy.optimize.minimize`` with L-BFGS-B (cheap for the
small number of parameters: ``3 * n_gates + (n_elements - 1)``). Per-gate
``IterativeCFLIBSSolver`` results seed the initial guess, which keeps the
search inside the convex basin around the truth.

References
----------
Multi-gate Saha-Boltzmann reconciliation is discussed in recent
Applied Physics B / Spectrochimica Acta B work; see e.g. Appl. Phys. B
(2025), https://link.springer.com/article/10.1007/s00340-025-08606-9 .
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from cflibs.core.constants import EV_TO_K, SAHA_CONST_CM3
from cflibs.core.logging_config import get_logger
from cflibs.inversion.common.data_structures import LineObservation
from cflibs.inversion.physics.closure import ilr_inverse, ilr_transform
from cflibs.inversion.runtime.temporal import TemporalGateConfig, TimeResolvedSpectrum

logger = get_logger("inversion.multi_gate")


# =============================================================================
# Result dataclass
# =============================================================================


@dataclass
class JointMultiGateResult:
    """
    Result of a joint multi-gate Saha-Boltzmann fit.

    Attributes
    ----------
    temperatures_K_per_gate : np.ndarray
        Fitted plasma temperature per gate (K), ordered by ``gate_delays_ns``.
    electron_densities_per_gate : np.ndarray
        Fitted electron density per gate (cm^-3).
    composition_shared : Dict[str, float]
        Single composition vector (number fractions, sums to 1) shared across
        all gates.
    composition_uncertainty : Dict[str, float]
        1-sigma uncertainty on each shared composition fraction, propagated
        from the inverse Hessian of the chi^2 surface at the optimum.
    chi2_total : float
        Final pooled chi^2 value.
    n_gates : int
        Number of gates included in the joint fit.
    per_gate_chi2 : np.ndarray
        Per-gate chi^2 contribution at the optimum, ordered by
        ``gate_delays_ns``.
    gate_delays_ns : np.ndarray
        Gate delay times (ns) corresponding to the per-gate arrays. Provided
        so callers can re-pair results with their original
        ``TemporalGateConfig`` objects.
    converged : bool
        Whether the optimizer reported convergence.
    n_iterations : int
        Number of L-BFGS-B iterations performed.
    elements : List[str]
        Element symbols, in the canonical order used internally for the
        composition vector and uncertainty diagonal.
    """

    temperatures_K_per_gate: np.ndarray
    electron_densities_per_gate: np.ndarray
    composition_shared: Dict[str, float]
    composition_uncertainty: Dict[str, float]
    chi2_total: float
    n_gates: int
    per_gate_chi2: np.ndarray
    gate_delays_ns: np.ndarray
    converged: bool = False
    n_iterations: int = 0
    elements: List[str] = field(default_factory=list)


# =============================================================================
# Public API
# =============================================================================


GatePayload = Tuple[TemporalGateConfig, Sequence[LineObservation]]


def joint_multi_gate_fit(
    spectra: Iterable[TimeResolvedSpectrum] | Iterable[GatePayload],
    *,
    ionization_potentials_eV: Optional[Dict[str, float]] = None,
    partition_func_I: Optional[Dict[str, float]] = None,
    partition_func_II: Optional[Dict[str, float]] = None,
    initial_temperatures_K: Optional[Sequence[float]] = None,
    initial_electron_densities_cm3: Optional[Sequence[float]] = None,
    initial_composition: Optional[Dict[str, float]] = None,
    base_solver=None,
    atomic_db=None,
    closure_mode: str = "standard",
    T_bounds_K: Tuple[float, float] = (3000.0, 30000.0),
    log_ne_bounds: Tuple[float, float] = (np.log(1e14), np.log(1e20)),
    log_alpha_bounds: Tuple[float, float] = (-50.0, 50.0),
    ilr_bound: float = 15.0,
    max_iter: int = 500,
) -> JointMultiGateResult:
    """
    Run a joint Saha-Boltzmann fit across multiple temporal gates with
    a single shared composition vector.

    Parameters
    ----------
    spectra : iterable
        Either ``TimeResolvedSpectrum`` instances or ``(TemporalGateConfig,
        list[LineObservation])`` pairs. Both forms are accepted.
    ionization_potentials_eV : dict, optional
        First-ion ionisation potentials by element symbol. If omitted and
        ``atomic_db`` is provided, IPs are looked up from the database; if
        both are absent any ion-stage lines are dropped with a warning
        (neutral lines still contribute).
    partition_func_I, partition_func_II : dict, optional
        Pre-computed neutral and singly-ionised partition functions. Both
        are treated as temperature-independent across the (modest) per-gate
        ``T`` range — adequate for the joint refinement which is seeded near
        the per-gate optimum and only needs to track *relative* changes.
        Defaults: ``25.0`` for I, ``15.0`` for II.
    initial_temperatures_K, initial_electron_densities_cm3, initial_composition :
        Optional warm starts. If omitted, the per-gate
        ``IterativeCFLIBSSolver`` is run on each gate to produce a starting
        point — this is the recommended path described in the algorithm
        docstring above.
    base_solver, atomic_db : optional
        Used only to construct the per-gate warm-start solver. Either supply
        a configured ``IterativeCFLIBSSolver`` as ``base_solver`` or pass
        ``atomic_db`` and one will be built. Required only if all of
        ``initial_temperatures_K``, ``initial_electron_densities_cm3``, and
        ``initial_composition`` are omitted.
    closure_mode : str
        Closure mode for the per-gate warm-start solver.
    T_bounds_K, log_ne_bounds, log_alpha_bounds, ilr_bound :
        L-BFGS-B box bounds. The defaults span the entire plausible LIBS
        operating range and rarely need adjustment.
    max_iter : int
        L-BFGS-B iteration cap.

    Returns
    -------
    JointMultiGateResult
        Fitted plasma states per gate plus the shared composition.
    """
    gate_configs, gate_observations = _normalize_inputs(spectra)
    n_gates = len(gate_configs)
    if n_gates == 0:
        raise ValueError("joint_multi_gate_fit requires at least one gate")

    # --- Element bookkeeping (canonical order) ----------------------------
    elements: List[str] = sorted(
        {obs.element for obs_list in gate_observations for obs in obs_list}
    )
    if not elements:
        raise ValueError("No line observations supplied to joint_multi_gate_fit")
    el_index = {el: i for i, el in enumerate(elements)}
    D = len(elements)

    # --- Atomic data: IPs, partition functions ----------------------------
    ips = _resolve_ips(elements, ionization_potentials_eV, atomic_db)
    U_I = (
        {el: float(partition_func_I.get(el, 25.0)) for el in elements}
        if partition_func_I
        else {el: 25.0 for el in elements}
    )
    U_II = (
        {el: float(partition_func_II.get(el, 15.0)) for el in elements}
        if partition_func_II
        else {el: 15.0 for el in elements}
    )

    # --- Warm start from per-gate iterative solver ------------------------
    (
        T_init,
        ne_init,
        comp_init,
    ) = _warm_start(
        gate_configs,
        gate_observations,
        elements,
        initial_temperatures_K,
        initial_electron_densities_cm3,
        initial_composition,
        base_solver,
        atomic_db,
        closure_mode,
    )

    # --- Pack observations into flat arrays for vectorised residuals ------
    pooled = _pack_observations(gate_observations, el_index)

    # --- Initial alphas: pick alpha_g so that mean residual is zero -------
    alpha_init = _initial_alphas(pooled, T_init, ne_init, comp_init, U_I, U_II, ips, elements)

    # --- Pack parameter vector --------------------------------------------
    # theta = [T_1..T_G, log_ne_1..log_ne_G, alpha_1..alpha_G, ilr_coords (D-1)]
    if D >= 2:
        ilr_init = ilr_transform(np.asarray(comp_init, dtype=float))
    else:
        ilr_init = np.zeros(0, dtype=float)
    theta_init = np.concatenate(
        [
            np.asarray(T_init, dtype=float),
            np.log(np.asarray(ne_init, dtype=float)),
            np.asarray(alpha_init, dtype=float),
            ilr_init,
        ]
    )

    # Bounds
    bounds: List[Tuple[float, float]] = []
    bounds.extend([T_bounds_K] * n_gates)
    bounds.extend([log_ne_bounds] * n_gates)
    bounds.extend([log_alpha_bounds] * n_gates)
    bounds.extend([(-ilr_bound, ilr_bound)] * (D - 1))

    # --- Optimize ---------------------------------------------------------
    from scipy.optimize import minimize

    def chi2_fn(theta: np.ndarray) -> float:
        return _chi2(theta, pooled, n_gates, D, U_I, U_II, ips, elements)

    result = minimize(
        chi2_fn,
        theta_init,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-10, "gtol": 1e-7},
    )

    theta_opt = result.x
    T_opt = theta_opt[:n_gates].copy()
    ne_opt = np.exp(theta_opt[n_gates : 2 * n_gates])
    if D >= 2:
        ilr_opt = theta_opt[3 * n_gates :]
        comp_opt = ilr_inverse(ilr_opt, D)
    else:
        comp_opt = np.array([1.0])

    chi2_total = float(result.fun)
    per_gate = _per_gate_chi2(theta_opt, pooled, n_gates, D, U_I, U_II, ips, elements)

    # --- Composition uncertainty via finite-difference Hessian -----------
    comp_unc = _composition_uncertainty(
        chi2_fn,
        theta_opt,
        n_gates=n_gates,
        D=D,
        n_observations=pooled["n_obs"],
    )

    composition_shared = {el: float(comp_opt[i]) for i, el in enumerate(elements)}
    composition_uncertainty = {el: float(comp_unc[i]) for i, el in enumerate(elements)}

    logger.info(
        "joint_multi_gate_fit: n_gates=%d, elements=%s, chi2=%.3g, converged=%s",
        n_gates,
        elements,
        chi2_total,
        bool(result.success),
    )

    return JointMultiGateResult(
        temperatures_K_per_gate=T_opt,
        electron_densities_per_gate=ne_opt,
        composition_shared=composition_shared,
        composition_uncertainty=composition_uncertainty,
        chi2_total=chi2_total,
        n_gates=n_gates,
        per_gate_chi2=per_gate,
        gate_delays_ns=np.array([g.delay_ns for g in gate_configs], dtype=float),
        converged=bool(result.success),
        n_iterations=int(result.nit),
        elements=list(elements),
    )


# =============================================================================
# Internal helpers
# =============================================================================


def _normalize_inputs(
    spectra: Iterable[TimeResolvedSpectrum] | Iterable[GatePayload],
) -> Tuple[List[TemporalGateConfig], List[List[LineObservation]]]:
    """Accept either ``TimeResolvedSpectrum`` or ``(gate, observations)`` tuples."""
    gates: List[TemporalGateConfig] = []
    observations: List[List[LineObservation]] = []
    for item in spectra:
        if isinstance(item, TimeResolvedSpectrum):
            gates.append(item.gate)
            observations.append(list(item.observations))
        else:
            gate, obs = item
            if not isinstance(gate, TemporalGateConfig):
                raise TypeError(
                    "Expected TemporalGateConfig in (gate, observations) tuple, got "
                    f"{type(gate).__name__}"
                )
            gates.append(gate)
            observations.append(list(obs))
    # Sort by gate delay so per-gate arrays line up with chronological order
    order = sorted(range(len(gates)), key=lambda i: gates[i].delay_ns)
    gates = [gates[i] for i in order]
    observations = [observations[i] for i in order]
    return gates, observations


def _resolve_ips(
    elements: Sequence[str],
    ips: Optional[Dict[str, float]],
    atomic_db,
) -> Dict[str, float]:
    """Look up first-ion IPs from explicit dict or atomic database, with fallback."""
    out: Dict[str, float] = {}
    for el in elements:
        if ips and el in ips:
            out[el] = float(ips[el])
            continue
        if atomic_db is not None:
            try:
                value = atomic_db.get_ionization_potential(el, 1)
            except Exception:  # noqa: BLE001 — DB backends raise heterogeneous errors
                value = None
            if value is not None:
                out[el] = float(value)
                continue
        logger.warning("No IP for %s; defaulting to 15.0 eV (ion lines may bias fit)", el)
        out[el] = 15.0
    return out


def _warm_start(
    gate_configs: Sequence[TemporalGateConfig],
    gate_observations: Sequence[Sequence[LineObservation]],
    elements: Sequence[str],
    T_init_user: Optional[Sequence[float]],
    ne_init_user: Optional[Sequence[float]],
    comp_init_user: Optional[Dict[str, float]],
    base_solver,
    atomic_db,
    closure_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the joint-fit warm start. Falls back to per-gate iterative CF-LIBS
    when the caller hasn't supplied an explicit guess.
    """
    n_gates = len(gate_configs)

    need_per_gate = T_init_user is None or ne_init_user is None or comp_init_user is None

    T_init = np.full(n_gates, 10000.0)
    ne_init = np.full(n_gates, 1e17)
    comp_init = np.full(len(elements), 1.0 / len(elements))

    if need_per_gate:
        if base_solver is None and atomic_db is None:
            # No warm-start data and no way to build it: use defaults.
            logger.info(
                "joint_multi_gate_fit: no atomic_db/base_solver supplied; "
                "using neutral defaults (T=10000 K, ne=1e17, uniform composition)"
            )
        else:
            solver = base_solver
            if solver is None:
                from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver

                solver = IterativeCFLIBSSolver(atomic_db)

            running_comp = np.zeros(len(elements))
            comp_weight = 0.0
            for g_idx, obs_list in enumerate(gate_observations):
                if not obs_list:
                    continue
                try:
                    res = solver.solve(list(obs_list), closure_mode=closure_mode)
                except Exception as exc:  # noqa: BLE001 — fall back to defaults
                    logger.warning(
                        "Per-gate warm start failed for gate %d (%s); using defaults",
                        g_idx,
                        exc,
                    )
                    continue
                T_init[g_idx] = float(res.temperature_K)
                ne_init[g_idx] = float(res.electron_density_cm3)
                gate_vec = np.array(
                    [res.concentrations.get(el, 0.0) for el in elements], dtype=float
                )
                if gate_vec.sum() > 0:
                    running_comp += gate_vec / gate_vec.sum()
                    comp_weight += 1.0
            if comp_weight > 0:
                comp_init = running_comp / comp_weight

    if T_init_user is not None:
        T_arr = np.asarray(T_init_user, dtype=float)
        if T_arr.size != n_gates:
            raise ValueError("initial_temperatures_K must have length n_gates")
        T_init = T_arr
    if ne_init_user is not None:
        ne_arr = np.asarray(ne_init_user, dtype=float)
        if ne_arr.size != n_gates:
            raise ValueError("initial_electron_densities_cm3 must have length n_gates")
        ne_init = ne_arr
    if comp_init_user is not None:
        comp_init = np.array([comp_init_user.get(el, 0.0) for el in elements], dtype=float)
        s = comp_init.sum()
        if s <= 0:
            raise ValueError("initial_composition must contain at least one positive entry")
        comp_init = comp_init / s

    # Safety floor: ensure no zero composition entries (ILR requires positives)
    comp_init = np.clip(comp_init, 1e-6, None)
    comp_init = comp_init / comp_init.sum()

    return T_init, ne_init, comp_init


def _pack_observations(
    gate_observations: Sequence[Sequence[LineObservation]],
    el_index: Dict[str, int],
) -> Dict[str, np.ndarray]:
    """Flatten all (gate, line) observations into NumPy arrays for vectorisation."""
    gate_idx: List[int] = []
    el_idx: List[int] = []
    ion_stage: List[int] = []
    log_y: List[float] = []  # ln(I * lambda / (g_k * A_ki))
    sigma_log: List[float] = []
    E_k: List[float] = []
    for g, obs_list in enumerate(gate_observations):
        for obs in obs_list:
            if obs.intensity <= 0 or obs.intensity_uncertainty <= 0:
                continue
            if obs.element not in el_index:
                continue
            gate_idx.append(g)
            el_idx.append(el_index[obs.element])
            ion_stage.append(int(obs.ionization_stage))
            log_y.append(float(obs.y_value))
            sigma_log.append(float(obs.y_uncertainty) or 1.0)
            E_k.append(float(obs.E_k_ev))

    if not gate_idx:
        raise ValueError("No usable line observations after filtering")

    return {
        "gate_idx": np.array(gate_idx, dtype=int),
        "el_idx": np.array(el_idx, dtype=int),
        "ion_stage": np.array(ion_stage, dtype=int),
        "log_y": np.array(log_y, dtype=float),
        "sigma_log": np.array(sigma_log, dtype=float),
        "E_k": np.array(E_k, dtype=float),
        "n_obs": len(gate_idx),
    }


def _predicted_log_y(
    T_per_gate: np.ndarray,
    log_ne_per_gate: np.ndarray,
    alpha_per_gate: np.ndarray,
    composition: np.ndarray,
    pooled: Dict[str, np.ndarray],
    U_I: Dict[str, float],
    U_II: Dict[str, float],
    ips: Dict[str, float],
    elements: Sequence[str],
) -> np.ndarray:
    """
    Vectorised prediction for each (gate, line) pair.

    Returns the predicted ``ln(I * lambda / (g_k * A_ki))`` value -- the same
    ``y`` quantity stored on ``LineObservation``. Composition enters as
    ``log C_s`` and the per-gate ``alpha_g`` absorbs the absolute scale.
    """
    g_idx = pooled["gate_idx"]
    s_idx = pooled["el_idx"]
    z_arr = pooled["ion_stage"]
    E_k = pooled["E_k"]

    T_g = T_per_gate[g_idx]
    log_ne_g = log_ne_per_gate[g_idx]
    alpha_g = alpha_per_gate[g_idx]
    log_C_s = np.log(np.clip(composition[s_idx], 1e-30, None))

    # Per-element atomic constants (broadcast through indexing arrays)
    log_U_I = np.array([np.log(U_I[el]) for el in elements])[s_idx]
    log_U_II = np.array([np.log(U_II[el]) for el in elements])[s_idx]
    ip_arr = np.array([ips[el] for el in elements])[s_idx]

    T_eV = np.maximum(T_g / EV_TO_K, 1e-3)

    # Neutral plane: y = alpha_g + log C_s - log U_I - E_k / T_eV
    y_pred = alpha_g + log_C_s - log_U_I - E_k / T_eV

    # Ion lines (z == 2): add Saha term log[(SAHA_CONST / n_e) * T_eV^1.5 * U_II/U_I * exp(-IP/T_eV)]
    # which after factoring gives:
    #   y_pred_ion = y_pred_neutral + log_S
    # where log_S = log(SAHA_CONST_CM3) - log_ne + 1.5*log T_eV + (log U_II - log U_I) - IP/T_eV
    is_ion = z_arr == 2
    if np.any(is_ion):
        log_S = (
            np.log(SAHA_CONST_CM3)
            - log_ne_g[is_ion]
            + 1.5 * np.log(T_eV[is_ion])
            + (log_U_II[is_ion] - log_U_I[is_ion])
            - ip_arr[is_ion] / T_eV[is_ion]
        )
        y_pred[is_ion] = y_pred[is_ion] + log_S
    return y_pred


def _initial_alphas(
    pooled: Dict[str, np.ndarray],
    T_init: np.ndarray,
    ne_init: np.ndarray,
    comp_init: np.ndarray,
    U_I: Dict[str, float],
    U_II: Dict[str, float],
    ips: Dict[str, float],
    elements: Sequence[str],
) -> np.ndarray:
    """Pick per-gate alpha so the mean prediction matches the mean observation."""
    n_gates = T_init.size
    alphas = np.zeros(n_gates)
    # Compute predictions with alpha=0 then take per-gate mean residual.
    y_pred0 = _predicted_log_y(
        T_init,
        np.log(ne_init),
        np.zeros(n_gates),
        comp_init,
        pooled,
        U_I,
        U_II,
        ips,
        elements,
    )
    residual = pooled["log_y"] - y_pred0
    g_idx = pooled["gate_idx"]
    for g in range(n_gates):
        mask = g_idx == g
        if np.any(mask):
            w = 1.0 / np.maximum(pooled["sigma_log"][mask] ** 2, 1e-12)
            alphas[g] = float(np.sum(w * residual[mask]) / np.sum(w))
    return alphas


def _unpack_theta(
    theta: np.ndarray, n_gates: int, D: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode the flat parameter vector into named per-gate arrays + composition."""
    T_per_gate = theta[:n_gates]
    log_ne_per_gate = theta[n_gates : 2 * n_gates]
    alpha_per_gate = theta[2 * n_gates : 3 * n_gates]
    if D >= 2:
        ilr_coords = theta[3 * n_gates :]
        composition = ilr_inverse(ilr_coords, D)
    else:
        composition = np.array([1.0])
    return T_per_gate, log_ne_per_gate, alpha_per_gate, composition


def _chi2(
    theta: np.ndarray,
    pooled: Dict[str, np.ndarray],
    n_gates: int,
    D: int,
    U_I: Dict[str, float],
    U_II: Dict[str, float],
    ips: Dict[str, float],
    elements: Sequence[str],
) -> float:
    T_per_gate, log_ne_per_gate, alpha_per_gate, composition = _unpack_theta(theta, n_gates, D)
    y_pred = _predicted_log_y(
        T_per_gate,
        log_ne_per_gate,
        alpha_per_gate,
        composition,
        pooled,
        U_I,
        U_II,
        ips,
        elements,
    )
    residual = pooled["log_y"] - y_pred
    sigma = pooled["sigma_log"]
    return float(np.sum((residual / sigma) ** 2))


def _per_gate_chi2(
    theta: np.ndarray,
    pooled: Dict[str, np.ndarray],
    n_gates: int,
    D: int,
    U_I: Dict[str, float],
    U_II: Dict[str, float],
    ips: Dict[str, float],
    elements: Sequence[str],
) -> np.ndarray:
    T_per_gate, log_ne_per_gate, alpha_per_gate, composition = _unpack_theta(theta, n_gates, D)
    y_pred = _predicted_log_y(
        T_per_gate,
        log_ne_per_gate,
        alpha_per_gate,
        composition,
        pooled,
        U_I,
        U_II,
        ips,
        elements,
    )
    residual = pooled["log_y"] - y_pred
    sigma = pooled["sigma_log"]
    sq = (residual / sigma) ** 2
    out = np.zeros(n_gates)
    for g in range(n_gates):
        mask = pooled["gate_idx"] == g
        out[g] = float(np.sum(sq[mask]))
    return out


def _composition_uncertainty(
    chi2_fn,
    theta_opt: np.ndarray,
    *,
    n_gates: int,
    D: int,
    n_observations: int,
) -> np.ndarray:
    """
    Estimate 1-sigma uncertainty on each shared composition fraction.

    Strategy: build a finite-difference Hessian of ``chi2`` in ILR space at
    the optimum, invert it to get the parameter covariance (inverse Hessian
    of ``chi2/2``), take the ILR-space covariance block, then propagate to
    composition space via the analytic Jacobian ``d C / d ilr``.

    For poorly conditioned cases (singular Hessian, single element) the
    routine returns zeros — the caller should still report a useful nominal
    composition.
    """
    n_params = theta_opt.size
    if D < 2:
        return np.zeros(D)

    ilr_offset = 3 * n_gates
    n_ilr = D - 1
    ilr_idx = np.arange(ilr_offset, ilr_offset + n_ilr)

    # Step size: small relative to bound width, large enough to avoid noise
    eps = np.full(n_params, 1e-3)
    eps[:n_gates] = 5.0  # T in K
    eps[n_gates : 2 * n_gates] = 1e-3  # log_ne
    eps[2 * n_gates : 3 * n_gates] = 1e-3  # alpha
    eps[ilr_offset:] = 1e-3  # ilr coords

    # Build full Hessian (small; n_params is typically <= ~30)
    H = np.zeros((n_params, n_params))
    f0 = chi2_fn(theta_opt)
    for i in range(n_params):
        for j in range(i, n_params):
            ei = np.zeros(n_params)
            ei[i] = eps[i]
            ej = np.zeros(n_params)
            ej[j] = eps[j]
            f_pp = chi2_fn(theta_opt + ei + ej)
            f_pm = chi2_fn(theta_opt + ei - ej)
            f_mp = chi2_fn(theta_opt - ei + ej)
            f_mm = chi2_fn(theta_opt - ei - ej)
            H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps[i] * eps[j])
            H[j, i] = H[i, j]

    # Parameter covariance for chi^2: 2 * H^{-1} (since fisher = H/2 for
    # gaussian residuals with sigma^2 already in chi^2). Reduced chi^2 scaling
    # absorbs over/under-estimated noise.
    dof = max(n_observations - n_params, 1)
    reduced_chi2 = max(f0 / dof, 1e-30)
    try:
        H_inv = np.linalg.pinv(H)
    except np.linalg.LinAlgError:
        return np.zeros(D)

    cov = 2.0 * H_inv * reduced_chi2
    cov_ilr = cov[np.ix_(ilr_idx, ilr_idx)]

    # Jacobian dC / d ilr (D x D-1) computed by finite difference
    ilr_opt = theta_opt[ilr_offset:]
    composition_opt = ilr_inverse(ilr_opt, D)
    jac = np.zeros((D, n_ilr))
    for k in range(n_ilr):
        delta = np.zeros(n_ilr)
        h = 1e-4
        delta[k] = h
        comp_plus = ilr_inverse(ilr_opt + delta, D)
        comp_minus = ilr_inverse(ilr_opt - delta, D)
        jac[:, k] = (comp_plus - comp_minus) / (2.0 * h)

    cov_comp = jac @ cov_ilr @ jac.T
    diag = np.diag(cov_comp)
    diag = np.where(diag > 0, diag, 0.0)
    sigma = np.sqrt(diag)
    # composition_opt unused except to keep parity with sigma indexing
    _ = composition_opt
    return sigma
