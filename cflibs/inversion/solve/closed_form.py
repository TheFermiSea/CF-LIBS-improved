"""Closed-form CF-LIBS solver via Aitchison log-ratio regression.

Reformulates the Boltzmann equation + closure constraint as a single
weighted least squares solve in ILR (Isometric Log-Ratio) compositional
space, eliminating the iterative loop of the traditional CF-LIBS algorithm.

Mathematical basis:
    y_k + ln(U_s) + ln(M_s) = -E_k/(kT) + ln(C_s) + β

Decomposing ln(C_s) via ILR with Helmert basis V:
    y_adj_k = m·E_k + Σ_j V[s(k),j]·α_j + β

This is ordinary WLS with θ = [m, α_1...α_{D-1}, β].

References
----------
Egozcue, J.J. et al. (2003). "Isometric Logratio Transformations for
Compositional Data Analysis." Mathematical Geology 35(3), 279-300.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict

from cflibs.core.constants import KB, KB_EV, SAHA_CONST_CM3, STP_PRESSURE, EV_TO_K
from cflibs.atomic.database import AtomicDatabase
from cflibs.inversion.physics.boltzmann import LineObservation
from cflibs.inversion.physics.closure import (
    _helmert_basis,
    default_oxide_stoichiometry,
    ilr_inverse,
)
from cflibs.inversion.solve.iterative import CFLIBSResult
from cflibs.core.logging_config import get_logger

logger = get_logger("inversion.closed_form_solver")


@dataclass
class ClosedFormConfig:
    """Configuration for the closed-form ILR solver.

    Attributes
    ----------
    closure_mode : str
        Compositional closure applied to the ILR-recovered relative
        concentrations.  One of:

        * ``"standard"`` -- ``sum(C_s) = 1`` over detected elements (the ILR
          regression already lands on this simplex).
        * ``"matrix"`` -- fix one element to ``matrix_fraction`` (metallurgical
          matrices); requires ``matrix_element``.
        * ``"oxide"`` -- close on oxide mass for geological matrices: each
          element's relative concentration is weighted by its oxide
          stoichiometry so the implied oxides sum to 1 (default factors from
          :func:`cflibs.inversion.physics.closure.default_oxide_stoichiometry`).
    matrix_element : str, optional
        Element held fixed in ``"matrix"`` mode.
    matrix_fraction : float
        Fixed fraction for ``matrix_element`` in ``"matrix"`` mode.
    oxide_stoichiometry : dict, optional
        Per-element oxygen-per-cation factors for ``"oxide"`` mode.  If
        ``None``, the default geological table is used.
    """

    saha_passes: int = 2
    partition_refine: bool = True
    ne_mode: str = "pressure"
    pressure_pa: float = STP_PRESSURE
    apply_ipd: bool = False
    closure_mode: str = "standard"
    matrix_element: Optional[str] = None
    matrix_fraction: float = 0.9
    oxide_stoichiometry: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        valid = {"standard", "matrix", "oxide"}
        if self.closure_mode not in valid:
            raise ValueError(
                f"closure_mode must be one of {sorted(valid)}, got {self.closure_mode!r}"
            )
        if self.closure_mode == "matrix" and self.matrix_element is None:
            raise ValueError("closure_mode='matrix' requires matrix_element")
        if self.closure_mode == "matrix" and not (0.0 < self.matrix_fraction < 1.0):
            raise ValueError("matrix_fraction must be in the open interval (0, 1)")


class ClosedFormILRSolver:
    """Closed-form CF-LIBS solver using ILR compositional regression.

    Instead of iterating between Boltzmann fits and closure, this solver
    casts the entire problem as a single weighted least squares regression
    in ILR space, recovering temperature and compositions simultaneously.
    """

    def __init__(
        self,
        atomic_db: AtomicDatabase,
        config: Optional[ClosedFormConfig] = None,
    ):
        self.atomic_db = atomic_db
        self.config = config or ClosedFormConfig()
        if self.config.saha_passes not in (1, 2):
            raise ValueError("saha_passes must be 1 or 2")

    # ------------------------------------------------------------------
    # Private helpers (partition functions, Saha)
    # ------------------------------------------------------------------

    def _evaluate_partition_function(
        self, element: str, ionization_stage: int, T_K: float
    ) -> float:
        """Evaluate partition function through the single provider factory.

        Routes U(T) through :meth:`AtomicDatabase.partition_function_for` (the
        one source of the direct-sum-preferred, always-guarded policy).  Stays
        bit-for-bit identical to the prior direct-sum path for species with
        levels; the hardcoded estimates cover only species the factory cannot
        resolve.
        """
        provider = self.atomic_db.partition_function_for(element, ionization_stage)
        if provider is not None:
            return float(provider.at(T_K))
        if ionization_stage == 1:
            return 25.0
        if ionization_stage == 2:
            return 15.0
        return 2.0

    def _compute_saha_ratio(
        self,
        element: str,  # kept for API parity with IterativeCFLIBSSolver
        T_K: float,
        n_e_cm3: float,
        U_I: float,
        U_II: float,
        ip_ev: float,
    ) -> float:
        """Compute n_II / n_I using the first Saha ratio."""
        safe_ne = max(float(n_e_cm3), 1e10)
        T_eV = max(T_K / EV_TO_K, 0.1)
        return (SAHA_CONST_CM3 / safe_ne) * (T_eV**1.5) * (U_II / U_I) * np.exp(-ip_ev / T_eV)

    def _compute_abundance_multipliers(
        self,
        elements: List[str],
        T_K: float,
        n_e_cm3: float,
        partition_funcs_I: Dict[str, float],
        partition_funcs_II: Dict[str, float],
        ips: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute M_s = 1 + S_s for each element."""
        multipliers: Dict[str, float] = {}
        for el in elements:
            U_I = partition_funcs_I.get(el, 25.0)
            U_II = partition_funcs_II.get(el, 15.0)
            S = self._compute_saha_ratio(el, T_K, n_e_cm3, U_I, U_II, ips[el])
            multipliers[el] = 1.0 + max(S, 0.0)
        return multipliers

    def _apply_saha_correction(
        self,
        obs_by_element: Dict[str, List[LineObservation]],
        T_K: float,
        n_e: float,
        ips: Dict[str, float],
    ) -> Dict[str, List[LineObservation]]:
        """Map ionic lines to the neutral energy plane via Saha-Boltzmann."""
        T_eV = max(T_K / EV_TO_K, 0.1)
        safe_ne = max(n_e, 1e10)
        correction_term = np.log((SAHA_CONST_CM3 / safe_ne) * (T_eV**1.5))
        scale = np.exp(-correction_term)
        corrected: Dict[str, List[LineObservation]] = defaultdict(list)

        for el, obs_list in obs_by_element.items():
            ip = ips.get(el, 15.0)
            for obs in obs_list:
                if obs.ionization_stage == 2:
                    corrected[el].append(
                        LineObservation(
                            wavelength_nm=obs.wavelength_nm,
                            intensity=obs.intensity * scale,
                            intensity_uncertainty=obs.intensity_uncertainty * scale,
                            element=obs.element,
                            ionization_stage=obs.ionization_stage,
                            E_k_ev=obs.E_k_ev + ip,
                            g_k=obs.g_k,
                            A_ki=obs.A_ki,
                        )
                    )
                elif obs.ionization_stage == 1:
                    corrected[el].append(
                        LineObservation(
                            wavelength_nm=obs.wavelength_nm,
                            intensity=obs.intensity,
                            intensity_uncertainty=obs.intensity_uncertainty,
                            element=obs.element,
                            ionization_stage=obs.ionization_stage,
                            E_k_ev=obs.E_k_ev,
                            g_k=obs.g_k,
                            A_ki=obs.A_ki,
                        )
                    )
                else:
                    logger.warning(
                        "Ionization stage %d for %s not supported; skipping",
                        obs.ionization_stage,
                        el,
                    )

        return dict(corrected)

    # ------------------------------------------------------------------
    # Core regression methods
    # ------------------------------------------------------------------

    def _build_design_matrix(
        self,
        corrected_obs: Dict[str, List[LineObservation]],
        element_order: List[str],
        partition_funcs: Dict[str, float],
        abundance_multipliers: Dict[str, float],
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Build WLS design matrix, response vector, and weight vector.

        Returns
        -------
        tuple or None
            (X, y_adj, W) where X has shape (N, D+1), y_adj shape (N,),
            W shape (N,). Returns None if insufficient valid data.
        """
        D = len(element_order)
        el_to_idx = {el: i for i, el in enumerate(element_order)}

        if D >= 2:
            V = _helmert_basis(D)
            n_cols = 1 + (D - 1) + 1  # slope + ILR coords + intercept
        else:
            V = None
            n_cols = 2  # slope + intercept

        rows_X: List[np.ndarray] = []
        rows_y: List[float] = []
        rows_w: List[float] = []

        for el in element_order:
            obs_list = corrected_obs.get(el, [])
            U_s = partition_funcs.get(el, 25.0)
            M_s = abundance_multipliers.get(el, 1.0)
            s_idx = el_to_idx[el]

            for obs in obs_list:
                y_val = obs.y_value
                if not np.isfinite(y_val):
                    continue
                y_unc = obs.y_uncertainty
                if y_unc <= 0:
                    logger.debug(
                        "Non-positive uncertainty for %s at %.1f nm; using fallback 0.1",
                        obs.element,
                        obs.wavelength_nm,
                    )
                    y_unc = 0.1

                # Pre-adjust: y_adj = y + ln(U_s) + ln(M_s)
                y_adj = y_val + np.log(max(U_s, 1e-30)) + np.log(max(M_s, 1e-30))

                row = np.zeros(n_cols)
                row[0] = obs.E_k_ev
                if D >= 2:
                    assert V is not None  # guaranteed when D >= 2
                    row[1:D] = V[s_idx, :]
                row[-1] = 1.0

                rows_X.append(row)
                rows_y.append(y_adj)
                rows_w.append(1.0 / y_unc**2)

        if len(rows_X) < n_cols:
            return None

        return np.array(rows_X), np.array(rows_y), np.array(rows_w)

    @staticmethod
    def _solve_wls(
        X: np.ndarray, y: np.ndarray, W: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Solve weighted least squares: θ = (X'WX)⁻¹ X'Wy."""
        WX = X * W[:, np.newaxis]
        XtWX = X.T @ WX
        XtWy = WX.T @ y

        try:
            theta = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            logger.warning("WLS solve failed (singular matrix)")
            return None, None

        residuals = y - X @ theta
        dof = max(len(y) - len(theta), 1)
        sigma2_hat = float(np.sum(W * residuals**2)) / dof

        try:
            cov_theta = sigma2_hat * np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            cov_theta = np.full((len(theta), len(theta)), np.nan)

        return theta, cov_theta

    @staticmethod
    def _extract_parameters(
        theta: np.ndarray, D: int, element_order: List[str]
    ) -> Tuple[float, Dict[str, float], bool]:
        """Extract T and compositions from regression parameters.

        Returns
        -------
        tuple
            (T_K, compositions, physical) where physical is False if the
            fitted slope was non-negative (indicating a non-physical fit).
        """
        m = theta[0]
        physical = True
        if m >= 0:
            T_K = 50000.0
            physical = False
            logger.warning("Non-negative slope; clamping T to 50000 K")
        else:
            T_K = -1.0 / (m * KB_EV)

        if D >= 2:
            alpha = theta[1:D]
            comp_arr = ilr_inverse(alpha, D)
            compositions = {el: float(comp_arr[i]) for i, el in enumerate(element_order)}
        else:
            compositions = {element_order[0]: 1.0}

        return T_K, compositions, physical

    def _apply_closure_mode(self, compositions: Dict[str, float]) -> Dict[str, float]:
        """Re-close standard (sum=1) compositions under the configured mode.

        The ILR regression returns a standard-closed simplex
        (``sum(C_s) = 1``) that is proportional to the relative concentrations
        ``rel_C_s = M_s · U_s · exp(q_s)``.  ``matrix`` and ``oxide`` closure
        only change the *experimental factor* F that scales those relative
        concentrations, so they can be applied as a re-normalization of the
        recovered simplex without re-running the regression.

        * ``standard`` -- returned unchanged.
        * ``matrix``  -- fix ``matrix_element`` to ``matrix_fraction``; all
          other elements scale by the same factor
          ``F = rel_C_matrix / matrix_fraction`` (Bulajic-style metallurgical
          closure).  Returned fractions need not sum to 1.
        * ``oxide``   -- weight each element by its oxide stoichiometry so the
          implied oxides sum to 1; returned values are ELEMENTAL fractions
          (``sum(C_s · factor_s) = 1``).
        """
        mode = self.config.closure_mode
        if mode == "standard" or not compositions:
            return compositions

        if mode == "matrix":
            matrix_el = self.config.matrix_element
            if matrix_el not in compositions:
                logger.warning(
                    "matrix_element %s absent from solution; falling back to standard closure",
                    matrix_el,
                )
                return compositions
            # rel_C_s ∝ compositions[el]; F = rel_C_matrix / matrix_fraction.
            F = compositions[matrix_el] / self.config.matrix_fraction
            if F <= 0.0:
                return compositions
            return {el: rel / F for el, rel in compositions.items()}

        # oxide
        factors = self.config.oxide_stoichiometry
        if factors is None:
            factors = default_oxide_stoichiometry(list(compositions.keys()))
        total_oxide = sum(rel * factors.get(el, 1.0) for el, rel in compositions.items())
        if total_oxide <= 0.0:
            return compositions
        return {el: rel / total_oxide for el, rel in compositions.items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        observations: List[LineObservation],
        initial_T_K: float = 10000.0,
        initial_ne_cm3: float = 1e17,
    ) -> CFLIBSResult:
        """Solve CF-LIBS inversion in closed form via ILR regression.

        Parameters
        ----------
        observations : List[LineObservation]
            Spectral line observations to invert.
        initial_T_K : float
            Initial temperature guess for partition function evaluation.
        initial_ne_cm3 : float
            Initial electron density for Saha correction.

        Returns
        -------
        CFLIBSResult
        """
        obs_by_element: Dict[str, List[LineObservation]] = defaultdict(list)
        for obs in observations:
            obs_by_element[obs.element].append(obs)

        elements = sorted(obs_by_element.keys())
        D = len(elements)
        if D == 0:
            return self._empty_result()

        ips: Dict[str, float] = {}
        for el in elements:
            ip = self.atomic_db.get_ionization_potential(el, 1)
            ips[el] = ip if ip is not None else 15.0

        T_K = initial_T_K
        n_e = initial_ne_cm3
        effective_ips = ips

        # ── Pass 1: neutral-only temperature estimate ──────────────────
        neutral_obs: Dict[str, List[LineObservation]] = {}
        for el, obs_list in obs_by_element.items():
            neutrals = [o for o in obs_list if o.ionization_stage == 1]
            if neutrals:
                neutral_obs[el] = neutrals

        neutral_elements = sorted(neutral_obs.keys())
        if neutral_elements:
            pf_I = {el: self._evaluate_partition_function(el, 1, T_K) for el in neutral_elements}
            mult_one = {el: 1.0 for el in neutral_elements}

            result_p1 = self._build_design_matrix(neutral_obs, neutral_elements, pf_I, mult_one)
            if result_p1 is not None:
                X1, y1, W1 = result_p1
                theta1, _ = self._solve_wls(X1, y1, W1)
                if theta1 is not None:
                    T_pass1, _, _ = self._extract_parameters(
                        theta1, len(neutral_elements), neutral_elements
                    )
                    if 1000 < T_pass1 < 100000:
                        T_K = T_pass1

        # ── Pass 2: full solve with Saha correction ────────────────────
        theta: Optional[np.ndarray] = None
        cov_theta: Optional[np.ndarray] = None
        compositions: Dict[str, float] = {el: 1.0 / D for el in elements}
        X: Optional[np.ndarray] = None
        y_adj: Optional[np.ndarray] = None
        W: Optional[np.ndarray] = None
        pf_I_all: Dict[str, float] = {}
        pf_II_all: Dict[str, float] = {}
        converged = True

        if self.config.saha_passes == 2:
            T_eval = T_K if self.config.partition_refine else initial_T_K
            pf_I_all = {el: self._evaluate_partition_function(el, 1, T_eval) for el in elements}
            pf_II_all = {el: self._evaluate_partition_function(el, 2, T_eval) for el in elements}

            if self.config.apply_ipd:
                from cflibs.plasma.saha_boltzmann import (
                    ionization_potential_lowering,
                )

                delta_chi = ionization_potential_lowering(n_e, T_K)
                effective_ips = {el: max(ip - delta_chi, 0.0) for el, ip in ips.items()}

            corrected_obs = self._apply_saha_correction(obs_by_element, T_K, n_e, effective_ips)
            mult_all = self._compute_abundance_multipliers(
                elements, T_K, n_e, pf_I_all, pf_II_all, effective_ips
            )

            result_p2 = self._build_design_matrix(corrected_obs, elements, pf_I_all, mult_all)
            if result_p2 is None:
                return self._empty_result()
            X, y_adj, W = result_p2

            theta, cov_theta = self._solve_wls(X, y_adj, W)
            if theta is None:
                return self._empty_result()

            T_K, compositions, physical = self._extract_parameters(theta, D, elements)
            if not physical:
                converged = False

        else:
            # Single-pass: use neutral-only results
            pf_I_all = {el: self._evaluate_partition_function(el, 1, T_K) for el in elements}
            pf_II_all = {el: self._evaluate_partition_function(el, 2, T_K) for el in elements}
            mult_one = {el: 1.0 for el in elements}

            result_p1_full = self._build_design_matrix(
                dict(obs_by_element), elements, pf_I_all, mult_one
            )
            if result_p1_full is None:
                return self._empty_result()
            X, y_adj, W = result_p1_full

            theta, cov_theta = self._solve_wls(X, y_adj, W)
            if theta is None:
                return self._empty_result()

            T_K, compositions, physical = self._extract_parameters(theta, D, elements)
            if not physical:
                converged = False

        # ── Estimate n_e via pressure/charge balance (fixed-point) ─────
        if self.config.ne_mode == "pressure":
            for _ in range(20):
                ne_prev = n_e
                total_eps = 0.0
                for el, C_s in compositions.items():
                    U_I = pf_I_all.get(el, 25.0)
                    U_II = pf_II_all.get(el, 15.0)
                    S = self._compute_saha_ratio(el, T_K, n_e, U_I, U_II, effective_ips[el])
                    total_eps += C_s * S / (1.0 + S)
                avg_Z = total_eps
                n_tot = self.config.pressure_pa / (KB * T_K * (1.0 + avg_Z))
                n_e = avg_Z * n_tot * 1e-6  # cm^-3
                if ne_prev > 0 and abs(n_e - ne_prev) / ne_prev < 1e-4:
                    break

        # ── Uncertainties ──────────────────────────────────────────────
        T_uncertainty_K = 0.0
        concentration_uncertainties: Dict[str, float] = {}

        if theta is not None and cov_theta is not None and np.all(np.isfinite(cov_theta)):
            m = theta[0]
            var_m = cov_theta[0, 0]
            if m < 0 and var_m > 0:
                T_uncertainty_K = float(np.sqrt(var_m) / (m**2 * KB_EV))

            if D >= 2:
                V = _helmert_basis(D)
                comp_arr = np.array([compositions[el] for el in elements])
                weighted_V = comp_arr @ V  # (D-1,)
                J = np.zeros((D, D - 1))
                for i in range(D):
                    J[i, :] = comp_arr[i] * (V[i, :] - weighted_V)
                cov_alpha = cov_theta[1:D, 1:D]
                cov_C = J @ cov_alpha @ J.T
                for i, el in enumerate(elements):
                    concentration_uncertainties[el] = float(np.sqrt(max(cov_C[i, i], 0.0)))
            else:
                concentration_uncertainties = {elements[0]: 0.0}

        # ── R² ─────────────────────────────────────────────────────────
        r_squared = 0.0
        if X is not None and theta is not None and y_adj is not None and W is not None:
            y_pred = X @ theta
            ss_res = float(np.sum(W * (y_adj - y_pred) ** 2))
            y_mean = float(np.average(y_adj, weights=W))
            ss_tot = float(np.sum(W * (y_adj - y_mean) ** 2))
            r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # ── Boltzmann covariance (slope, intercept sub-matrix) ─────────
        boltz_cov = None
        if cov_theta is not None and cov_theta.shape[0] >= 2:
            idx = [0, cov_theta.shape[0] - 1]
            boltz_cov = cov_theta[np.ix_(idx, idx)]

        # ── Apply the configured compositional closure ────────────────────
        # n_e / pressure balance and the ILR uncertainty Jacobian operate on
        # the standard (sum=1) simplex above (the proper number fractions for
        # charge balance); matrix/oxide closure only rescales the *reported*
        # absolute fractions by changing the experimental factor F.
        reported_compositions = self._apply_closure_mode(compositions)
        closure_sum = sum(reported_compositions.values())

        quality_metrics: Dict[str, float] = {
            "r_squared_last": r_squared,
            "n_lines": float(len(X)) if X is not None else 0.0,
            "n_elements": float(D),
            "closure_mode_sum": float(closure_sum),
        }

        return CFLIBSResult(
            temperature_K=T_K,
            temperature_uncertainty_K=T_uncertainty_K,
            electron_density_cm3=n_e,
            concentrations=reported_compositions,
            concentration_uncertainties=concentration_uncertainties,
            iterations=self.config.saha_passes,
            converged=converged,
            quality_metrics=quality_metrics,
            electron_density_uncertainty_cm3=0.0,
            boltzmann_covariance=boltz_cov,
        )

    def solve_with_uncertainty(
        self,
        observations: List[LineObservation],
        initial_T_K: float = 10000.0,
        initial_ne_cm3: float = 1e17,
    ) -> CFLIBSResult:
        """Solve with uncertainty (always computed in closed-form solver)."""
        return self.solve(observations, initial_T_K, initial_ne_cm3)

    @staticmethod
    def _empty_result() -> CFLIBSResult:
        return CFLIBSResult(
            temperature_K=0.0,
            temperature_uncertainty_K=0.0,
            electron_density_cm3=0.0,
            concentrations={},
            concentration_uncertainties={},
            iterations=0,
            converged=False,
            quality_metrics={},
            electron_density_uncertainty_cm3=0.0,
            boltzmann_covariance=None,
        )
