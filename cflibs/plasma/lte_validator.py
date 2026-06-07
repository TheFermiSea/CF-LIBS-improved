"""
LTE validity checker for CF-LIBS plasma diagnostics.

Implements the McWhirter criterion (necessary condition for LTE) and a
simplified temporal relaxation check. Results are surfaced in CFLIBSResult
quality_metrics so users are warned when LTE assumptions may be invalid.

References
----------
- McWhirter, R.W.P. (1965) in "Plasma Diagnostic Techniques", ed. Huddlestone & Leonard
- Cristoforetti, G. et al. (2010) Spectrochim. Acta B 65, 86-95
  (notes McWhirter as necessary but not sufficient)
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from cflibs.core.constants import KB_EV, MCWHIRTER_CONST
from cflibs.core.logging_config import get_logger

logger = get_logger("plasma.lte_validator")

# Backward-compatible alias for tests/importers that still reference the old name.
MCWHIRTER_C = MCWHIRTER_CONST


@dataclass
class LTECheckResult:
    """Result of a single LTE validity check."""

    criterion: str
    satisfied: bool
    n_e_required: float
    n_e_actual: float
    ratio: float  # actual / required
    message: str


@dataclass
class LTEReport:
    """
    Full LTE validity report for a CF-LIBS result.

    Attributes
    ----------
    mcwhirter : LTECheckResult
        McWhirter criterion check
    temporal : LTECheckResult or None
        Temporal relaxation check (if performed)
    overall_satisfied : bool
        True only if all performed checks pass
    warnings : list of str
        Human-readable warning messages
    quality_metrics : dict
        Metrics suitable for inclusion in CFLIBSResult.quality_metrics
    """

    mcwhirter: LTECheckResult
    temporal: Optional[LTECheckResult] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def overall_satisfied(self) -> bool:
        checks = [self.mcwhirter]
        if self.temporal is not None:
            checks.append(self.temporal)
        return all(c.satisfied for c in checks)

    @property
    def quality_metrics(self) -> dict:
        metrics = {
            "lte_mcwhirter_satisfied": self.mcwhirter.satisfied,
            "lte_n_e_ratio": self.mcwhirter.ratio,
            "lte_n_e_required_cm3": self.mcwhirter.n_e_required,
        }
        if self.temporal is not None:
            metrics["lte_temporal_satisfied"] = self.temporal.satisfied
        return metrics


class LTEValidator:
    """
    Validates LTE assumptions for LIBS plasma diagnostics.

    The McWhirter criterion is a *necessary* (not sufficient) condition
    for LTE: the electron density must be high enough to thermalise the
    energy levels via collisional processes faster than they radiate.

    Usage
    -----
    >>> validator = LTEValidator()
    >>> report = validator.validate(T_K=10000, n_e_cm3=1e17, delta_E_eV=2.0)
    >>> if not report.overall_satisfied:
    ...     print(report.warnings)
    """

    @staticmethod
    def check_mcwhirter(
        T_K: float,
        n_e_cm3: float,
        delta_E_eV: float,
    ) -> LTECheckResult:
        """
        Apply the McWhirter criterion.

        n_e >= 1.6e12 * sqrt(T_K) * delta_E_eV^3

        Parameters
        ----------
        T_K : float
            Plasma temperature [K]
        n_e_cm3 : float
            Electron density [cm^-3]
        delta_E_eV : float
            Largest energy gap in the atomic term scheme [eV] —
            conventionally the resonance (ground -> first-excited)
            transition (Cristoforetti et al. 2010, Spectrochim. Acta B
            65, 86-95), NOT the gap between adjacent observed levels.

        Returns
        -------
        LTECheckResult
        """
        n_e_required = MCWHIRTER_CONST * np.sqrt(T_K) * delta_E_eV**3
        ratio = n_e_cm3 / n_e_required if n_e_required > 0 else float("inf")
        satisfied = n_e_cm3 >= n_e_required

        if satisfied:
            msg = (
                f"McWhirter: PASS  n_e = {n_e_cm3:.2e} >= {n_e_required:.2e} cm^-3 "
                f"(ratio = {ratio:.2f})"
            )
        else:
            msg = (
                f"McWhirter: FAIL  n_e = {n_e_cm3:.2e} < {n_e_required:.2e} cm^-3 "
                f"(ratio = {ratio:.2f}) — LTE may not hold"
            )

        return LTECheckResult(
            criterion="mcwhirter",
            satisfied=satisfied,
            n_e_required=n_e_required,
            n_e_actual=n_e_cm3,
            ratio=ratio,
            message=msg,
        )

    @staticmethod
    def check_temporal(
        T_K: float,
        n_e_cm3: float,
        plasma_lifetime_ns: float = 1000.0,
    ) -> LTECheckResult:
        """
        Simplified temporal LTE check.

        The electron-ion energy equilibration time must be much shorter
        than the plasma lifetime. Uses the Spitzer formula:

        tau_ei ~ 12 * pi^(3/2) * eps0^2 * sqrt(me) * (kT_e)^(3/2)
                 / (n_e * e^4 * sqrt(2) * ln_Lambda)

        Approximated as: tau_ei_ns ~ 2.8e-5 * T_eV^(3/2) / n_e_cm3

        The criterion is tau_ei << plasma_lifetime (default 1 µs = 1000 ns).

        Parameters
        ----------
        T_K : float
            Plasma temperature [K]
        n_e_cm3 : float
            Electron density [cm^-3]
        plasma_lifetime_ns : float
            Characteristic plasma lifetime in nanoseconds (default: 1000 ns)

        Returns
        -------
        LTECheckResult
        """
        T_eV = T_K * KB_EV
        # Electron thermal relaxation time (NRL Plasma Formulary):
        #   tau_ee [s] = T_eV^1.5 / (2.91e-5 * n_e * ln_Lambda)
        # with ln_Lambda ~ 10 for typical LIBS conditions:
        #   tau_ee [ns] = 3.44e13 * T_eV^1.5 / n_e_cm3
        SPITZER_NS = 3.44e13  # ns * cm^-3 * eV^(-3/2)
        tau_ei_ns = SPITZER_NS * (T_eV**1.5) / max(n_e_cm3, 1.0)

        # Criterion: tau_ee < 0.1 * plasma_lifetime (factor-of-10 margin)
        n_e_required = SPITZER_NS * (T_eV**1.5) / (0.1 * plasma_lifetime_ns)
        ratio = n_e_cm3 / n_e_required if n_e_required > 0 else float("inf")
        satisfied = tau_ei_ns < 0.1 * plasma_lifetime_ns

        if satisfied:
            msg = (
                f"Temporal: PASS  tau_ei = {tau_ei_ns:.2e} ns << {plasma_lifetime_ns:.0f} ns "
                f"(ratio = {ratio:.2f})"
            )
        else:
            msg = (
                f"Temporal: FAIL  tau_ei = {tau_ei_ns:.2e} ns not << {plasma_lifetime_ns:.0f} ns "
                f"(ratio = {ratio:.2f}) — equilibration may be too slow"
            )

        return LTECheckResult(
            criterion="temporal",
            satisfied=satisfied,
            n_e_required=n_e_required,
            n_e_actual=n_e_cm3,
            ratio=ratio,
            message=msg,
        )

    @staticmethod
    def _delta_e_from_observations(observations: list) -> float:
        """
        Derive the McWhirter delta_E_eV from observed lines.

        The McWhirter criterion requires delta_E to be the *largest
        energy gap in the atomic term scheme* — conventionally the
        resonance (ground -> first-excited) transition, which is
        ~3-5 eV for typical LIBS species. Cristoforetti et al. (2010)
        Spectrochim. Acta B 65, 86-95 codifies this: McWhirter's
        n_e >= 1.6e12 * sqrt(T) * delta_E^3 floor scales with the
        cube of delta_E, so using the small gap between *adjacent*
        observed upper-level energies (which cluster narrowly) badly
        under-estimates the required density and lets non-LTE plasmas
        pass the gate.

        The observed lines do not carry lower-level energies, so the
        largest term-scheme gap reachable from data in scope is the
        full span from the ground state (E_lower >= 0) up to the
        highest observed upper level: max(E_k) - 0. This bounds the
        resonance-to-upper-level transition and is far larger than the
        adjacent-gap value.
        """
        energies = [o.E_k_ev for o in observations]
        delta_E_eV = float(max(energies))
        if delta_E_eV <= 0.0:
            # All upper levels at the ground state: term-scheme gap is
            # undefined; use conservative default.
            delta_E_eV = 2.0
            logger.warning(
                "All observation upper levels at ground state; "
                "using default 2.0 eV for McWhirter criterion"
            )
        return max(delta_E_eV, 0.1)  # floor to avoid degenerate case

    @staticmethod
    def _resolve_delta_e_ev(
        delta_E_eV: Optional[float],
        observations: Optional[list],
    ) -> float:
        """Resolve delta_E_eV from an explicit value or observations."""
        if delta_E_eV is not None:
            return delta_E_eV
        if observations is not None and len(observations) > 0:
            return LTEValidator._delta_e_from_observations(observations)
        logger.warning(
            "delta_E_eV not specified and no observations provided; "
            "using default 2.0 eV for McWhirter criterion"
        )
        return 2.0  # conservative default

    @staticmethod
    def _collect_warnings(
        mcwhirter: LTECheckResult,
        temporal: Optional[LTECheckResult],
    ) -> List[str]:
        """Gather and log warning messages from failed checks."""
        warnings: List[str] = []
        if not mcwhirter.satisfied:
            warnings.append(mcwhirter.message)
        if temporal is not None and not temporal.satisfied:
            warnings.append(temporal.message)

        if warnings:
            for w in warnings:
                logger.warning(w)
        return warnings

    def validate(
        self,
        T_K: float,
        n_e_cm3: float,
        delta_E_eV: Optional[float] = None,
        observations: Optional[list] = None,
        check_temporal: bool = False,
        plasma_lifetime_ns: float = 1000.0,
    ) -> LTEReport:
        """
        Run all LTE validity checks.

        Parameters
        ----------
        T_K : float
            Plasma temperature [K]
        n_e_cm3 : float
            Electron density [cm^-3]
        delta_E_eV : float, optional
            Largest term-scheme energy gap [eV]. If None, derived from
            observations as the full span from the ground state to the
            highest observed upper level (max E_k), which bounds the
            resonance transition per Cristoforetti et al. (2010).
        observations : list of LineObservation, optional
            Used to derive delta_E_eV if not provided directly.
        check_temporal : bool
            Whether to also run the temporal equilibration check.
        plasma_lifetime_ns : float
            Plasma lifetime for temporal check [ns].

        Returns
        -------
        LTEReport
        """
        delta_E_eV = self._resolve_delta_e_ev(delta_E_eV, observations)

        mcwhirter = self.check_mcwhirter(T_K, n_e_cm3, delta_E_eV)

        temporal = None
        if check_temporal:
            temporal = self.check_temporal(T_K, n_e_cm3, plasma_lifetime_ns)

        warnings = self._collect_warnings(mcwhirter, temporal)

        return LTEReport(mcwhirter=mcwhirter, temporal=temporal, warnings=warnings)
