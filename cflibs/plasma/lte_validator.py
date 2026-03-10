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

from cflibs.core.logging_config import get_logger

logger = get_logger("plasma.lte_validator")

# McWhirter criterion constant: n_e >= C * sqrt(T_K) * delta_E_eV^3
# C = 1.6e12 cm^-3 K^{-1/2} eV^{-3}
MCWHIRTER_C = 1.6e12  # cm^-3 K^{-1/2} eV^{-3}

# kB in eV/K
KB_EV = 8.617333262e-5


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
            Largest energy gap between adjacent levels of interest [eV]

        Returns
        -------
        LTECheckResult
        """
        n_e_required = MCWHIRTER_C * np.sqrt(T_K) * delta_E_eV**3
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
            Largest energy gap [eV]. If None, extracted from observations.
        observations : list of LineObservation, optional
            Used to compute delta_E_eV if not provided directly.
        check_temporal : bool
            Whether to also run the temporal equilibration check.
        plasma_lifetime_ns : float
            Plasma lifetime for temporal check [ns].

        Returns
        -------
        LTEReport
        """
        # Determine delta_E_eV
        if delta_E_eV is None:
            if observations is not None and len(observations) > 0:
                # McWhirter criterion uses the largest gap between *adjacent*
                # energy levels, not the total span.  Sort unique E_k values
                # and take the maximum consecutive difference.
                energies = sorted({o.E_k_ev for o in observations})
                if len(energies) >= 2:
                    gaps = [energies[i + 1] - energies[i] for i in range(len(energies) - 1)]
                    delta_E_eV = float(max(gaps))
                else:
                    delta_E_eV = float(energies[0]) if energies else 2.0
                delta_E_eV = max(delta_E_eV, 0.1)  # floor to avoid degenerate case
            else:
                delta_E_eV = 2.0  # conservative default
                logger.warning(
                    "delta_E_eV not specified and no observations provided; "
                    "using default 2.0 eV for McWhirter criterion"
                )

        mcwhirter = self.check_mcwhirter(T_K, n_e_cm3, delta_E_eV)

        temporal = None
        if check_temporal:
            temporal = self.check_temporal(T_K, n_e_cm3, plasma_lifetime_ns)

        warnings = []
        if not mcwhirter.satisfied:
            warnings.append(mcwhirter.message)
        if temporal is not None and not temporal.satisfied:
            warnings.append(temporal.message)

        if warnings:
            for w in warnings:
                logger.warning(w)

        return LTEReport(mcwhirter=mcwhirter, temporal=temporal, warnings=warnings)
