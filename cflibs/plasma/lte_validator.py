"""
LTE validity checker for CF-LIBS plasma diagnostics.

Implements the McWhirter criterion (necessary condition for LTE), a
simplified electron-thermalisation (Spitzer) check, and — opt-in — the
Cristoforetti relaxation-time criterion (the time to reach excitation/
ionization equilibrium vs. the plasma evolution timescale). Results are
surfaced in CFLIBSResult quality_metrics so users are warned when LTE
assumptions may be invalid.

References
----------
- McWhirter, R.W.P. (1965) in "Plasma Diagnostic Techniques", ed. Huddlestone & Leonard
- Griem, H.R. (1964) "Plasma Spectroscopy", McGraw-Hill (relaxation times)
- Cristoforetti, G. et al. (2010) Spectrochim. Acta B 65, 86-95,
  "Local Thermodynamic Equilibrium in Laser-Induced Breakdown Spectroscopy:
  Beyond the McWhirter criterion" (notes McWhirter as necessary but not
  sufficient; Eq. 4 relaxation-time criterion)
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
        Electron-thermalisation (Spitzer) check (if performed)
    relaxation : LTECheckResult or None
        Cristoforetti relaxation-time criterion (if performed)
    overall_satisfied : bool
        True only if all performed checks pass
    warnings : list of str
        Human-readable warning messages
    quality_metrics : dict
        Metrics suitable for inclusion in CFLIBSResult.quality_metrics
    """

    mcwhirter: LTECheckResult
    temporal: Optional[LTECheckResult] = None
    relaxation: Optional[LTECheckResult] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def overall_satisfied(self) -> bool:
        checks = [self.mcwhirter]
        if self.temporal is not None:
            checks.append(self.temporal)
        if self.relaxation is not None:
            checks.append(self.relaxation)
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
        if self.relaxation is not None:
            metrics["lte_relaxation_satisfied"] = self.relaxation.satisfied
            # tau_rel [ns] is stored in n_e_actual; surface it as a diagnostic.
            metrics["lte_relaxation_tau_ns"] = self.relaxation.n_e_actual
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
    def check_relaxation(
        T_K: float,
        n_e_cm3: float,
        delta_E_eV: float,
        plasma_lifetime_ns: float = 1000.0,
        oscillator_strength: float = 0.5,
        gaunt_factor: float = 1.0,
        margin: float = 10.0,
    ) -> LTECheckResult:
        r"""
        Cristoforetti relaxation-time LTE criterion (opt-in; beyond McWhirter).

        The McWhirter density floor is a *necessary but not sufficient*
        condition: it guarantees only that collisional rates exceed
        radiative rates at steady state, and says nothing about whether a
        *transient* plasma has had enough time to reach the equilibrium
        populations. Cristoforetti et al. (2010), Spectrochim. Acta B 65,
        86-95 ("Local Thermodynamic Equilibrium in Laser-Induced Breakdown
        Spectroscopy: Beyond the McWhirter criterion") add a temporal
        criterion: the relaxation time to reach excitation/ionization
        equilibrium between two levels :math:`n,m` must be much shorter than
        the timescale over which the plasma thermodynamic parameters
        (:math:`T`, :math:`n_e`) evolve.

        The relaxation time (Cristoforetti et al. 2010, Eq. 4; after Griem
        1964) is

        .. math::

            \tau_{rel} \approx \frac{6.3\times10^{4}}
                {n_e\, \langle g\rangle\, f_{nm}}\;
                \Delta E_{nm}\, (k_B T)^{1/2}\,
                \exp\!\left(\frac{\Delta E_{nm}}{k_B T}\right)

        with :math:`n_e` in cm\ :sup:`-3`, :math:`\Delta E_{nm}` and
        :math:`k_B T` in eV, and :math:`\tau_{rel}` in **seconds**.
        :math:`f_{nm}` is the absorption oscillator strength of the
        transition and :math:`\langle g\rangle` the effective (EEDF-averaged)
        Gaunt factor (order unity). Verified against the worked example in
        the LIBS literature (a colliding-carbon plasma with
        :math:`\Delta E = 2.9\,\mathrm{eV}`, :math:`f_{12}=0.914`,
        :math:`n_e = 3\times10^{16}\,\mathrm{cm^{-3}}`,
        :math:`k_B T = 1.45\,\mathrm{eV}`, :math:`\langle g\rangle\approx1`)
        which yields :math:`\tau_{rel}\approx6\times10^{-11}\,\mathrm{s}`.

        The exponential :math:`\exp(\Delta E/k_B T)` makes :math:`\tau_{rel}`
        grow rapidly for large gaps / low temperatures, and the
        :math:`1/n_e` prefactor makes it *grow* as the plasma rarefies — so
        a low-:math:`n_e`, fast-evolving plasma can satisfy the (steady-state)
        McWhirter floor yet still fail this transient criterion, exactly the
        regime this check is meant to catch.

        Criterion (Cristoforetti et al. 2010): LTE requires
        :math:`\tau_{rel} \ll \tau_{evol}`, here imposed with a factor-of-
        ``margin`` safety margin as
        :math:`\tau_{rel} < \tau_{evol}/\text{margin}`, where
        :math:`\tau_{evol}` is the plasma lifetime (the characteristic time
        over which :math:`T` and :math:`n_e` vary). The conventional margin
        is one order of magnitude (``margin=10``).

        Parameters
        ----------
        T_K : float
            Plasma temperature [K].
        n_e_cm3 : float
            Electron density [cm^-3].
        delta_E_eV : float
            Energy gap :math:`\Delta E_{nm}` of the slowest-thermalising
            transition [eV] (largest resonance gap; same physical quantity
            used by the McWhirter check).
        plasma_lifetime_ns : float
            Plasma evolution timescale :math:`\tau_{evol}` [ns]
            (default 1000 ns = 1 µs, typical LIBS gate).
        oscillator_strength : float
            Absorption oscillator strength :math:`f_{nm}` (default 0.5,
            a representative strong-line value).
        gaunt_factor : float
            Effective Gaunt factor :math:`\langle g\rangle` (default 1.0,
            the order-unity value used in the canonical estimate).
        margin : float
            Safety margin: require :math:`\tau_{rel} < \tau_{evol}/`
            ``margin`` (default 10, i.e. one decade).

        Returns
        -------
        LTECheckResult
            ``n_e_required`` / ``n_e_actual`` / ``ratio`` are repurposed for
            the time-domain comparison: ``n_e_required`` holds the maximum
            tolerable :math:`\tau_{rel}` [ns], ``n_e_actual`` the computed
            :math:`\tau_{rel}` [ns], and ``ratio`` =
            :math:`(\tau_{evol}/\text{margin})/\tau_{rel}` (>= 1 passes).
        """
        T_eV = T_K * KB_EV
        f_nm = max(oscillator_strength, 1e-6)
        g_eff = max(gaunt_factor, 1e-6)

        # Cristoforetti et al. 2010 Eq. 4 (Griem 1964); n_e in cm^-3,
        # delta_E and kT in eV -> tau_rel in seconds. RELAX_CONST = 6.3e4.
        RELAX_CONST = 6.3e4
        tau_rel_s = (
            RELAX_CONST
            / (max(n_e_cm3, 1.0) * g_eff * f_nm)
            * delta_E_eV
            * np.sqrt(T_eV)
            * np.exp(delta_E_eV / T_eV)
        )
        tau_rel_ns = tau_rel_s * 1e9

        tau_max_ns = plasma_lifetime_ns / margin
        ratio = tau_max_ns / tau_rel_ns if tau_rel_ns > 0 else float("inf")
        satisfied = tau_rel_ns < tau_max_ns

        if satisfied:
            msg = (
                f"Relaxation: PASS  tau_rel = {tau_rel_ns:.2e} ns << "
                f"tau_evol/{margin:.0f} = {tau_max_ns:.2e} ns (margin = {ratio:.2f})"
            )
        else:
            msg = (
                f"Relaxation: FAIL  tau_rel = {tau_rel_ns:.2e} ns not << "
                f"tau_evol/{margin:.0f} = {tau_max_ns:.2e} ns (margin = {ratio:.2f}) "
                f"— plasma may evolve before reaching equilibrium (transient non-LTE)"
            )

        return LTECheckResult(
            criterion="relaxation",
            satisfied=satisfied,
            n_e_required=tau_max_ns,
            n_e_actual=tau_rel_ns,
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
        relaxation: Optional[LTECheckResult] = None,
    ) -> List[str]:
        """Gather and log warning messages from failed checks."""
        warnings: List[str] = []
        if not mcwhirter.satisfied:
            warnings.append(mcwhirter.message)
        if temporal is not None and not temporal.satisfied:
            warnings.append(temporal.message)
        if relaxation is not None and not relaxation.satisfied:
            warnings.append(relaxation.message)

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
        check_relaxation: bool = False,
        relaxation_oscillator_strength: float = 0.5,
        relaxation_gaunt_factor: float = 1.0,
        relaxation_margin: float = 10.0,
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
            Plasma evolution timescale (lifetime) for the temporal and
            relaxation checks [ns].
        check_relaxation : bool
            Opt-in. Whether to also run the Cristoforetti et al. (2010)
            relaxation-time criterion (``check_relaxation`` of
            :meth:`check_relaxation`). Defaults to ``False`` so the default
            behaviour is unchanged — this is a *diagnostic* flag and, like
            the temporal check, does not by itself cause lines/spectra to be
            rejected anywhere in the default pipeline.
        relaxation_oscillator_strength : float
            Oscillator strength :math:`f_{nm}` for the relaxation check
            (default 0.5).
        relaxation_gaunt_factor : float
            Effective Gaunt factor :math:`\\langle g\\rangle` for the
            relaxation check (default 1.0).
        relaxation_margin : float
            Safety margin for the relaxation check (default 10).

        Returns
        -------
        LTEReport
        """
        delta_E_eV = self._resolve_delta_e_ev(delta_E_eV, observations)

        mcwhirter = self.check_mcwhirter(T_K, n_e_cm3, delta_E_eV)

        temporal = None
        if check_temporal:
            temporal = self.check_temporal(T_K, n_e_cm3, plasma_lifetime_ns)

        relaxation = None
        if check_relaxation:
            relaxation = self.check_relaxation(
                T_K,
                n_e_cm3,
                delta_E_eV,
                plasma_lifetime_ns=plasma_lifetime_ns,
                oscillator_strength=relaxation_oscillator_strength,
                gaunt_factor=relaxation_gaunt_factor,
                margin=relaxation_margin,
            )

        warnings = self._collect_warnings(mcwhirter, temporal, relaxation)

        return LTEReport(
            mcwhirter=mcwhirter,
            temporal=temporal,
            relaxation=relaxation,
            warnings=warnings,
        )
