"""
Automatic line selection for CF-LIBS analysis.

Provides quality scoring and filtering of spectral lines to improve
Boltzmann plot reliability.

Scoring Algorithm
-----------------
Each spectral line receives a composite quality score:

    score = SNR × (1/σ_atomic) × isolation

Where:
    - **SNR**: Signal-to-noise ratio = intensity / uncertainty
    - **σ_atomic**: Atomic data uncertainty (transition probability accuracy)
    - **isolation**: Spectral isolation factor (1.0 = isolated, 0.0 = blended)

Isolation Factor
----------------
Computed as:

    isolation = 1 - exp(-Δλ / λ_iso)

Where:
    - Δλ: Wavelength separation to nearest line (nm)
    - λ_iso: Characteristic isolation scale (default 0.1 nm)

Default Thresholds
------------------
- min_snr: 10.0 (minimum signal-to-noise ratio)
- min_energy_spread_ev: 2.0 eV (for reliable temperature determination)
- min_lines_per_element: 3 (minimum lines needed per element)
- isolation_wavelength_nm: 0.1 nm (isolation characteristic scale)
- max_lines_per_element: 20 (to avoid overweighting single element)

Atomic Data Uncertainty Grades (NIST)
-------------------------------------
- AAA: ≤0.3%    - A: ≤3%      - C: ≤25%
- AA:  ≤1%     - B+: ≤7%     - D+: ≤40%
- A+:  ≤2%     - B:  ≤10%    - D:  ≤50%
               - C+: ≤18%    - E:  >50%

Selection Criteria
------------------
Lines are rejected if any of:
1. SNR < min_snr threshold
2. Line is a resonance transition (high self-absorption risk)
3. Isolation factor < 0.5 (severely blended)
4. Exceeds max_lines_per_element after sorting by score

Warnings are issued for:
- Energy spread below minimum for reliable T determination
- Fewer than min_lines_per_element available

Boltzmann Fitness
-----------------
Added functions to assess line suitability for Boltzmann analysis:
- check_energy_spread(): Warns if E_k spread < 1 eV (unreliable fit)
- filter_self_absorbed(): Exclude lines with high Aki and low E_k
- compute_boltzmann_fitness(): Composite score considering all factors

Partial-LTE Thermalization Cut (opt-in)
---------------------------------------
In a laser-induced plasma the free-electron temperature thermalizes only those
bound levels that are strongly coupled to the electron continuum by inelastic
collisions. Low-lying levels (large energy gaps to neighbouring levels) are
populated/depopulated predominantly by radiative processes and therefore deviate
from a Boltzmann distribution at the electron temperature — a state of *partial*
LTE (pLTE). Cristoforetti et al. (2010) show that only excited levels lying close
to the ionization limit attain Saha–Boltzmann equilibrium, so lines whose lower
level sits below a thermalization limit E* bias the Boltzmann/Saha plot.

The thermalization limit is obtained by inverting the McWhirter criterion, which
gives the minimum electron density for collisional dominance over a radiative gap
ΔE [eV] at temperature T [K]:

    n_e [cm^-3] >= 1.6e12 * sqrt(T) * (ΔE)^3            (McWhirter)

Solving for the largest gap that the measured plasma can thermalize yields the
thermalization-limit energy

    E* = ( n_e / (1.6e12 * sqrt(T)) )^(1/3)   [eV]

A transition is retained only if its **lower-level** energy E_i >= E*; lines whose
lower level lies below E* are not collisionally thermalized and are excluded.
This cut is **strictly opt-in** (see ``apply_plte_thermalization_cut``); it does
not change the default ``select()`` behaviour.

Literature References
---------------------
- Clegg et al. (2017): Line selection strategies for LIBS quantification
- Tognoni et al. (2006): Quantitative LIBS analysis review
- NIST ASD: Atomic transition probability accuracy grades
- Cristoforetti et al. (2010), Spectrochim. Acta B 65, 86-95: "Local
  Thermodynamic Equilibrium in Laser-Induced Breakdown Spectroscopy: Beyond
  the McWhirter criterion" — partial-LTE thermalization limit.
- McWhirter, R.W.P. (1965), in *Plasma Diagnostic Techniques* — collisional
  LTE electron-density criterion.
"""

from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict, Optional, Sequence, Tuple, Set
import numpy as np

from cflibs.core.constants import KB_EV, MCWHIRTER_CONST
from cflibs.core.logging_config import get_logger
from cflibs.inversion.physics.boltzmann import LineObservation

logger = get_logger("inversion.line_selection")

# McWhirter criterion prefactor in the cgs/eV/K convention
#   n_e [cm^-3] >= MCWHIRTER_CONST * sqrt(T[K]) * (ΔE[eV])^3
# (McWhirter 1965; see e.g. Cristoforetti et al. 2010, Spectrochim. Acta B 65, 86)
MCWHIRTER_PREFACTOR_CM3_K = MCWHIRTER_CONST


def mcwhirter_thermalization_limit_ev(
    temperature_K: float,
    electron_density_cm3: float,
) -> float:
    r"""
    Partial-LTE thermalization-limit energy E\* from the McWhirter criterion.

    A bound level is collisionally thermalized to the free-electron temperature
    only when the McWhirter criterion is satisfied for the relevant energy gap
    :math:`\Delta E`:

    .. math::

        n_e\,[\mathrm{cm^{-3}}] \ge 1.6\times10^{12}\,\sqrt{T[\mathrm{K}]}\,
        (\Delta E[\mathrm{eV}])^{3}.

    Inverting for the largest gap the measured plasma can thermalize gives the
    thermalization-limit energy

    .. math::

        E^{*} = \left(\frac{n_e}{1.6\times10^{12}\,\sqrt{T}}\right)^{1/3}
        \quad[\mathrm{eV}].

    Levels whose energy lies below :math:`E^{*}` are populated/depopulated
    predominantly by radiation rather than collisions, so they deviate from the
    Saha–Boltzmann distribution (partial LTE).

    Parameters
    ----------
    temperature_K : float
        Plasma (excitation) temperature in kelvin. Must be > 0.
    electron_density_cm3 : float
        Free-electron number density in cm^-3. Must be > 0.

    Returns
    -------
    float
        Thermalization-limit energy E* in eV.

    References
    ----------
    Cristoforetti et al. (2010), Spectrochim. Acta Part B 65, 86-95.
    McWhirter, R.W.P. (1965), *Plasma Diagnostic Techniques*, Ch. 5.
    """
    if temperature_K <= 0.0:
        raise ValueError(f"temperature_K must be > 0, got {temperature_K}")
    if electron_density_cm3 <= 0.0:
        raise ValueError(f"electron_density_cm3 must be > 0, got {electron_density_cm3}")

    denominator = MCWHIRTER_PREFACTOR_CM3_K * np.sqrt(temperature_K)
    return float((electron_density_cm3 / denominator) ** (1.0 / 3.0))


@dataclass
class LineScore:
    """Score and metadata for a spectral line."""

    observation: LineObservation
    score: float
    snr: float
    isolation_factor: float
    atomic_uncertainty: float
    is_resonance: bool
    rejection_reason: Optional[str] = None


@dataclass
class LineSelectionResult:
    """Result of automatic line selection."""

    selected_lines: List[LineObservation]
    rejected_lines: List[LineObservation]
    scores: List[LineScore]
    energy_spread_ev: float
    n_elements: int
    warnings: List[str] = field(default_factory=list)


@dataclass
class PLTECutResult:
    """
    Result of the opt-in partial-LTE thermalization cut.

    Attributes
    ----------
    thermalized_lines : List[LineObservation]
        Lines whose lower-level energy lies at or above the thermalization
        limit E* (kept).
    sub_threshold_lines : List[LineObservation]
        Lines whose lower-level energy lies below E* (removed — not
        collisionally thermalized to the electron temperature).
    e_star_ev : float
        Thermalization-limit energy E* in eV used for the cut.
    skipped_lines : List[LineObservation]
        Lines for which no lower-level energy was supplied; conservatively
        kept (never removed) so missing data cannot silently drop lines.
    """

    thermalized_lines: List[LineObservation]
    sub_threshold_lines: List[LineObservation]
    e_star_ev: float
    skipped_lines: List[LineObservation] = field(default_factory=list)


class LineSelector:
    """
    Automatic line selection for CF-LIBS analysis.

    Scores lines based on:
    - Signal-to-noise ratio (SNR)
    - Atomic data uncertainty
    - Spectral isolation (no blends)
    - Energy spread requirement

    Filters out:
    - Low SNR lines
    - Blended lines
    - Resonance lines, only when ``exclude_resonance=True`` (default False:
      resonance lines are the brightest, most persistent LIBS lines and the
      only detectable lines for some majors, e.g. Al I 394.4/396.2 nm)
    """

    # Default atomic data uncertainties (relative, from NIST accuracy grades)
    ATOMIC_UNCERTAINTIES = {
        "AAA": 0.003,  # ≤0.3%
        "AA": 0.01,  # ≤1%
        "A+": 0.02,  # ≤2%
        "A": 0.03,  # ≤3%
        "B+": 0.07,  # ≤7%
        "B": 0.10,  # ≤10%
        "C+": 0.18,  # ≤18%
        "C": 0.25,  # ≤25%
        "D+": 0.40,  # ≤40%
        "D": 0.50,  # ≤50%
        "E": 1.0,  # >50%
    }

    def __init__(
        self,
        min_snr: float = 10.0,
        min_energy_spread_ev: float = 2.0,
        min_lines_per_element: int = 3,
        exclude_resonance: bool = False,
        isolation_wavelength_nm: float = 0.1,
        max_lines_per_element: int = 20,
        target_sigma_t: Optional[float] = None,
        plasma_temperature_K: float = 10000.0,
        reliability_ranked_selection: bool = False,
    ):
        """
        Initialize line selector.

        Parameters
        ----------
        min_snr : float
            Minimum signal-to-noise ratio
        min_energy_spread_ev : float
            Minimum energy spread for reliable temperature determination
        min_lines_per_element : int
            Minimum number of lines required per element
        exclude_resonance : bool
            Whether to exclude resonance lines (ground state transitions).
            Default False — matches the validated CLI default: resonance
            lines are kept because they are the only detectable lines for
            some major elements, and the solver corrects them when
            self-absorption correction is enabled.
        isolation_wavelength_nm : float
            Minimum wavelength separation for isolation check
        max_lines_per_element : int
            Maximum lines to select per element (to avoid overweighting)
        target_sigma_t : float, optional
            Target RELATIVE temperature accuracy (σ_T/T). When set (gated,
            default None=off), the SNR/spread/min-lines gates are DERIVED from
            this target + the spectrum's measured energy spread via the verified
            ``ErrorBudget`` (see ``derived_thresholds``), replacing the tuned
            magic numbers. ``target_sigma_t≈0.10`` reproduces the legacy min_snr=10.
        plasma_temperature_K : float
            Representative plasma T for the σ_T → slope-target conversion
            (τβ = σ_T/T / (kB·T)); only used when ``target_sigma_t`` is set.
        """
        self.min_snr = min_snr
        self.min_energy_spread_ev = min_energy_spread_ev
        self.min_lines_per_element = min_lines_per_element
        self.exclude_resonance = exclude_resonance
        self.isolation_wavelength_nm = isolation_wavelength_nm
        self.max_lines_per_element = max_lines_per_element
        self.target_sigma_t = target_sigma_t
        self.plasma_temperature_K = plasma_temperature_K
        self.reliability_ranked_selection = reliability_ranked_selection

    def select(
        self,
        observations: List[LineObservation],
        resonance_lines: Optional[Set[Tuple[str, int, float]]] = None,
        atomic_uncertainties: Optional[Dict[Tuple[str, int, float], float]] = None,
    ) -> LineSelectionResult:
        """
        Select optimal lines for CF-LIBS analysis.

        Parameters
        ----------
        observations : List[LineObservation]
            All candidate line observations
        resonance_lines : Set[Tuple[str, int, float]], optional
            Set of (element, ion_stage, wavelength) for resonance lines
        atomic_uncertainties : Dict[Tuple[str, int, float], float], optional
            Atomic data uncertainties by (element, ion_stage, wavelength)

        Returns
        -------
        LineSelectionResult
        """
        if resonance_lines is None:
            resonance_lines = set()

        if atomic_uncertainties is None:
            atomic_uncertainties = {}

        warnings: list[str] = []

        # Derived thresholds (gated): when a target σ_T/T is set, replace the tuned
        # min_snr/min_energy_spread/min_lines gates with values DERIVED from the target +
        # the spectrum's measured energy spread, via the verified ErrorBudget. No-op when off.
        if self.target_sigma_t is not None:
            (
                self.min_snr,
                self.min_energy_spread_ev,
                self.min_lines_per_element,
            ) = self._derive_thresholds(observations)

        # Score all lines
        scores = [
            self._score_line(obs, observations, resonance_lines, atomic_uncertainties)
            for obs in observations
        ]

        # Filter by criteria
        valid_scores, rejected_scores = self._partition_by_criteria(scores)

        # Group by element and ensure energy spread
        by_element = defaultdict(list)
        for score in valid_scores:
            by_element[score.observation.element].append(score)

        selected_scores = self._select_per_element(by_element, rejected_scores, warnings)

        # Check minimum lines per element
        self._warn_insufficient_lines(by_element, warnings)

        # Calculate overall energy spread
        all_energies = [s.observation.E_k_ev for s in selected_scores]
        energy_spread = (max(all_energies) - min(all_energies)) if all_energies else 0.0

        selected_lines = [s.observation for s in selected_scores]
        rejected_lines = [s.observation for s in rejected_scores]

        return LineSelectionResult(
            selected_lines=selected_lines,
            rejected_lines=rejected_lines,
            scores=scores,
            energy_spread_ev=energy_spread,
            n_elements=len(by_element),
            warnings=warnings,
        )

    def _derive_thresholds(self, observations: List[LineObservation]) -> Tuple[float, float, int]:
        """Derive (min_snr, min_energy_spread_ev, min_lines) from the target σ_T/T and the
        spectrum's measured per-element energy spread, via the verified ErrorBudget formulas
        (``derived_thresholds``). The hard gate is min_snr: a line needs per-line ordinate
        error ε ≤ ``max_per_line_error`` (= τβ·√(ssE/n)), i.e. SNR ≥ 1/ε_max. Falls back to
        the configured values if no element has ≥2 lines with positive energy spread."""
        from cflibs.core.constants import KB_EV
        from cflibs.inversion.physics import derived_thresholds as dt

        tau_beta = dt.slope_target_from_temp_rel(
            self.target_sigma_t, KB_EV, self.plasma_temperature_K
        )
        by_el: Dict[str, List[LineObservation]] = defaultdict(list)
        for o in observations:
            by_el[o.element].append(o)
        ns: list[float] = []
        ss_es: list[float] = []
        eps: list[float] = []
        for obs in by_el.values():
            energies = [o.E_k_ev for o in obs]
            n = len(energies)
            if n < 2:
                continue
            e_bar = sum(energies) / n
            ss_e = sum((e - e_bar) ** 2 for e in energies)
            if ss_e <= 0.0:
                continue
            ns.append(float(n))
            ss_es.append(ss_e)
            eps.extend(o.y_uncertainty for o in obs if o.y_uncertainty > 0)
        if not ns:
            return self.min_snr, self.min_energy_spread_ev, self.min_lines_per_element
        n_med = float(np.median(ns))
        ss_e_med = float(np.median(ss_es))
        eps_med = float(np.median(eps)) if eps else 0.02
        v_per_line = ss_e_med / n_med
        eps_max = dt.max_per_line_error(tau_beta, n_med, ss_e_med)
        derived_snr = (1.0 / eps_max) if eps_max > 0 else self.min_snr
        derived_lines = max(2, int(np.ceil(dt.required_min_lines(tau_beta, eps_med, v_per_line))))
        derived_spread = float(np.sqrt(dt.required_energy_spread(tau_beta, eps_med, n_med) / n_med))
        return derived_snr, derived_spread, derived_lines

    def _partition_by_criteria(
        self,
        scores: List[LineScore],
    ) -> Tuple[List[LineScore], List[LineScore]]:
        """Split scored lines into valid/rejected, tagging rejection reasons."""
        valid_scores: List[LineScore] = []
        rejected_scores: List[LineScore] = []

        for score in scores:
            if score.rejection_reason is not None:
                rejected_scores.append(score)
            elif score.snr < self.min_snr:
                score.rejection_reason = f"Low SNR ({score.snr:.1f} < {self.min_snr})"
                rejected_scores.append(score)
            elif score.is_resonance and self.exclude_resonance:
                score.rejection_reason = "Resonance line (self-absorption risk)"
                rejected_scores.append(score)
            elif score.isolation_factor < 0.5:
                score.rejection_reason = f"Blended (isolation={score.isolation_factor:.2f})"
                rejected_scores.append(score)
            else:
                valid_scores.append(score)

        return valid_scores, rejected_scores

    def _select_per_element(
        self,
        by_element: Dict[str, List[LineScore]],
        rejected_scores: List[LineScore],
        warnings: List[str],
    ) -> List[LineScore]:
        """Per element: keep the best lines up to the cap, reject the excess. With
        ``reliability_ranked_selection`` the kept subset MAXIMIZES upper-level energy spread
        (best temperature conditioning, ``twoLineBeta_stable_sharp``) when the cap binds;
        otherwise the highest-scored lines. Only differs from the baseline when the cap binds."""
        selected_scores: List[LineScore] = []
        cap = self.max_lines_per_element
        for element, elem_scores in by_element.items():
            # Sort by score descending
            elem_scores.sort(key=lambda s: s.score, reverse=True)

            # Check energy spread
            self._warn_energy_spread(element, elem_scores, warnings)

            if self.reliability_ranked_selection and len(elem_scores) > cap:
                keep = self._max_spread_subset(elem_scores, cap)
            else:
                keep = elem_scores[:cap]

            keep_ids = {id(s) for s in keep}
            selected_scores.extend(keep)
            for score in elem_scores:
                if id(score) not in keep_ids:
                    score.rejection_reason = "Exceeded max lines per element"
                    rejected_scores.append(score)

        return selected_scores

    @staticmethod
    def _max_spread_subset(scores: List[LineScore], k: int) -> List[LineScore]:
        """Pick ``k`` lines maximizing upper-level energy spread via farthest-point sampling
        on ``E_k``: seed with the two extremes (the best-conditioned pair,
        ``twoLineBeta_stable_sharp``), then greedily add the line farthest from the chosen
        set. Better temperature conditioning than a score-only top-k when the cap binds."""
        if k >= len(scores):
            return list(scores)
        energies = [s.observation.E_k_ev for s in scores]
        lo = min(range(len(scores)), key=lambda i: energies[i])
        hi = max(range(len(scores)), key=lambda i: energies[i])
        chosen = [lo] if lo == hi else [lo, hi]
        chosen_set = set(chosen)
        while len(chosen) < k:
            best_i, best_d = None, -1.0
            for i in range(len(scores)):
                if i in chosen_set:
                    continue
                d = min(abs(energies[i] - energies[c]) for c in chosen)
                if d > best_d:
                    best_d, best_i = d, i
            if best_i is None:
                break
            chosen.append(best_i)
            chosen_set.add(best_i)
        return [scores[i] for i in chosen]

    def _warn_energy_spread(
        self,
        element: str,
        elem_scores: List[LineScore],
        warnings: List[str],
    ) -> None:
        """Append a warning when an element's energy spread is below threshold."""
        if len(elem_scores) >= 2:
            energies = [s.observation.E_k_ev for s in elem_scores]
            spread = max(energies) - min(energies)

            if spread < self.min_energy_spread_ev:
                warnings.append(
                    f"{element}: Energy spread {spread:.2f} eV < {self.min_energy_spread_ev} eV"
                )

    def _warn_insufficient_lines(
        self,
        by_element: Dict[str, List[LineScore]],
        warnings: List[str],
    ) -> None:
        """Append a warning for any element with fewer than the minimum valid lines."""
        for element, elem_scores in by_element.items():
            n_valid = sum(1 for s in elem_scores if s.rejection_reason is None)
            if n_valid < self.min_lines_per_element:
                warnings.append(
                    f"{element}: Only {n_valid} lines available "
                    f"(need {self.min_lines_per_element})"
                )

    def _score_line(
        self,
        obs: LineObservation,
        all_observations: List[LineObservation],
        resonance_lines: Set[Tuple[str, int, float]],
        atomic_uncertainties: Dict[Tuple[str, int, float], float],
    ) -> LineScore:
        """Score a single line."""
        # SNR
        if obs.intensity_uncertainty > 0:
            snr = obs.intensity / obs.intensity_uncertainty
        else:
            snr = 100.0  # Assume good if no uncertainty provided

        # Atomic uncertainty
        key = (obs.element, obs.ionization_stage, obs.wavelength_nm)
        atomic_unc = atomic_uncertainties.get(key, 0.10)  # Default 10%

        # Isolation factor (1.0 = fully isolated, 0.0 = heavily blended)
        isolation = self._compute_isolation(obs, all_observations)

        # Check if resonance
        is_resonance = key in resonance_lines

        # Composite score: SNR × (1/uncertainty) × isolation
        # Higher is better
        if atomic_unc > 0:
            score = snr * (1.0 / atomic_unc) * isolation
        else:
            score = snr * isolation

        return LineScore(
            observation=obs,
            score=score,
            snr=snr,
            isolation_factor=isolation,
            atomic_uncertainty=atomic_unc,
            is_resonance=is_resonance,
        )

    def _compute_isolation(
        self,
        obs: LineObservation,
        all_observations: List[LineObservation],
    ) -> float:
        """
        Compute spectral isolation factor.

        Returns 1.0 if no nearby lines, decreasing toward 0.0 for blends.
        """
        min_separation = float("inf")

        for other in all_observations:
            if other is obs:
                continue

            separation = abs(obs.wavelength_nm - other.wavelength_nm)
            if separation < min_separation:
                min_separation = separation

        if min_separation == float("inf"):
            return 1.0

        # Sigmoid-like function: 1.0 for well-separated, 0.0 for blended
        # Characteristic scale is isolation_wavelength_nm
        isolation = 1.0 - np.exp(-min_separation / self.isolation_wavelength_nm)

        return isolation

    def apply_plte_thermalization_cut(
        self,
        observations: List[LineObservation],
        lower_level_energies_ev: Dict[Tuple[str, int, float], float],
        *,
        enable: bool = False,
        temperature_K: Optional[float] = None,
        electron_density_cm3: Optional[float] = None,
        e_star_ev: Optional[float] = None,
    ) -> PLTECutResult:
        r"""
        Opt-in partial-LTE cut: drop lines whose lower level is below E\*.

        In partial LTE only levels lying above the thermalization limit E* are
        collisionally coupled to the free-electron temperature and follow the
        Saha–Boltzmann distribution; lines whose **lower** level lies below E*
        bias the Boltzmann/Saha plot and are removed. ``LineObservation`` carries
        only the upper-level energy, so the lower-level energies are supplied
        explicitly via ``lower_level_energies_ev`` (keyed by
        ``(element, ionization_stage, wavelength_nm)``), matching the existing
        ``select()`` convention for ``resonance_lines``/``atomic_uncertainties``.

        This method is **strictly opt-in**: it is a no-op (returns every line as
        thermalized) unless ``enable=True``. It never mutates ``select()`` nor
        any default path.

        Parameters
        ----------
        observations : List[LineObservation]
            Candidate lines.
        lower_level_energies_ev : Dict[Tuple[str, int, float], float]
            Lower-level energy E_i [eV] keyed by
            ``(element, ionization_stage, wavelength_nm)``. Lines absent from
            this mapping are conservatively kept (reported in ``skipped_lines``).
        enable : bool, keyword-only, default False
            Master switch. When ``False`` the cut is a no-op and **all** lines
            are returned as thermalized (default behaviour unchanged).
        temperature_K : float, optional
            Plasma temperature [K], used with ``electron_density_cm3`` to derive
            E* from the McWhirter criterion. Ignored if ``e_star_ev`` is given.
        electron_density_cm3 : float, optional
            Electron density [cm^-3]; see ``temperature_K``.
        e_star_ev : float, optional
            Thermalization limit E* [eV] supplied directly. Overrides the
            ``(temperature_K, electron_density_cm3)`` derivation when provided
            (e.g. for tests with a known E*).

        Returns
        -------
        PLTECutResult
            Thermalized (kept) lines, sub-threshold (removed) lines, the E*
            used, and any lines skipped for lack of lower-level data.

        Raises
        ------
        ValueError
            If ``enable=True`` but neither ``e_star_ev`` nor both
            ``temperature_K`` and ``electron_density_cm3`` are provided.

        References
        ----------
        Cristoforetti et al. (2010), Spectrochim. Acta Part B 65, 86-95.

        Notes
        -----
        The lower-level criterion is additive to the existing upper-level energy
        *span* requirement (``min_energy_spread_ev``): the span controls slope
        precision, while this cut removes levels that are physically not in pLTE.
        """
        if not enable:
            # No-op: cut disabled, every line is treated as thermalized.
            return PLTECutResult(
                thermalized_lines=list(observations),
                sub_threshold_lines=[],
                e_star_ev=0.0,
            )

        if e_star_ev is None:
            if temperature_K is None or electron_density_cm3 is None:
                raise ValueError(
                    "pLTE cut requires either e_star_ev, or both temperature_K "
                    "and electron_density_cm3."
                )
            e_star_ev = mcwhirter_thermalization_limit_ev(temperature_K, electron_density_cm3)

        thermalized: List[LineObservation] = []
        sub_threshold: List[LineObservation] = []
        skipped: List[LineObservation] = []

        for obs in observations:
            key = (obs.element, obs.ionization_stage, obs.wavelength_nm)
            e_lower = lower_level_energies_ev.get(key)
            if e_lower is None:
                # No lower-level data: keep conservatively (cannot judge pLTE).
                skipped.append(obs)
                thermalized.append(obs)
            elif e_lower < e_star_ev:
                sub_threshold.append(obs)
            else:
                thermalized.append(obs)

        if sub_threshold:
            logger.info(
                "pLTE cut (E*=%.3f eV): removed %d sub-threshold line(s), "
                "kept %d thermalized, skipped %d (no lower-level data).",
                e_star_ev,
                len(sub_threshold),
                len(thermalized),
                len(skipped),
            )

        return PLTECutResult(
            thermalized_lines=thermalized,
            sub_threshold_lines=sub_threshold,
            e_star_ev=e_star_ev,
            skipped_lines=skipped,
        )

    def recommend_lines(
        self,
        observations: List[LineObservation],
        n_per_element: int = 5,
    ) -> Dict[str, List[LineObservation]]:
        """
        Recommend the best lines for each element.

        Parameters
        ----------
        observations : List[LineObservation]
            All candidate lines
        n_per_element : int
            Number of lines to recommend per element

        Returns
        -------
        Dict[str, List[LineObservation]]
            Best lines by element
        """
        result = self.select(observations)

        by_element = defaultdict(list)
        for score in result.scores:
            if score.rejection_reason is None:
                by_element[score.observation.element].append(score)

        recommendations = {}
        for element, scores in by_element.items():
            scores.sort(key=lambda s: s.score, reverse=True)
            recommendations[element] = [s.observation for s in scores[:n_per_element]]

        return recommendations


def identify_resonance_lines(
    observations: List[LineObservation],
    ground_state_threshold_ev: float = 0.5,
) -> Set[Tuple[str, int, float]]:
    """
    Identify resonance lines (transitions from/to ground state).

    Parameters
    ----------
    observations : List[LineObservation]
        Line observations with lower level energy if available
    ground_state_threshold_ev : float
        Energy threshold to consider as ground state

    Returns
    -------
    Set[Tuple[str, int, float]]
        Set of (element, ion_stage, wavelength) for resonance lines
    """
    # Unused parameters kept for future API use
    _ = observations, ground_state_threshold_ev

    # Note: This requires lower level energy which is not in the basic
    # LineObservation dataclass. In practice, this would query the database.
    # For now, return empty set - caller should provide from database.
    return set()


# ---------------------------------------------------------------------------
# DB + window candidate-line selection policies (opt-in)
# ---------------------------------------------------------------------------
#
# The ``LineSelector`` above scores already-extracted ``LineObservation``s. The
# functions below operate one step earlier: they pick *candidate* atomic-data
# lines per element directly from the atomic DB over a wavelength window, so a
# downstream extractor knows where to integrate intensities. This is the entry
# point used by the known-matrix / OPC mode.
#
# Two selection policies are offered via the ``policy`` argument of
# :func:`select_lines_by_policy`:
#
# * ``"emissivity"`` (DEFAULT, current behavior): rank by Boltzmann-weighted
#   emissivity over the neutral + singly-ionized stages, then take a wide
#   upper-level-energy (E_k) spread of strong, isolated lines (resonance lines
#   excluded with a too-few fallback). This mirrors the established
#   ``tests/benchmarks/ded_precision/line_lists.select_lines`` behavior, so the
#   default path is unchanged.
# * ``"neutral_anchor"`` (opt-in): the real-steel accuracy lever L2 promoted
#   from ``tests/benchmarks/real_steel/lever_l2_lines.select_l2_lines`` (see
#   docs/research/real-steel-opc-promotion.md). Per element it prefers NEUTRAL
#   (stage 1) lines with wide E_k spread so the Saha ion->total back-correction
#   is applied on (or near) the neutral plane instead of extrapolated up from
#   ion lines; admits a strong neutral *resonance* anchor when the element is
#   effectively neutral-resonance-only; and DROPS an element that has no usable
#   neutral line at all (better to omit a trace minor than let an ion-only Saha
#   extrapolation soak the closure, e.g. Cu 0.2 wt% recovered as ~93%).
#
# Both policies are pure atomic-data + window functions (no measured intensity,
# no recovered composition) — physics-only, no new dependencies.

#: Selection temperature (K) for the emissivity ranking under the default
#: ``"emissivity"`` policy. A fixed, physically reasonable alloy-plasma value;
#: never fit per sample or tuned to any ground truth.
DEFAULT_SELECT_T_K: float = 11000.0
#: Selection temperature (K) for the ``"neutral_anchor"`` policy (lever L2).
NEUTRAL_ANCHOR_SELECT_T_K: float = 8000.0
#: Under ``"neutral_anchor"``, a neutral resonance line is admitted as an anchor
#: only when the strongest neutral line is a resonance line that beats the best
#: non-resonance neutral line by at least this factor (i.e. the element is
#: effectively neutral-resonance-only; weak non-resonance neutrals will not
#: extract from real spectra).
RESONANCE_ANCHOR_RATIO: float = 8.0


@dataclass(frozen=True)
class SelectedLine:
    """A candidate atomic-data line chosen from the DB by a selection policy.

    This is the output of DB + window line selection: a pure atomic-data
    candidate carrying no measured intensity (that is added later by the
    extractor). Field set matches the benchmark ``LineSpec`` so existing line
    extractors can consume it unchanged.
    """

    element: str
    ionization_stage: int
    wavelength_nm: float
    E_k_ev: float
    g_k: float
    A_ki: float
    aki_uncertainty: Optional[float] = None
    is_resonance: bool = False


def _emissivity_weight(transition, temperature_K: float) -> float:
    """Boltzmann-weighted relative emissivity ``g_k * A_ki * exp(-E_k / kT)``."""
    return float(
        transition.g_k * transition.A_ki * np.exp(-transition.E_k_ev / (KB_EV * temperature_K))
    )


def _gather_candidate_transitions(
    db,
    element: str,
    stages: Sequence[int],
    window: Tuple[float, float],
    temperature_K: float,
    *,
    allow_resonance: bool,
) -> List:
    """Collect usable transitions over ``stages`` in ``window``, emissivity-sorted.

    Drops transitions with non-positive ``A_ki``/``g_k`` or a missing upper-level
    energy, and (when ``allow_resonance`` is False) resonance lines. The result
    is sorted by Boltzmann-weighted emissivity at ``temperature_K`` (descending).
    """
    wmin, wmax = window
    out: List = []
    for stage in stages:
        for tr in db.get_transitions(
            element, ionization_stage=stage, wavelength_min=wmin, wavelength_max=wmax
        ):
            if not tr.A_ki or tr.A_ki <= 0 or not tr.g_k or tr.g_k <= 0:
                continue
            if tr.E_k_ev is None:
                continue
            if not allow_resonance and getattr(tr, "is_resonance", False):
                continue
            out.append(tr)
    out.sort(key=lambda t: _emissivity_weight(t, temperature_K), reverse=True)
    return out


def _spread_pick(
    cands: List, n_lines: int, min_separation_nm: float, prefer_spread: bool = True
) -> List:
    """Take up to ``n_lines`` strong, isolated lines for the Boltzmann plane.

    With ``prefer_spread`` (default), bin the strong-line pool by upper-level
    energy and take the strongest isolated line per bin, so the chosen set is
    well-conditioned for a single-element Boltzmann slope (falls back to a
    strongest-first isolated greedy pick when the pool is too small or has no
    energy spread). With ``prefer_spread=False`` always use the strongest-first
    isolated greedy pick: when T is constrained jointly across elements (the
    Saha-Boltzmann graph) each element only needs accurate, isolated intensities,
    and forcing per-element E_k spread selects weak, blended high-E_k lines that
    corrupt dense-spectrum elements (the DED constrained-extraction regime).
    """

    def _isolated(tr, picked: List) -> bool:
        return all(abs(tr.wavelength_nm - c.wavelength_nm) >= min_separation_nm for c in picked)

    pool = cands[: max(n_lines * 4, n_lines)]
    eks = np.array([t.E_k_ev for t in pool], dtype=float)
    chosen: List = []
    if prefer_spread and len(pool) > n_lines and eks.size and float(eks.max() - eks.min()) > 0:
        edges = np.linspace(eks.min(), eks.max(), n_lines + 1)
        used: set = set()
        for b in range(n_lines):
            lo, hi = edges[b], edges[b + 1]
            last = b == n_lines - 1
            for i, tr in enumerate(pool):
                if i in used:
                    continue
                ek = tr.E_k_ev
                if (lo <= ek < hi) or (last and ek <= hi):
                    if _isolated(tr, chosen):
                        chosen.append(tr)
                        used.add(i)
                        break
        for i, tr in enumerate(pool):
            if len(chosen) >= n_lines:
                break
            if i not in used and _isolated(tr, chosen):
                chosen.append(tr)
                used.add(i)
    else:
        for tr in cands:
            if _isolated(tr, chosen):
                chosen.append(tr)
            if len(chosen) >= n_lines:
                break
    return chosen


def _to_selected_lines(element: str, transitions: List) -> List[SelectedLine]:
    return [
        SelectedLine(
            element=element,
            ionization_stage=int(tr.ionization_stage),
            wavelength_nm=float(tr.wavelength_nm),
            E_k_ev=float(tr.E_k_ev),
            g_k=float(tr.g_k),
            A_ki=float(tr.A_ki),
            aki_uncertainty=getattr(tr, "aki_uncertainty", None),
            is_resonance=bool(getattr(tr, "is_resonance", False)),
        )
        for tr in transitions
    ]


def _select_emissivity(
    db,
    element: str,
    window: Tuple[float, float],
    n_lines: int,
    stages: Sequence[int],
    select_temperature_K: float,
    min_separation_nm: float,
    exclude_resonance: bool,
    prefer_spread: bool = True,
) -> List[SelectedLine]:
    """Default policy: emissivity-ranked, wide-E_k spread over ``stages``."""
    cands = _gather_candidate_transitions(
        db, element, stages, window, select_temperature_K, allow_resonance=not exclude_resonance
    )
    if exclude_resonance and len(cands) < max(2, n_lines // 2):
        # Too few non-resonance lines: admit resonance lines as a fallback.
        cands = _gather_candidate_transitions(
            db, element, stages, window, select_temperature_K, allow_resonance=True
        )
    return _to_selected_lines(
        element, _spread_pick(cands, n_lines, min_separation_nm, prefer_spread=prefer_spread)
    )


def _select_neutral_anchor(
    db,
    element: str,
    window: Tuple[float, float],
    n_lines: int,
    select_temperature_K: float,
    min_separation_nm: float,
) -> List[SelectedLine]:
    """Lever-L2 policy: neutral-anchored, wide-E_k selection (stage 1 only).

    Returns an empty list when the element has no usable neutral line in band
    (so the caller drops it from the closure rather than observing it ion-only).
    """
    neutral = _gather_candidate_transitions(
        db, element, (1,), window, select_temperature_K, allow_resonance=True
    )
    nonres = [t for t in neutral if not getattr(t, "is_resonance", False)]
    res = [t for t in neutral if getattr(t, "is_resonance", False)]

    if len(nonres) >= 2:
        pool = list(nonres)
        # Admit the neutral resonance lines as an anchor only when the strongest
        # neutral line overall is a much stronger resonance line (the element is
        # effectively neutral-resonance-only in real spectra).
        if res and neutral and getattr(neutral[0], "is_resonance", False):
            best_res = _emissivity_weight(res[0], select_temperature_K)
            best_nonres = _emissivity_weight(nonres[0], select_temperature_K)
            if best_nonres <= 0 or best_res / best_nonres >= RESONANCE_ANCHOR_RATIO:
                pool = list(neutral)
    elif neutral:
        pool = list(neutral)  # too few non-resonance: admit resonance anchor
    else:
        return []  # no neutral line in band -> drop element from closure

    return _to_selected_lines(element, _spread_pick(pool, n_lines, min_separation_nm))


def select_lines_by_policy(
    db,
    element: str,
    window: Tuple[float, float],
    n_lines: int = 8,
    *,
    policy: str = "emissivity",
    stages: Sequence[int] = (1, 2),
    select_temperature_K: Optional[float] = None,
    min_separation_nm: float = 0.12,
    exclude_resonance: bool = True,
    prefer_spread: bool = True,
) -> List[SelectedLine]:
    """Pick up to ``n_lines`` candidate lines for ``element`` from the DB + window.

    Pure atomic-data + window selection (no measured intensity, no recovered
    composition). The ``policy`` argument selects the strategy; the default keeps
    the current emissivity-spread behavior so existing callers are unaffected.

    Parameters
    ----------
    db
        Atomic database exposing
        ``get_transitions(element, ionization_stage, wavelength_min, wavelength_max)``.
    element : str
        Element symbol.
    window : tuple of float
        ``(wavelength_min_nm, wavelength_max_nm)`` selection window.
    n_lines : int, default 8
        Maximum number of lines to return for the element.
    policy : {"emissivity", "neutral_anchor"}, default "emissivity"
        ``"emissivity"`` (default, unchanged behavior): emissivity-ranked,
        wide-E_k spread over ``stages``. ``"neutral_anchor"``: the opt-in
        real-steel lever-L2 policy (neutral-plane anchored, ion-only minors
        dropped); ``stages`` is ignored (neutral stage 1 only).
    stages : sequence of int, default (1, 2)
        Ionization stages to consider under the ``"emissivity"`` policy.
    select_temperature_K : float, optional
        Boltzmann-weighting temperature for the emissivity ranking. Defaults to
        a policy-appropriate fixed value (``DEFAULT_SELECT_T_K`` for
        ``"emissivity"``, ``NEUTRAL_ANCHOR_SELECT_T_K`` for ``"neutral_anchor"``).
    min_separation_nm : float, default 0.12
        Minimum wavelength separation between selected lines (isolation).
    exclude_resonance : bool, default True
        Under ``"emissivity"``, exclude resonance lines (with a too-few
        fallback). Ignored by ``"neutral_anchor"`` (which manages resonance
        anchoring itself).
    prefer_spread : bool, default True
        Under ``"emissivity"``, force a wide upper-level-energy (E_k) spread for
        a single-element Boltzmann slope. Set ``False`` to take the strongest,
        cleanest isolated lines instead (the DED constrained-extraction regime,
        where T is constrained jointly across elements via the Saha-Boltzmann
        graph so per-element spread-forcing only admits weak/blended high-E_k
        lines). Ignored by ``"neutral_anchor"``.

    Returns
    -------
    list of SelectedLine
        Selected candidate lines (possibly empty under ``"neutral_anchor"`` when
        the element has no usable neutral line in band).

    Raises
    ------
    ValueError
        If ``policy`` is not a recognized value.
    """
    if policy == "emissivity":
        temp = DEFAULT_SELECT_T_K if select_temperature_K is None else select_temperature_K
        return _select_emissivity(
            db,
            element,
            window,
            n_lines,
            stages,
            temp,
            min_separation_nm,
            exclude_resonance,
            prefer_spread=prefer_spread,
        )
    if policy == "neutral_anchor":
        temp = NEUTRAL_ANCHOR_SELECT_T_K if select_temperature_K is None else select_temperature_K
        return _select_neutral_anchor(db, element, window, n_lines, temp, min_separation_nm)
    raise ValueError(
        f"Unknown line-selection policy: {policy!r} (expected 'emissivity' or 'neutral_anchor')"
    )


# ---------------------------------------------------------------------------
# Matrix-isolation filter for dominant-matrix alloys (opt-in)
# ---------------------------------------------------------------------------
#
# In a dominant-matrix alloy (e.g. Ti-6Al-4V: Ti ~90 at.%, Al ~6 %, V ~4 %) the
# matrix element emits a dense forest of lines. A trace/minor element's analytical
# line that falls within the instrument resolution element of a strong matrix
# transition is *blended*: the extracted integrated intensity is contaminated by
# matrix emission, biasing the trace element high (V over-attribution) -- a
# per-line error that no per-element OPC scalar ``F`` can undo. Classical CF-LIBS
# line-selection guidance (Tognoni et al. 2006; Aragón & Aguilera 2008) requires
# *isolated, interference-free* analytical lines for exactly this reason.
#
# :func:`filter_matrix_blended_lines` removes the trace-element observations that
# are blended with a comparable-or-stronger matrix transition, while leaving the
# matrix element's own lines untouched. It is a pure atomic-data + window function
# (queries the DB for matrix transitions in the resolution window; never reads a
# recovered composition) -- physics-only, opt-in, default path unchanged.

#: Default assumed matrix:trace abundance ratio for the matrix-isolation filter.
#: An order-of-magnitude property of a "dominant-matrix" alloy (the matrix
#: element vastly outnumbers the traces) -- NOT a per-sample composition. Because
#: the matrix is this many times more abundant, a matrix transition need only
#: carry ~1/MATRIX_DOMINANCE of a trace line's per-atom emissivity to contribute
#: equally to a blended peak, so even a per-atom-weak matrix line contaminates.
DEFAULT_MATRIX_DOMINANCE: float = 15.0


def _instrument_fwhm_nm(
    wavelength_nm: float,
    resolving_power: Optional[float],
    fwhm_nm: Optional[float],
) -> float:
    """Instrument FWHM (nm) at ``wavelength_nm`` from an explicit value or R."""
    if fwhm_nm is not None and fwhm_nm > 0:
        return float(fwhm_nm)
    if resolving_power is not None and resolving_power > 0:
        return float(wavelength_nm) / float(resolving_power)
    return 0.0


def _matrix_contamination_ratio(
    obs: LineObservation,
    db,
    matrix_element: str,
    *,
    half_window_nm: float,
    matrix_dominance: float,
    abundance_prior: Optional[Dict[str, float]],
    select_temperature_K: float,
    stages: Sequence[int],
) -> Optional[float]:
    """Largest matrix:trace blend-contribution ratio inside the window.

    Returns the maximum, over matrix transitions within ``half_window_nm`` of the
    trace line ``obs``, of the abundance-scaled per-atom-emissivity ratio
    ``(g_k A_ki e^{-E_k/kT})_matrix * a_matrix / ((g_k A_ki e^{-E_k/kT})_trace *
    a_trace)``. ``a_*`` come from ``abundance_prior`` when both elements are
    present, else the scalar ``matrix_dominance`` (= a_matrix / a_trace). Returns
    ``None`` when the ratio cannot be judged (zero window, non-positive trace
    emissivity, or no matrix transition in band) so the caller keeps the line.
    """
    if half_window_nm <= 0.0:
        return None
    trace_w = float(obs.g_k * obs.A_ki * np.exp(-obs.E_k_ev / (KB_EV * select_temperature_K)))
    if trace_w <= 0.0:
        return None

    if abundance_prior is not None:
        a_matrix = float(abundance_prior.get(matrix_element, 0.0))
        a_trace = float(abundance_prior.get(obs.element, 0.0))
        dominance = (a_matrix / a_trace) if a_trace > 0 else matrix_dominance
    else:
        dominance = matrix_dominance

    lo, hi = obs.wavelength_nm - half_window_nm, obs.wavelength_nm + half_window_nm
    best: Optional[float] = None
    for stage in stages:
        for tr in db.get_transitions(
            matrix_element, ionization_stage=stage, wavelength_min=lo, wavelength_max=hi
        ):
            if not tr.A_ki or tr.A_ki <= 0 or not tr.g_k or tr.g_k <= 0 or tr.E_k_ev is None:
                continue
            matrix_w = float(tr.g_k * tr.A_ki * np.exp(-tr.E_k_ev / (KB_EV * select_temperature_K)))
            ratio = matrix_w * dominance / trace_w
            if best is None or ratio > best:
                best = ratio
    return best


def filter_matrix_blended_lines(
    observations: List[LineObservation],
    db,
    matrix_element: str,
    *,
    resolving_power: Optional[float] = None,
    fwhm_nm: Optional[float] = None,
    n_fwhm: float = 1.0,
    contamination_ratio: float = 0.3,
    matrix_dominance: float = DEFAULT_MATRIX_DOMINANCE,
    abundance_prior: Optional[Dict[str, float]] = None,
    select_temperature_K: float = DEFAULT_SELECT_T_K,
    stages: Sequence[int] = (1, 2),
    min_lines_per_element: int = 1,
) -> Tuple[List[LineObservation], List[LineObservation]]:
    """Drop trace-element lines blended with a strong matrix-element transition.

    For a dominant-matrix alloy (one element vastly more abundant than the rest),
    each non-``matrix_element`` observation is tested for blending: if a matrix
    transition lies within ``n_fwhm`` instrument-FWHM of the trace line AND its
    abundance-scaled per-atom emissivity is at least ``contamination_ratio`` of
    the trace line's contribution (see :func:`_matrix_contamination_ratio`), the
    trace line's extracted intensity is contaminated by matrix emission and the
    line is dropped. Matrix-element observations are always kept.

    A per-element floor protects against over-pruning: if filtering would leave a
    trace element with fewer than ``min_lines_per_element`` lines, its
    *least-contaminated* dropped lines are restored up to the floor (a faint but
    least-blended line still anchors the element's abundance on the shared
    Saha-Boltzmann plane). The default ``1`` keeps every element that had at
    least one line.

    Pure atomic-data + window function: it queries the DB for matrix transitions
    and computes Boltzmann-weighted emissivities, never a recovered composition.
    Physics-only and opt-in -- callers that do not invoke it are unaffected.

    Parameters
    ----------
    observations : list of LineObservation
        Detected/selected lines (mixed elements).
    db
        Atomic database exposing ``get_transitions(element, ionization_stage,
        wavelength_min, wavelength_max)``.
    matrix_element : str
        The dominant matrix element (e.g. ``"Ti"``). Its lines are never dropped.
    resolving_power : float, optional
        Instrument resolving power R; the FWHM at each trace line is ``lambda/R``.
        Ignored when ``fwhm_nm`` is given.
    fwhm_nm : float, optional
        Explicit instrument FWHM (nm), constant across the band. Overrides
        ``resolving_power`` when provided.
    n_fwhm : float, default 1.0
        Blend half-window in instrument FWHM: a matrix transition within
        ``n_fwhm * FWHM`` of a trace line is a potential blend.
    contamination_ratio : float, default 0.3
        Drop a trace line when the matrix:trace blend-contribution ratio (see
        :func:`_matrix_contamination_ratio`) is at least this value, i.e. the
        matrix contributes >= 30 % of the trace line's expected signal.
    matrix_dominance : float, default ``DEFAULT_MATRIX_DOMINANCE``
        Assumed matrix:trace abundance ratio used when ``abundance_prior`` is not
        supplied. An order-of-magnitude "dominant matrix" property, not a sample
        composition.
    abundance_prior : dict, optional
        Per-element number-fraction prior (e.g. a known feedstock spec). When it
        contains both the matrix and a trace element, their ratio replaces
        ``matrix_dominance`` for that trace. ``None`` (default) uses the scalar.
    select_temperature_K : float, default ``DEFAULT_SELECT_T_K``
        Boltzmann-weighting temperature for the per-atom emissivities.
    stages : sequence of int, default (1, 2)
        Matrix ionization stages searched for contaminating transitions.
    min_lines_per_element : int, default 1
        Per-trace-element floor restored from the least-contaminated drops.

    Returns
    -------
    (kept, dropped) : tuple of list of LineObservation
        ``kept`` preserves the input order; ``dropped`` are the removed blended
        trace lines.
    """
    half_base = float(n_fwhm)
    kept: List[LineObservation] = []
    # (obs, contamination_ratio) for trace lines flagged as blended, by element.
    flagged: Dict[str, List[Tuple[LineObservation, float]]] = defaultdict(list)
    kept_count: Dict[str, int] = defaultdict(int)

    for obs in observations:
        if obs.element == matrix_element:
            kept.append(obs)
            continue
        fwhm = _instrument_fwhm_nm(obs.wavelength_nm, resolving_power, fwhm_nm)
        ratio = _matrix_contamination_ratio(
            obs,
            db,
            matrix_element,
            half_window_nm=half_base * fwhm,
            matrix_dominance=matrix_dominance,
            abundance_prior=abundance_prior,
            select_temperature_K=select_temperature_K,
            stages=stages,
        )
        if ratio is not None and ratio >= contamination_ratio:
            flagged[obs.element].append((obs, ratio))
        else:
            kept.append(obs)
            kept_count[obs.element] += 1

    # Restore the least-contaminated flagged lines for any element left below the
    # per-element floor, so the filter never silently deletes an element.
    dropped: List[LineObservation] = []
    restore: Set[int] = set()
    for element, items in flagged.items():
        deficit = min_lines_per_element - kept_count[element]
        if deficit > 0:
            for obs, _ratio in sorted(items, key=lambda t: t[1])[:deficit]:
                restore.add(id(obs))

    if restore:
        # Re-emit in original order with restored lines kept.
        kept_ids = {id(o) for o in kept}
        kept = [o for o in observations if id(o) in kept_ids or id(o) in restore]
        for items in flagged.values():
            for obs, _ratio in items:
                if id(obs) not in restore:
                    dropped.append(obs)
    else:
        for items in flagged.values():
            for obs, _ratio in items:
                dropped.append(obs)

    if dropped:
        logger.info(
            "matrix-isolation filter (matrix=%s): dropped %d blended trace line(s), kept %d.",
            matrix_element,
            len(dropped),
            len(kept),
        )
    return kept, dropped
