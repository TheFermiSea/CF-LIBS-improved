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
from typing import List, Dict, Optional, Tuple, Set
import numpy as np

from cflibs.core.logging_config import get_logger
from cflibs.inversion.physics.boltzmann import LineObservation

logger = get_logger("inversion.line_selection")

# McWhirter criterion prefactor in the cgs/eV/K convention
#   n_e [cm^-3] >= 1.6e12 * sqrt(T[K]) * (ΔE[eV])^3
# (McWhirter 1965; see e.g. Cristoforetti et al. 2010, Spectrochim. Acta B 65, 86)
MCWHIRTER_PREFACTOR_CM3_K = 1.6e12


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
        """
        self.min_snr = min_snr
        self.min_energy_spread_ev = min_energy_spread_ev
        self.min_lines_per_element = min_lines_per_element
        self.exclude_resonance = exclude_resonance
        self.isolation_wavelength_nm = isolation_wavelength_nm
        self.max_lines_per_element = max_lines_per_element

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
        """Per element: sort, warn on spread, take top lines, reject the excess."""
        selected_scores: List[LineScore] = []
        for element, elem_scores in by_element.items():
            # Sort by score descending
            elem_scores.sort(key=lambda s: s.score, reverse=True)

            # Check energy spread
            self._warn_energy_spread(element, elem_scores, warnings)

            # Select top lines up to max
            n_select = min(len(elem_scores), self.max_lines_per_element)
            selected_scores.extend(elem_scores[:n_select])

            # Mark excess as rejected
            for score in elem_scores[n_select:]:
                score.rejection_reason = "Exceeded max lines per element"
                rejected_scores.append(score)

        return selected_scores

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
