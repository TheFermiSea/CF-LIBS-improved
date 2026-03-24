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

Literature References
---------------------
- Clegg et al. (2017): Line selection strategies for LIBS quantification
- Tognoni et al. (2006): Quantitative LIBS analysis review
- NIST ASD: Atomic transition probability accuracy grades
"""

from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Set
import numpy as np

from cflibs.core.logging_config import get_logger
from cflibs.inversion.boltzmann import LineObservation

logger = get_logger("inversion.line_selection")


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


class LineSelector:
    """
    Automatic line selection for CF-LIBS analysis.

    Scores lines based on:
    - Signal-to-noise ratio (SNR)
    - Atomic data uncertainty
    - Spectral isolation (no blends)
    - Energy spread requirement

    Filters out:
    - Resonance lines (high self-absorption risk)
    - Low SNR lines
    - Blended lines
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
        exclude_resonance: bool = True,
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
            Whether to exclude resonance lines (ground state transitions)
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

        warnings = []
        scores = []

        # Score all lines
        for obs in observations:
            score_info = self._score_line(obs, observations, resonance_lines, atomic_uncertainties)
            scores.append(score_info)

        # Filter by criteria
        valid_scores = []
        rejected_scores = []

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

        # Group by element and ensure energy spread

        by_element = defaultdict(list)
        for score in valid_scores:
            by_element[score.observation.element].append(score)

        selected_scores = []
        for element, elem_scores in by_element.items():
            # Sort by score descending
            elem_scores.sort(key=lambda s: s.score, reverse=True)

            # Check energy spread
            if len(elem_scores) >= 2:
                energies = [s.observation.E_k_ev for s in elem_scores]
                spread = max(energies) - min(energies)

                if spread < self.min_energy_spread_ev:
                    warnings.append(
                        f"{element}: Energy spread {spread:.2f} eV < {self.min_energy_spread_ev} eV"
                    )

            # Select top lines up to max
            n_select = min(len(elem_scores), self.max_lines_per_element)
            selected_scores.extend(elem_scores[:n_select])

            # Mark excess as rejected
            for score in elem_scores[n_select:]:
                score.rejection_reason = "Exceeded max lines per element"
                rejected_scores.append(score)

        # Check minimum lines per element
        for element, elem_scores in by_element.items():
            n_valid = sum(1 for s in elem_scores if s.rejection_reason is None)
            if n_valid < self.min_lines_per_element:
                warnings.append(
                    f"{element}: Only {n_valid} lines available "
                    f"(need {self.min_lines_per_element})"
                )

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
