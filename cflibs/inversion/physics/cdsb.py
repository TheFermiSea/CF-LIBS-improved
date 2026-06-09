"""
Column Density Saha-Boltzmann (CD-SB) line-observation data structures.

This module provides the :class:`CDSBLineObservation` data structure (a
:class:`LineObservation` extended with lower-level information) and its
factory helpers. The self-absorption *correction* itself now lives in the
production :class:`cflibs.inversion.physics.self_absorption.SelfAbsorptionCorrector`;
the legacy ``CDSBPlotter`` algorithm was removed (it carried physics-audit
defects #14b/#14c/#13c).

Theory
------
The CD-SB data model captures resonance lines with significant
self-absorption, which cause standard Boltzmann plots to fail.

Theory
------
Standard Boltzmann plot:
    ln(I*lambda/gA) = ln(F*C/U) - E_k/(k*T)

For optically thin lines, this gives a linear relationship with slope -1/(kT).
However, for optically thick lines (especially resonance lines), the observed
intensity is reduced by self-absorption:

    I_obs = I_true * (1 - exp(-tau)) / tau

where tau is the optical depth at line center.

The CD-SB method modifies the Boltzmann plot ordinate to account for this:
    y_corrected = ln(I*lambda*tau/(gA*(1-exp(-tau)))) = ln(N*L/U) - E_k/(k*T)

where N*L is the column density (number density times path length).

Key Physics
-----------
- Optical depth: tau = alpha * N_i * L
  where alpha is the absorption cross-section and N_i is the lower level population

- For resonance lines (E_i = 0), the lower level is the ground state,
  maximizing self-absorption

- The CD-SB iteration:
  1. Start with initial tau estimate (from line intensities or plasma parameters)
  2. Calculate corrected y-values
  3. Fit Boltzmann plot to get T
  4. Update tau estimates using new T
  5. Repeat until convergence

References
----------
- Aragon, C. & Aguilera, J.A. (2008): CSigma graphs for quantitative LIBS
- El Sherbini et al. (2020): Self-absorption correction methods
- Cowan, R.D. (1981): Theory of Atomic Structure and Spectra
- Aguilera, J.A. & Aragon, C. (2004): Multi-element LIBS analysis
"""

from dataclasses import dataclass
from typing import Optional

from cflibs.inversion.physics.boltzmann import LineObservation


@dataclass
class CDSBLineObservation(LineObservation):
    """
    Extended line observation with lower level information for CD-SB.

    Additional Attributes
    ---------------------
    E_i_ev : float
        Lower level energy in eV (0 for resonance lines)
    g_i : int
        Lower level statistical weight
    is_resonance : bool
        True if this is a resonance line (ground state transition)
    f_ik : float, optional
        Absorption oscillator strength (calculated from A_ki if not provided)
    """

    E_i_ev: float = 0.0
    g_i: int = 1
    is_resonance: bool = False
    f_ik: Optional[float] = None

    def __post_init__(self):
        """Calculate oscillator strength if not provided."""
        if self.f_ik is None:
            # f_ik = (m_e * c / 8 * pi^2 * e^2) * (g_k / g_i) * lambda^2 * A_ki
            # Simplified formula: f_ik ~ 1.499e-14 * (g_k/g_i) * lambda_nm^2 * A_ki
            self.f_ik = 1.499e-14 * (self.g_k / self.g_i) * self.wavelength_nm**2 * self.A_ki


def create_cdsb_observation(
    wavelength_nm: float,
    intensity: float,
    intensity_uncertainty: float,
    element: str,
    ionization_stage: int,
    E_k_ev: float,
    E_i_ev: float,
    g_k: int,
    g_i: int,
    A_ki: float,
    is_resonance: Optional[bool] = None,
) -> CDSBLineObservation:
    """
    Factory function to create a CDSBLineObservation.

    Parameters
    ----------
    wavelength_nm : float
        Transition wavelength in nm
    intensity : float
        Measured line intensity (integrated area)
    intensity_uncertainty : float
        Uncertainty in intensity measurement
    element : str
        Element symbol (e.g., 'Fe', 'Ca')
    ionization_stage : int
        Ionization stage (1=neutral, 2=singly ionized)
    E_k_ev : float
        Upper level energy in eV
    E_i_ev : float
        Lower level energy in eV
    g_k : int
        Upper level statistical weight
    g_i : int
        Lower level statistical weight
    A_ki : float
        Einstein A coefficient (spontaneous emission rate) in s^-1
    is_resonance : bool, optional
        Whether this is a resonance line. If None, auto-detected from E_i_ev.

    Returns
    -------
    CDSBLineObservation
    """
    if is_resonance is None:
        # Auto-detect: resonance if lower level is near ground state
        is_resonance = E_i_ev < 0.1  # Within 0.1 eV of ground

    return CDSBLineObservation(
        wavelength_nm=wavelength_nm,
        intensity=intensity,
        intensity_uncertainty=intensity_uncertainty,
        element=element,
        ionization_stage=ionization_stage,
        E_k_ev=E_k_ev,
        g_k=g_k,
        A_ki=A_ki,
        E_i_ev=E_i_ev,
        g_i=g_i,
        is_resonance=is_resonance,
    )


def from_transition(
    transition,  # Transition dataclass from cflibs.atomic.structures
    intensity: float,
    intensity_uncertainty: float,
) -> CDSBLineObservation:
    """
    Create CDSBLineObservation from a Transition and measured intensity.

    Parameters
    ----------
    transition : Transition
        Transition dataclass from atomic database
    intensity : float
        Measured line intensity
    intensity_uncertainty : float
        Intensity uncertainty

    Returns
    -------
    CDSBLineObservation
    """
    is_resonance = getattr(transition, "is_resonance", None)
    if is_resonance is None:
        is_resonance = transition.E_i_ev < 0.1

    return CDSBLineObservation(
        wavelength_nm=transition.wavelength_nm,
        intensity=intensity,
        intensity_uncertainty=intensity_uncertainty,
        element=transition.element,
        ionization_stage=transition.ionization_stage,
        E_k_ev=transition.E_k_ev,
        g_k=transition.g_k,
        A_ki=transition.A_ki,
        E_i_ev=transition.E_i_ev,
        g_i=transition.g_i,
        is_resonance=is_resonance,
    )
