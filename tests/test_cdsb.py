"""
Tests for Column Density Saha-Boltzmann (CD-SB) line-observation data structures.

Covers:
- CDSBLineObservation creation and oscillator strength calculation
- create_cdsb_observation / from_transition factory functions

The legacy CDSBPlotter algorithm was removed (audit defects #14b/#14c/#13c);
its tests were dropped with it. The production self-absorption correction is
covered by tests/test_self_absorption.py.
"""

import numpy as np
from cflibs.inversion.physics.cdsb import (
    CDSBLineObservation,
    create_cdsb_observation,
    from_transition,
)
from cflibs.atomic.structures import Transition


class TestCDSBLineObservation:
    """Tests for CDSBLineObservation dataclass."""

    def test_basic_creation(self):
        """Test basic observation creation."""
        obs = CDSBLineObservation(
            wavelength_nm=500.0,
            intensity=1000.0,
            intensity_uncertainty=50.0,
            element="Fe",
            ionization_stage=1,
            E_k_ev=4.0,
            g_k=5,
            A_ki=1e8,
            E_i_ev=0.0,
            g_i=3,
            is_resonance=True,
        )

        assert obs.wavelength_nm == 500.0
        assert obs.element == "Fe"
        assert obs.is_resonance is True
        assert obs.E_i_ev == 0.0

    def test_oscillator_strength_calculation(self):
        """Test that oscillator strength is calculated from A_ki."""
        obs = CDSBLineObservation(
            wavelength_nm=500.0,
            intensity=1000.0,
            intensity_uncertainty=50.0,
            element="Fe",
            ionization_stage=1,
            E_k_ev=4.0,
            g_k=5,
            A_ki=1e8,
            E_i_ev=0.0,
            g_i=3,
            is_resonance=True,
        )

        # f_ik should be calculated
        assert obs.f_ik is not None
        assert obs.f_ik > 0

        # Verify formula: f_ik ~ 1.499e-14 * (g_k/g_i) * lambda^2 * A_ki
        expected_f = 1.499e-14 * (5 / 3) * 500.0**2 * 1e8
        assert abs(obs.f_ik - expected_f) / expected_f < 1e-6

    def test_y_value_inherited(self):
        """Test that y_value property is inherited from base class."""
        obs = CDSBLineObservation(
            wavelength_nm=500.0,
            intensity=100.0,
            intensity_uncertainty=5.0,
            element="Fe",
            ionization_stage=1,
            E_k_ev=3.0,
            g_k=2,
            A_ki=1e6,
            E_i_ev=0.0,
            g_i=1,
            is_resonance=True,
        )

        # y = ln(I * lam / (g * A))
        expected_y = np.log(100.0 * 500.0 / (2 * 1e6))
        assert abs(obs.y_value - expected_y) < 1e-10


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_cdsb_observation(self):
        """Test create_cdsb_observation factory function."""
        obs = create_cdsb_observation(
            wavelength_nm=589.0,
            intensity=5000.0,
            intensity_uncertainty=100.0,
            element="Na",
            ionization_stage=1,
            E_k_ev=2.1,
            E_i_ev=0.0,
            g_k=2,
            g_i=2,
            A_ki=6.16e7,
        )

        assert obs.element == "Na"
        assert obs.is_resonance is True  # Auto-detected from E_i_ev < 0.1

    def test_create_cdsb_observation_nonresonance(self):
        """Test auto-detection of non-resonance lines."""
        obs = create_cdsb_observation(
            wavelength_nm=600.0,
            intensity=1000.0,
            intensity_uncertainty=50.0,
            element="Fe",
            ionization_stage=1,
            E_k_ev=5.0,
            E_i_ev=1.5,  # Excited lower level
            g_k=7,
            g_i=5,
            A_ki=1e7,
        )

        assert obs.is_resonance is False  # Auto-detected

    def test_from_transition(self):
        """Test from_transition factory function."""
        transition = Transition(
            element="Ca",
            ionization_stage=1,
            wavelength_nm=422.67,
            A_ki=2.18e8,
            E_k_ev=2.93,
            E_i_ev=0.0,
            g_k=3,
            g_i=1,
            is_resonance=True,
        )

        obs = from_transition(transition, intensity=10000.0, intensity_uncertainty=200.0)

        assert obs.element == "Ca"
        assert obs.wavelength_nm == 422.67
        assert obs.is_resonance is True
        assert obs.E_i_ev == 0.0
