import numpy as np
import pytest
from cflibs.inversion.physics.cdsb import CDSBPlotter, CDSBLineObservation

def test_cdsb_plotter_tuned_defaults():
    """Verify that CDSBPlotter uses the tuned defaults for Vrabel2020."""
    plotter = CDSBPlotter()
    assert plotter.max_iterations == 50
    assert plotter.convergence_tolerance == 0.001

def test_estimate_initial_tau_scaling():
    """Verify that initial tau estimation uses the increased base_tau and resonance boost."""
    plotter = CDSBPlotter()
    
    # Create a resonance line and a non-resonance line
    obs_res = CDSBLineObservation(
        wavelength_nm=285.21, # Mg I resonance
        intensity=1000.0,
        intensity_uncertainty=10.0,
        element="Mg",
        ionization_stage=1,
        E_k_ev=4.346,
        E_i_ev=0.0,
        g_k=3,
        g_i=1,
        A_ki=4.91e8,
        is_resonance=True
    )
    
    obs_non_res = CDSBLineObservation(
        wavelength_nm=383.8, # Mg I non-resonance
        intensity=1000.0,
        intensity_uncertainty=10.0,
        element="Mg",
        ionization_stage=1,
        E_k_ev=5.946,
        E_i_ev=2.709,
        g_k=5,
        g_i=3,
        A_ki=1.6e8,
        is_resonance=False
    )
    
    partition_funcs = {"Mg": 2.0}
    tau_values = plotter._estimate_initial_tau(
        [obs_res, obs_non_res],
        T_K=10000.0,
        n_e=1e17,
        partition_funcs=partition_funcs,
        stark_widths_nm=None
    )
    
    # Verify resonance line has significantly higher tau
    assert tau_values[285.21] > 3.0
    assert tau_values[383.8] < tau_values[285.21]
