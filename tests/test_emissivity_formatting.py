import numpy as np
from cflibs.radiation.emissivity import calculate_line_emissivity, calculate_spectrum_emissivity

class MockTransition:
    def __init__(self, element="H", ion=1, ek=10.0, wl=500.0, aki=1e8):
        self.element = element
        self.ionization_stage = ion
        self.E_k_ev = ek
        self.wavelength_nm = wl
        self.A_ki = aki

def test_calculate_line_emissivity_basic():
    trans = MockTransition()
    pop = 1e12  # cm^-3
    eps = calculate_line_emissivity(trans, pop)
    assert isinstance(eps, (float, np.float64))
    assert np.isfinite(eps)
    assert eps > 0

def test_calculate_spectrum_emissivity_empty():
    wavelength_grid = np.linspace(400, 600, 100)
    # Empty transitions should return zeros
    spectrum = calculate_spectrum_emissivity([], {}, wavelength_grid, 0.1)
    assert np.all(spectrum == 0)
    assert spectrum.shape == wavelength_grid.shape
