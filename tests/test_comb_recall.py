import numpy as np
import pytest
from cflibs.inversion.identify.comb import CombIdentifier


@pytest.mark.unit
def test_comb_identifier_trace_logic():
    """Verify that CombIdentifier logic supports trace elements with few lines."""

    # Mock a database-like object that returns a few lines for an element
    class MockDB:
        def get_available_elements(self):
            return ["V"]

        def get_transitions(self, element, wavelength_min, wavelength_max):
            # Return 50 lines for V, but only 2 will be 'active' in our test
            from cflibs.atomic.structures import Transition

            return [
                Transition(
                    element=element,
                    wavelength_nm=310.0 + i * 0.1,
                    ionization_stage=1,
                    A_ki=1e8,
                    g_k=1,
                    g_i=1,
                    E_k_ev=4.0,
                    E_i_ev=0.0,
                )
                for i in range(50)
            ]

    db = MockDB()
    # Initialize with tuned parameters
    identifier = CombIdentifier(
        db,
        min_correlation=0.05,
        min_active_teeth=2,
        threshold_percentile=0.0,
    )

    # Create spectrum with 2 strong peaks for V
    wavelength = np.linspace(300, 320, 2000)
    intensity = np.zeros_like(wavelength)
    intensity[0] = 1.0
    # Add 2 peaks at 310.0 and 310.1
    for p in [310.0, 310.1]:
        idx = np.argmin(np.abs(wavelength - p))
        intensity[idx - 2 : idx + 3] = 100

    result = identifier.identify(wavelength, intensity)

    # With 50 lines, and 2 active teeth (correlation ~1.0 each)
    # Old score: 2.0 / 50 = 0.04 (Rejected if min_correlation=0.1 or 0.05)
    # New score: 2.0 / 10 = 0.20 (Detected!)

    detected_elements = [e.element for e in result.detected_elements]
    assert "V" in detected_elements
    v_id = next(e for e in result.detected_elements if e.element == "V")
    assert v_id.score >= 0.15
    assert v_id.score <= 1.0
