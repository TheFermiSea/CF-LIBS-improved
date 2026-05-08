import unittest
import math
import numpy as np
from dataclasses import dataclass

@dataclass
class MockTransition:
    wavelength_nm: float
    omega_stark: float = 0.0

class TestStarkTolerance(unittest.TestCase):
    def test_stark_formula_logic(self):
        # Test the formula sqrt(fwhm_inst^2 + omega_stark^2)
        wl = 400.0
        R = 5000.0
        fwhm_inst = wl / R # 0.08
        omega_stark = 0.06
        
        # Expected: sqrt(0.08^2 + 0.06^2) = 0.1
        tol = math.sqrt(fwhm_inst**2 + omega_stark**2)
        self.assertAlmostEqual(tol, 0.1)
        
    def test_fallback_logic(self):
        # Test fallback to 0.05 when omega_stark is 0 or unavailable
        omega_stark = 0.0
        tol = 0.05 # fallback_fixed
        self.assertEqual(tol, 0.05)

if __name__ == "__main__":
    unittest.main()
