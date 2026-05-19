#!/usr/bin/env python
"""Quick test to understand current comb behavior."""

import sys
import numpy as np
from pathlib import Path

from cflibs.atomic.database import AtomicDatabase
from cflibs.inversion.identify.comb import CombIdentifier

def main():
    db_path = Path("ASD_da/libs_production.db")
    
    if not db_path.exists():
        print(f"Atomic DB not found: {db_path}")
        return
    
    # Create a simple synthetic spectrum
    wavelength = np.linspace(370.0, 450.0, 1000)
    baseline = np.ones_like(wavelength) * 10.0
    
    # Add some peaks at known Fe lines
    intensity = baseline.copy()
    intensity += 1000.0 * np.exp(-0.5 * ((wavelength - 373.49) / 0.05) ** 2)
    intensity += 800.0 * np.exp(-0.5 * ((wavelength - 385.99) / 0.05) ** 2)
    intensity += 600.0 * np.exp(-0.5 * ((wavelength - 392.37) / 0.05) ** 2)
    
    print("Testing comb identifier with different parameters...")
    print(f"Spectrum: {wavelength[0]:.1f}-{wavelength[-1]:.1f} nm, {len(wavelength)} points")
    
    with AtomicDatabase(str(db_path)) as db:
        # Test different parameter combinations
        test_configs = [
            (0.12, 0.5),  # Current defaults
            (0.10, 0.5),
            (0.08, 0.5),
            (0.05, 0.5),
            (0.12, 0.4),
            (0.12, 0.3),
            (0.10, 0.4),
            (0.10, 0.3),
            (0.08, 0.4),
            (0.08, 0.3),
            (0.05, 0.4),
            (0.05, 0.3),
        ]
        
        for min_corr, tooth_act in test_configs:
            identifier = CombIdentifier(
                atomic_db=db,
                elements=["Fe"],
                min_correlation=min_corr,
                tooth_activation_threshold=tooth_act,
            )
            
            result = identifier.identify(wavelength, intensity)
            detected = [e.element for e in result.detected_elements]
            rejected = [e.element for e in result.rejected_elements]
            
            print(f"min_corr={min_corr:.2f}, tooth_act={tooth_act:.1f}: "
                  f"detected={detected}, rejected={rejected}")
            if result.detected_elements:
                for e in result.detected_elements:
                    print(f"  {e.element}: score={e.score:.4f}, n_matched={e.n_matched_lines}")

if __name__ == "__main__":
    main()
