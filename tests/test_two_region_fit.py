import pytest
import numpy as np
from cflibs.plasma.state import TwoRegionPlasma
from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver, LineObservation
from cflibs.atomic.database import AtomicDatabase

def test_two_region_plasma_state():
    species = {"Fe": 1e17, "Ca": 1e16}
    plasma = TwoRegionPlasma(
        T_core=10000.0,
        T_corona=8000.0,
        n_e=1e17,
        species=species
    )
    assert plasma.T_core == 10000.0
    assert plasma.T_corona == 8000.0
    assert plasma.T_e == 10000.0
    assert plasma.validate()

def test_two_region_plasma_validation_warning(caplog):
    species = {"Fe": 1e17}
    plasma = TwoRegionPlasma(
        T_core=8000.0,
        T_corona=10000.0,
        n_e=1e17,
        species=species
    )
    # validate() returns True but logs a warning
    assert plasma.validate() is True
    assert "Unphysical temperature gradient" in caplog.text

@pytest.mark.parametrize("two_region", [True, False])
def test_solver_two_region_flag(two_region):
    # Mock database and observations
    with AtomicDatabase(":memory:") as db:
        solver = IterativeCFLIBSSolver(db, two_region=two_region)
        assert solver.two_region == two_region
        
        # Create dummy observations
        obs = [
            LineObservation(
                wavelength_nm=300.0, intensity=1000.0, intensity_uncertainty=10.0,
                element="Fe", ionization_stage=1, E_k_ev=4.0, g_k=5.0, A_ki=1e8
            ),
            LineObservation(
                wavelength_nm=301.0, intensity=500.0, intensity_uncertainty=5.0,
                element="Fe", ionization_stage=1, E_k_ev=5.0, g_k=5.0, A_ki=1e8
            )
        ]
        
        # We don't expect it to converge with dummy data, but check result structure
        result = solver.solve(obs)
        if two_region:
            assert result.temperature_corona_K is not None
            # In our implementation, T_corona = 0.8 * T_K
            assert result.temperature_corona_K == pytest.approx(0.8 * result.temperature_K)
        else:
            assert result.temperature_corona_K is None
