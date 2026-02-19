"""
Tests for plasma state and Saha-Boltzmann solver.
"""

import pytest
from cflibs.plasma.state import PlasmaState, SingleZoneLTEPlasma
from cflibs.plasma import saha_boltzmann as saha_module
from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolver


def test_plasma_state_creation():
    """Test creating a plasma state."""
    plasma = PlasmaState(T_e=10000.0, n_e=1e17, species={"Fe": 1e15})
    assert plasma.T_e == 10000.0
    assert plasma.n_e == 1e17
    assert plasma.species == {"Fe": 1e15}


def test_plasma_state_temperature_properties():
    """Test temperature property conversions."""
    plasma = PlasmaState(T_e=10000.0, n_e=1e17, species={"Fe": 1e15})

    T_ev = plasma.T_e_eV
    assert T_ev > 0
    assert T_ev < 1.0  # Should be ~0.86 eV

    T_g_ev = plasma.T_g_eV
    assert T_g_ev == T_ev  # Defaults to T_e


def test_plasma_state_custom_tg():
    """Test plasma state with custom gas temperature."""
    plasma = PlasmaState(T_e=10000.0, n_e=1e17, species={"Fe": 1e15}, T_g=8000.0)
    assert plasma.T_g == 8000.0
    assert plasma.T_g_eV < plasma.T_e_eV


def test_single_zone_lte_plasma_creation(sample_plasma):
    """Test creating single-zone LTE plasma."""
    assert sample_plasma.T_e == 10000.0
    assert sample_plasma.n_e == 1e17
    assert len(sample_plasma.species) == 2


def test_single_zone_lte_plasma_validate_valid(sample_plasma):
    """Test validating valid plasma state."""
    assert sample_plasma.validate() is True


def test_single_zone_lte_plasma_validate_negative_temp():
    """Test validation fails for negative temperature."""
    plasma = SingleZoneLTEPlasma(T_e=-1000.0, n_e=1e17, species={"Fe": 1e15})
    with pytest.raises(ValueError, match="temperature must be positive"):
        plasma.validate()


def test_single_zone_lte_plasma_validate_negative_density():
    """Test validation fails for negative density."""
    plasma = SingleZoneLTEPlasma(T_e=10000.0, n_e=-1e17, species={"Fe": 1e15})
    with pytest.raises(ValueError, match="density must be positive"):
        plasma.validate()


def test_single_zone_lte_plasma_validate_no_species():
    """Test validation fails with no species."""
    plasma = SingleZoneLTEPlasma(T_e=10000.0, n_e=1e17, species={})
    with pytest.raises(ValueError, match="At least one species"):
        plasma.validate()


def test_saha_boltzmann_solver_init(atomic_db):
    """Test initializing Saha-Boltzmann solver."""
    solver = SahaBoltzmannSolver(atomic_db)
    assert solver.atomic_db == atomic_db


def test_solve_ionization_balance(atomic_db):
    """Test solving ionization balance."""
    solver = SahaBoltzmannSolver(atomic_db)

    result = solver.solve_ionization_balance(
        element="Fe", T_e_eV=0.86, n_e_cm3=1e17, total_density_cm3=1e15  # ~10000 K
    )

    assert isinstance(result, dict)
    assert 1 in result  # Neutral stage
    assert result[1] > 0
    assert result[1] <= 1e15  # Should be less than total


def test_solve_ionization_balance_conservation(atomic_db):
    """Test that ionization balance conserves total density."""
    solver = SahaBoltzmannSolver(atomic_db)

    total_density = 1e15
    result = solver.solve_ionization_balance(
        element="Fe", T_e_eV=0.86, n_e_cm3=1e17, total_density_cm3=total_density
    )

    total_result = sum(result.values())
    assert abs(total_result - total_density) / total_density < 0.01  # Within 1%


def test_calculate_partition_function(atomic_db):
    """Test calculating partition function."""
    solver = SahaBoltzmannSolver(atomic_db)

    U = solver.calculate_partition_function("Fe", 1, 0.86)
    assert U > 0
    # Partition function should be reasonable (typically 1-100 for most elements)
    assert 1 <= U <= 1000


@pytest.mark.unit
def test_missing_levels_warning_emitted_once(caplog):
    """Repeated missing-level fallback should warn only once per element/stage."""

    class _NoLevelsDB:
        def get_partition_coefficients(self, element, ionization_stage):
            return None

        def get_energy_levels(self, element, ionization_stage):
            return []

    solver = SahaBoltzmannSolver(_NoLevelsDB())
    previous = set(saha_module._MISSING_LEVEL_WARNED)
    try:
        saha_module._MISSING_LEVEL_WARNED.clear()
        with caplog.at_level("WARNING"):
            solver.calculate_partition_function.__wrapped__(solver, "Xx", 3, 1.0)
            solver.calculate_partition_function.__wrapped__(solver, "Xx", 3, 1.2)

        messages = [m for m in caplog.messages if "No energy levels for" in m]
        assert len(messages) == 1
    finally:
        saha_module._MISSING_LEVEL_WARNED.clear()
        saha_module._MISSING_LEVEL_WARNED.update(previous)


def test_solve_level_population(atomic_db):
    """Test solving level populations."""
    solver = SahaBoltzmannSolver(atomic_db)

    populations = solver.solve_level_population(
        element="Fe", ionization_stage=1, stage_density_cm3=1e15, T_e_eV=0.86
    )

    assert isinstance(populations, dict)
    assert len(populations) > 0

    # Check that populations are positive
    for pop in populations.values():
        assert pop >= 0

    # Check that total population is reasonable
    total_pop = sum(populations.values())
    assert total_pop > 0
    assert total_pop <= 1e15 * 1.1  # Should be close to stage density


def test_solve_plasma(atomic_db, sample_plasma):
    """Test solving complete plasma."""
    solver = SahaBoltzmannSolver(atomic_db)

    populations = solver.solve_plasma(sample_plasma)

    assert isinstance(populations, dict)
    assert len(populations) > 0

    # Should have populations for both Fe and H
    fe_keys = [k for k in populations.keys() if k[0] == "Fe"]
    h_keys = [k for k in populations.keys() if k[0] == "H"]

    assert len(fe_keys) > 0
    assert len(h_keys) > 0


def test_solve_plasma_temperature_dependence(atomic_db):
    """Test that populations depend on temperature."""
    solver = SahaBoltzmannSolver(atomic_db)

    plasma1 = SingleZoneLTEPlasma(T_e=5000.0, n_e=1e17, species={"Fe": 1e15})

    plasma2 = SingleZoneLTEPlasma(T_e=15000.0, n_e=1e17, species={"Fe": 1e15})

    pop1 = solver.solve_plasma(plasma1)
    pop2 = solver.solve_plasma(plasma2)

    # Higher temperature should have different populations
    # (exact comparison depends on energy levels)
    assert pop1 != pop2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
