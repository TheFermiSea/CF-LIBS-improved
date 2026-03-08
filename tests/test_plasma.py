"""
Tests for plasma state and Saha-Boltzmann solver.
"""

import pytest
from cflibs.plasma.state import (
    PlasmaState,
    SingleZoneLTEPlasma,
    mass_fractions_to_number_fractions,
    mass_fractions_to_species_densities,
    number_fractions_to_species_densities,
    species_densities_to_number_fractions,
)
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


def test_species_densities_to_number_fractions():
    fractions = species_densities_to_number_fractions({"Fe": 2.0e15, "Cu": 1.0e15})
    assert sum(fractions.values()) == pytest.approx(1.0)
    assert fractions["Fe"] == pytest.approx(2.0 / 3.0)
    assert fractions["Cu"] == pytest.approx(1.0 / 3.0)


def test_species_densities_to_number_fractions_rejects_negative_components():
    with pytest.raises(ValueError, match="species cannot contain negative components: Cu"):
        species_densities_to_number_fractions({"Fe": 0.8, "Cu": -0.2})


def test_number_fractions_to_species_densities():
    species = number_fractions_to_species_densities(
        {"Fe": 0.25, "Cu": 0.75},
        total_number_density_cm3=2.0e17,
    )
    assert sum(species.values()) == pytest.approx(2.0e17)
    assert species["Fe"] == pytest.approx(5.0e16)
    assert species["Cu"] == pytest.approx(1.5e17)


def test_mass_fractions_to_number_fractions():
    fractions = mass_fractions_to_number_fractions(
        {"Fe": 0.5, "Cu": 0.5},
        {"Fe": 55.85, "Cu": 63.55},
    )
    assert sum(fractions.values()) == pytest.approx(1.0)
    assert fractions["Fe"] > fractions["Cu"]


def test_mass_fractions_to_number_fractions_rejects_negative_components():
    with pytest.raises(
        ValueError,
        match="mass_fractions cannot contain negative components: Cu",
    ):
        mass_fractions_to_number_fractions(
            {"Fe": 0.8, "Cu": -0.2},
            {"Fe": 55.85, "Cu": 63.55},
        )


def test_mass_fractions_to_species_densities():
    species = mass_fractions_to_species_densities(
        {"Fe": 0.5, "Cu": 0.5},
        total_number_density_cm3=1.0e17,
        atomic_masses_amu={"Fe": 55.85, "Cu": 63.55},
    )
    assert sum(species.values()) == pytest.approx(1.0e17)
    assert species["Fe"] > species["Cu"]


def test_single_zone_lte_plasma_from_number_fractions():
    plasma = SingleZoneLTEPlasma.from_number_fractions(
        T_e=10000.0,
        n_e=1.0e17,
        number_fractions={"Fe": 0.25, "Cu": 0.75},
        total_species_density_cm3=2.0e17,
    )
    assert plasma.species["Fe"] == pytest.approx(5.0e16)
    assert plasma.species["Cu"] == pytest.approx(1.5e17)
    assert plasma.species_number_fractions["Fe"] == pytest.approx(0.25)


def test_single_zone_lte_plasma_from_mass_fractions():
    plasma = SingleZoneLTEPlasma.from_mass_fractions(
        T_e=10000.0,
        n_e=1.0e17,
        mass_fractions={"Fe": 0.5, "Cu": 0.5},
        total_species_density_cm3=1.0e17,
        atomic_masses_amu={"Fe": 55.85, "Cu": 63.55},
    )
    assert sum(plasma.species.values()) == pytest.approx(1.0e17)
    assert plasma.species["Fe"] > plasma.species["Cu"]


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


# --- Ionization Fraction Diagnostic Tests ---


def test_get_ionization_fractions(atomic_db):
    """Test get_ionization_fractions returns valid fractions."""
    solver = SahaBoltzmannSolver(atomic_db)

    fractions = solver.get_ionization_fractions("Fe", T_e_eV=0.8, n_e_cm3=1e17)

    assert isinstance(fractions, dict)
    assert len(fractions) >= 2  # At least neutral + singly ionized

    # Fractions should sum to 1
    total = sum(fractions.values())
    assert total == pytest.approx(1.0, rel=1e-6)

    # All fractions between 0 and 1
    for stage, frac in fractions.items():
        assert 0.0 <= frac <= 1.0, f"Stage {stage} fraction {frac} out of range"


def test_get_ionization_fractions_temperature_dependence(atomic_db):
    """Test that ion fractions shift with temperature."""
    solver = SahaBoltzmannSolver(atomic_db)

    fracs_low = solver.get_ionization_fractions("Fe", T_e_eV=0.5, n_e_cm3=1e17)
    fracs_high = solver.get_ionization_fractions("Fe", T_e_eV=1.5, n_e_cm3=1e17)

    # Higher T should have more ionization (lower neutral fraction)
    assert fracs_high[1] < fracs_low[1]  # Less neutral at higher T
    assert fracs_high[2] > fracs_low[2]  # More singly ionized at higher T


def test_three_stage_ionization_coupling(atomic_db):
    """Test that 3-stage Saha coupling solves the full system correctly.

    The coupled system should satisfy:
        n_I + n_II + n_III = n_total
        n_II / n_I = S1
        n_III / n_II = S2

    The old sequential approach computed n_I = n_total/(1+S1) first
    (ignoring S2), then split n_II. The correct solution is
    n_I = n_total / (1 + S1 + S1*S2).
    """
    solver = SahaBoltzmannSolver(atomic_db)

    # Use high T and low n_e to get significant third-stage population
    T_eV = 2.0
    n_e = 1e15
    n_total = 1e15

    result = solver.solve_ionization_balance("Fe", T_eV, n_e, n_total)

    # Verify conservation
    total = sum(result.values())
    assert total == pytest.approx(n_total, rel=1e-10)

    # At T=2 eV and n_e=1e15, stage 3 must be populated
    assert (
        3 in result and result[3] > 0
    ), f"Expected 3-stage ionization at T=2 eV, n_e=1e15; got stages {list(result.keys())}"
    n_I = result[1]
    n_II = result[2]
    n_III = result[3]

    # Recompute S1 and S2
    import numpy as np
    from cflibs.core.constants import SAHA_CONST_CM3

    U_I = solver.calculate_partition_function("Fe", 1, T_eV)
    U_II = solver.calculate_partition_function("Fe", 2, T_eV)
    U_III = solver.calculate_partition_function("Fe", 3, T_eV)
    ip_I = atomic_db.get_ionization_potential("Fe", 1)
    ip_II = atomic_db.get_ionization_potential("Fe", 2)

    S1 = (SAHA_CONST_CM3 / n_e) * (T_eV**1.5) * (U_II / U_I) * np.exp(-ip_I / T_eV)
    S2 = (SAHA_CONST_CM3 / n_e) * (T_eV**1.5) * (U_III / U_II) * np.exp(-ip_II / T_eV)

    # Verify n_I = n_total / (1 + S1 + S1*S2)
    expected_n_I = n_total / (1.0 + S1 + S1 * S2)
    assert n_I == pytest.approx(expected_n_I, rel=1e-10)
    assert n_II == pytest.approx(S1 * n_I, rel=1e-10)
    assert n_III == pytest.approx(S2 * n_II, rel=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
