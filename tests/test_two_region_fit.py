import pytest

from cflibs.plasma.state import TwoRegionPlasma


def test_two_region_plasma_state():
    species = {"Fe": 1e17, "Ca": 1e16}
    plasma = TwoRegionPlasma(
        T_core=10000.0,
        T_corona=8000.0,
        n_e=1e17,
        species=species,
    )
    assert plasma.T_core == pytest.approx(10000.0)
    assert plasma.T_corona == pytest.approx(8000.0)
    assert plasma.T_e == pytest.approx(10000.0)
    assert plasma.validate()


def test_two_region_plasma_validation_warning(caplog):
    caplog.set_level("WARNING", logger="plasma.state")
    species = {"Fe": 1e17}
    plasma = TwoRegionPlasma(
        T_core=8000.0,
        T_corona=10000.0,
        n_e=1e17,
        species=species,
    )
    # validate() returns True but logs a warning
    assert plasma.validate() is True
    assert "Unphysical temperature gradient" in caplog.text

