"""Tests for air <-> vacuum wavelength conversion (cflibs.atomic.wavelength_conversion).

Anchors use canonical air/vacuum line pairs from the standard spectroscopic literature
(NIST ASD / Morton 2000), which is the convention the converter implements.
"""

import numpy as np
import pytest

from cflibs.atomic import (
    AIR_VACUUM_BOUNDARY_NM,
    air_to_vacuum_nm,
    vacuum_to_air_nm,
)

# (air_nm, vacuum_nm) canonical pairs. Tolerance 1e-3 nm = 0.01 Angstrom.
ANCHORS = [
    (656.279, 656.461),  # H-alpha (Balmer)
    (588.995, 589.158),  # Na D2
    (589.592, 589.755),  # Na D1
    (430.790, 430.911),  # Ca / Fe blend region (mid-visible)
]


@pytest.mark.physics
@pytest.mark.parametrize("air_nm, vac_nm", ANCHORS)
def test_air_to_vacuum_matches_literature(air_nm, vac_nm):
    assert air_to_vacuum_nm(air_nm) == pytest.approx(vac_nm, abs=1e-3)


@pytest.mark.physics
@pytest.mark.parametrize("air_nm, vac_nm", ANCHORS)
def test_vacuum_to_air_matches_literature(air_nm, vac_nm):
    assert vacuum_to_air_nm(vac_nm) == pytest.approx(air_nm, abs=1e-3)


@pytest.mark.physics
@pytest.mark.parametrize("wl_nm", [250.0, 400.0, 500.0, 656.279, 850.0, 970.0])
def test_round_trip_air_vac_air(wl_nm):
    """air -> vacuum -> air recovers the input to sub-pm precision."""
    assert vacuum_to_air_nm(air_to_vacuum_nm(wl_nm)) == pytest.approx(wl_nm, abs=1e-4)


@pytest.mark.physics
@pytest.mark.parametrize("wl_nm", [250.0, 400.0, 500.0, 850.0, 970.0])
def test_round_trip_vac_air_vac(wl_nm):
    """vacuum -> air -> vacuum recovers the input to sub-pm precision."""
    assert air_to_vacuum_nm(vacuum_to_air_nm(wl_nm)) == pytest.approx(wl_nm, abs=1e-4)


@pytest.mark.physics
def test_vacuum_longer_than_air():
    """Vacuum wavelength always exceeds the air wavelength above the boundary (n_air > 1)."""
    for air_nm in (250.0, 500.0, 900.0):
        assert air_to_vacuum_nm(air_nm) > air_nm


@pytest.mark.physics
def test_offset_magnitude_at_500nm():
    """The air/vacuum offset is ~+0.14 nm at 500 nm (n_air - 1 ~ 2.8e-4) -- the headline figure."""
    offset = air_to_vacuum_nm(500.0) - 500.0
    assert offset == pytest.approx(0.139, abs=0.01)


@pytest.mark.physics
def test_boundary_is_identity_in_vacuum_band():
    """At/below the 200 nm boundary both conventions are vacuum -> conversion is the identity."""
    for wl in (120.0, 180.0, AIR_VACUUM_BOUNDARY_NM):
        assert air_to_vacuum_nm(wl) == pytest.approx(wl, abs=1e-12)
        assert vacuum_to_air_nm(wl) == pytest.approx(wl, abs=1e-12)


@pytest.mark.physics
def test_array_input_elementwise_and_dtype():
    """Array inputs convert element-wise and respect the boundary per element."""
    wl = np.array([150.0, 250.0, 500.0, 656.279])  # first is below the boundary
    vac = air_to_vacuum_nm(wl)
    assert isinstance(vac, np.ndarray)
    assert vac.shape == wl.shape
    # below-boundary element unchanged
    assert vac[0] == pytest.approx(150.0, abs=1e-12)
    # above-boundary elements grew
    assert np.all(vac[1:] > wl[1:])
    # matches the scalar path
    for i, w in enumerate(wl):
        assert vac[i] == pytest.approx(air_to_vacuum_nm(float(w)), abs=1e-12)


@pytest.mark.physics
def test_scalar_in_scalar_out():
    """Scalar input returns a Python float, not a 0-d array."""
    assert isinstance(air_to_vacuum_nm(500.0), float)
    assert isinstance(vacuum_to_air_nm(500.0), float)
