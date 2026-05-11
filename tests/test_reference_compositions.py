"""Tests for cflibs/benchmark/reference_compositions.py.

Verifies that certified compositions for community CF-LIBS reference
materials (BHVO-2, BIR-1, NIST SRM 612) are internally consistent and
match published values from GeoReM / NIST databases.
"""
from cflibs.benchmark.reference_compositions import (
    BHVO2_BASALT_USGS,
    BIR1_BASALT_USGS,
    NIST_SRM_612_GLASS,
    REFERENCE_COMPOSITIONS,
    get_reference_composition,
)


def test_bhvo2_majors_match_georem_oxide_compilation():
    """BHVO-2 elemental mass fractions, derived from Jochum 2005 GeoReM
    oxide compilation, should sum to ~0.55 (the rest is oxygen). Each
    major element should match the stoichiometric conversion within 1%
    of the source oxide value."""
    # Reference oxide values (wt%) from GeoReM compilation
    # × conversion factor → elemental mass fraction (0-1 scale)
    expected_oxide_to_element = {
        "Si": 0.499 * (28.09 / 60.08),   # SiO2
        "Al": 0.135 * (53.96 / 101.96),  # Al2O3
        "Fe": 0.123 * (111.69 / 159.69), # Fe2O3 (total Fe)
        "Mg": 0.0723 * (24.31 / 40.30),  # MgO
        "Ca": 0.114 * (40.08 / 56.08),   # CaO
    }
    for element, expected in expected_oxide_to_element.items():
        assert element in BHVO2_BASALT_USGS, f"{element} missing from BHVO-2"
        actual = BHVO2_BASALT_USGS[element]
        assert abs(actual - expected) / expected < 0.01, (
            f"BHVO-2 {element}: {actual:.4f} vs expected {expected:.4f} "
            f"(>1% deviation from oxide-derived value)"
        )


def test_bhvo2_mass_balance():
    """Sum of cation mass fractions + implicit oxygen should be ≤ 1.0.
    The implicit oxygen for a basalt (~45% by mass) brings the total
    near 1.0; cations alone should be ~0.55."""
    cation_sum = sum(BHVO2_BASALT_USGS.values())
    assert 0.50 < cation_sum < 0.60, (
        f"BHVO-2 cation sum {cation_sum:.4f} outside expected 0.50-0.60 "
        "(basalts are ~55% cations / 45% oxygen by mass)"
    )


def test_bir1_low_K_preserved():
    """BIR-1's very low K (0.027 wt% K2O → 0.000224 mass fraction) is a
    diagnostic trace-element feature; it should NOT be filtered out as
    'trace below LOQ'. Used to test composition_strata.minors recall."""
    assert "K" in BIR1_BASALT_USGS
    assert BIR1_BASALT_USGS["K"] < 0.001  # in minor range per protocol
    assert BIR1_BASALT_USGS["K"] > 0.0001  # but not below LOQ


def test_nist_srm_612_majors_only():
    """SRM 612 is doped with ~38 ppm trace elements but the table here
    should contain ONLY the matrix-glass majors (Si, Al, Ca, Na).
    Trace dopants are below LOQ for typical CF-LIBS and stored
    elsewhere per protocol.yaml composition_strata.traces.bound."""
    assert set(NIST_SRM_612_GLASS.keys()) == {"Si", "Al", "Ca", "Na"}
    # All four elements should be in the >0.001 (minor or major) range
    for element, fraction in NIST_SRM_612_GLASS.items():
        assert fraction > 0.001, (
            f"SRM 612 {element}={fraction:.4f} is in trace range; "
            "should not be in majors-only table"
        )


def test_registry_contains_expected_datasets():
    """The REFERENCE_COMPOSITIONS dict must include all three CRMs so
    downstream ingest code can look them up by dataset_id."""
    expected = {"bhvo2_usgs", "bir1_usgs", "nist_srm_612"}
    assert set(REFERENCE_COMPOSITIONS.keys()) == expected


def test_get_reference_composition_returns_known_and_none():
    """The accessor returns the right dict for known dataset_ids and
    None for unknown ones (callers fall through to legacy
    true_composition for aalto_libs / aa1100_substrate)."""
    assert get_reference_composition("bhvo2_usgs") == BHVO2_BASALT_USGS
    assert get_reference_composition("nist_srm_612") == NIST_SRM_612_GLASS
    assert get_reference_composition("aalto_libs") is None
    assert get_reference_composition("aa1100_substrate") is None


def test_si_present_in_all_crms():
    """Si is the matrix element for all three reference materials —
    every CRM should have a non-zero Si entry. This is the element
    against which subcompositional ratios (Fe/Si, Mg/Si, Ca/Si,
    Al/Si per protocol.yaml) are normalised."""
    for did, comp in REFERENCE_COMPOSITIONS.items():
        assert "Si" in comp, f"{did} missing Si"
        assert comp["Si"] > 0.001, f"{did} Si below LOQ"
