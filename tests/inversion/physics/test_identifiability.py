"""Tests for the identifiability guards / refuse-to-report gate.

Every guard's TRUE and FALSE branch is exercised (a test that only ever passes proves
nothing). Each guard cites a proven ``CflibsFormal`` identifiability theorem; here we pin
the structural precondition each theorem encodes:

- temperature: >=2 lines with DISTINCT upper-level energies (else ssE=0, slope 0/0).
- composition: >=1 line per element AND a temperature anchor.
- self-absorption: a single thick line with UNKNOWN tau is the (N,tau) alias; known tau OR
  >=2 distinct-optical-depth lines (curve-of-growth ratio) restores identifiability.
- refuse_to_report: combined gate; returns 'identifiable' or names the failed precondition.
"""

from __future__ import annotations

from cflibs.inversion.physics.identifiability import (
    IdentifiabilityResult,
    composition_identifiable,
    refuse_to_report,
    self_absorption_identifiable,
    temperature_identifiable,
)


# --------------------------------------------------------------- temperature guard
def test_temperature_identifiable_true_two_distinct_energies():
    res = temperature_identifiable([0.0, 1.2, 2.5])
    assert isinstance(res, IdentifiabilityResult)
    assert res.identifiable is True
    assert "temperature_identifiability" in res.reason


def test_temperature_not_identifiable_single_line():
    res = temperature_identifiable([1.5])
    assert res.identifiable is False
    assert "one species alone" in res.reason


def test_temperature_not_identifiable_all_equal_energies():
    # Several lines but all upper levels at the same energy -> zero lever arm (ssE=0).
    res = temperature_identifiable([2.0, 2.0, 2.0, 2.0])
    assert res.identifiable is False
    assert "ssE = 0" in res.reason


def test_temperature_not_identifiable_near_degenerate_within_tol():
    # Energies differing by less than the degeneracy tolerance count as one level.
    res = temperature_identifiable([1.0, 1.0 + 1e-12])
    assert res.identifiable is False


def test_temperature_identifiable_just_above_tol():
    # A spread above the tolerance IS a usable lever arm.
    res = temperature_identifiable([1.0, 1.0 + 1e-6])
    assert res.identifiable is True


# --------------------------------------------------------------- composition guard
def test_composition_identifiable_true():
    res = composition_identifiable({"Fe": 5, "Cu": 3}, has_temperature_anchor=True)
    assert res.identifiable is True
    assert "compositionIdentifiability" in res.reason


def test_composition_not_identifiable_missing_element_line():
    res = composition_identifiable({"Fe": 5, "Cu": 0}, has_temperature_anchor=True)
    assert res.identifiable is False
    assert "Cu" in res.reason


def test_composition_not_identifiable_no_temperature_anchor():
    res = composition_identifiable({"Fe": 5, "Cu": 3}, has_temperature_anchor=False)
    assert res.identifiable is False
    assert "temperature anchor" in res.reason


def test_composition_not_identifiable_empty():
    res = composition_identifiable({}, has_temperature_anchor=True)
    assert res.identifiable is False


# --------------------------------------------------------------- self-absorption guard
def test_self_absorption_identifiable_tau_known():
    # A single thick line is fine IF tau is externally known.
    res = self_absorption_identifiable(n_lines_same_species=1, tau_known=True)
    assert res.identifiable is True
    assert "tau is externally known" in res.reason


def test_self_absorption_identifiable_two_distinct_depth():
    res = self_absorption_identifiable(
        n_lines_same_species=3, tau_known=False, n_distinct_optical_depth=3
    )
    assert res.identifiable is True
    assert "cogRatio_injOn" in res.reason


def test_self_absorption_not_identifiable_single_thick_unknown_tau():
    # The headline (N, tau) alias: one thick line, unknown tau -> NOT identifiable.
    res = self_absorption_identifiable(n_lines_same_species=1, tau_known=False)
    assert res.identifiable is False
    assert "selfAbsorption_breaks_identifiability" in res.reason


def test_self_absorption_not_identifiable_no_distinct_depth():
    # Multiple lines but all at the SAME optical depth -> ratio uninformative.
    res = self_absorption_identifiable(
        n_lines_same_species=3, tau_known=False, n_distinct_optical_depth=1
    )
    assert res.identifiable is False
    assert "distinct optical depth" in res.reason


# --------------------------------------------------------------- refuse_to_report gate
def test_refuse_to_report_all_pass():
    res = refuse_to_report(
        upper_level_energies_ev=[0.0, 1.2, 2.5],
        lines_by_element={"Fe": 5, "Cu": 3},
        self_absorption_n_lines=1,
        self_absorption_tau_known=True,
    )
    assert res.identifiable is True
    assert res.reason == "identifiable"


def test_refuse_to_report_flags_temperature_first():
    res = refuse_to_report(
        upper_level_energies_ev=[1.5],  # single line -> T not identifiable
        lines_by_element={"Fe": 5},
    )
    assert res.identifiable is False
    assert res.reason.startswith("temperature_not_identifiable")


def test_refuse_to_report_infers_anchor_from_temperature():
    # No explicit anchor: it is inferred from the temperature guard passing.
    res = refuse_to_report(
        upper_level_energies_ev=[0.0, 1.2, 2.5],
        lines_by_element={"Fe": 5, "Cu": 3},
    )
    assert res.identifiable is True
    assert res.reason == "identifiable"


def test_refuse_to_report_flags_composition():
    # Temperature OK, but an element has no line.
    res = refuse_to_report(
        upper_level_energies_ev=[0.0, 1.2],
        lines_by_element={"Fe": 5, "Cu": 0},
    )
    assert res.identifiable is False
    assert res.reason.startswith("composition_not_identifiable")


def test_refuse_to_report_flags_composition_when_no_anchor_supplied():
    # Composition checked alone with no temperature info -> anchor inferred absent -> flag.
    res = refuse_to_report(lines_by_element={"Fe": 5, "Cu": 3})
    assert res.identifiable is False
    assert res.reason.startswith("composition_not_identifiable")
    assert "temperature anchor" in res.reason


def test_refuse_to_report_flags_self_absorption():
    res = refuse_to_report(
        upper_level_energies_ev=[0.0, 1.2, 2.5],
        lines_by_element={"Fe": 5},
        self_absorption_n_lines=1,
        self_absorption_tau_known=False,
    )
    assert res.identifiable is False
    assert res.reason.startswith("self_absorption_not_identifiable")


def test_refuse_to_report_nothing_supplied_refuses():
    # No preconditions checked -> cannot be trusted.
    res = refuse_to_report()
    assert res.identifiable is False
    assert "no identifiability preconditions" in res.reason
