"""Tests for the canonical partition-function fallback ladder.

Bead CF-LIBS-improved-16m7 (audit 2026-06-09, findings 01-F3 / 02-F1): the
historical hardcoded 25.0 / 15.0 / 2.0 ladder silently returned U = 15.0 for
Na II — a Ne-like closed shell whose true U is ~1.00 (first excited level
≈ 33 eV) — inflating the Na Saha multiplier ~15× at LIBS temperatures.  The
canonical helper replaces every scattered fallback site with one ladder:

(a) exact isoelectronic values (bare / hydrogen-like / noble-gas-like),
(b) the ground-level degeneracy g0 from ``energy_levels`` when available,
(c) the legacy generic constants, WITH a warning naming the species.
"""

import logging

import pytest

from cflibs.plasma.partition import (
    _FALLBACK_WARNED,
    canonical_partition_fallback,
    lookup_partition_function,
)


@pytest.fixture(autouse=True)
def _reset_warn_registry():
    """Each test sees a fresh warn-once registry."""
    saved = set(_FALLBACK_WARNED)
    _FALLBACK_WARNED.clear()
    yield
    _FALLBACK_WARNED.clear()
    _FALLBACK_WARNED.update(saved)


class _FakeLevel:
    def __init__(self, g, energy_ev):
        self.g = g
        self.energy_ev = energy_ev


class _FakeDb:
    """Minimal AtomicDataSource stand-in exposing only get_energy_levels."""

    db_path = ":fake:"

    def __init__(self, levels_by_species):
        self._levels = levels_by_species

    def get_energy_levels(self, element, ionization_stage):
        return self._levels.get((element, ionization_stage), [])


class TestClosedShellExactValues:
    """Tier (a): exact isoelectronic values — never warned, never generic."""

    @pytest.mark.parametrize(
        "element, stage, expected",
        [
            ("H", 2, 1.0),  # bare proton
            ("He", 3, 1.0),  # bare alpha
            ("H", 1, 2.0),  # hydrogen: ²S₁/₂ ground
            ("He", 2, 2.0),  # hydrogen-like
            ("He", 1, 1.0),  # He-like closed shell
            ("Li", 2, 1.0),  # He-like
            ("Na", 2, 1.0),  # Ne-like — THE audit F1 case (was 15.0)
            ("Mg", 3, 1.0),  # Ne-like
            ("K", 2, 1.0),  # Ar-like
            ("Ca", 3, 1.0),  # Ar-like
        ],
    )
    def test_exact_values(self, element, stage, expected):
        assert canonical_partition_fallback(element, stage) == expected

    def test_na_ii_regression_under_2(self):
        """Audit 02-F1 acceptance: U(Na II) < 2 at any LIBS temperature."""
        assert canonical_partition_fallback("Na", 2) < 2.0

    def test_closed_shell_emits_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="cflibs.plasma.partition"):
            canonical_partition_fallback("Na", 2)
        assert not caplog.records

    def test_open_shell_species_do_not_match(self):
        """Fe II (open 3d shell) must NOT get a closed-shell value."""
        value = canonical_partition_fallback("Fe", 2)
        assert value == 15.0  # generic tier (warned), not 1.0/2.0


class TestGroundDegeneracyTier:
    """Tier (b): g0 from energy_levels when a DB handle resolves levels."""

    def test_g0_used_when_levels_exist(self, caplog):
        db = _FakeDb({("Fe", 2): [_FakeLevel(10.0, 0.0), _FakeLevel(8.0, 0.5)]})
        with caplog.at_level(logging.WARNING):
            value = canonical_partition_fallback("Fe", 2, db)
        assert value == 10.0
        assert any("Fe" in rec.message and "g0" in rec.message for rec in caplog.records)

    def test_g0_skipped_without_levels(self):
        db = _FakeDb({})
        assert canonical_partition_fallback("Fe", 2, db) == 15.0


class TestGenericTierWarns:
    """Tier (c): legacy constants, loudly."""

    @pytest.mark.parametrize("stage, expected", [(1, 25.0), (2, 15.0), (3, 2.0)])
    def test_generic_constants(self, stage, expected):
        # 'Xx' is not a known element symbol -> straight to the generic tier.
        assert canonical_partition_fallback("Xx", stage) == expected

    def test_unknown_species_warns_naming_species(self, caplog):
        with caplog.at_level(logging.WARNING):
            canonical_partition_fallback("Xx", 1)
        messages = [rec.message for rec in caplog.records]
        assert any("Xx" in msg and "25.0" in msg for msg in messages)

    def test_warning_emitted_once_per_species(self, caplog):
        with caplog.at_level(logging.WARNING):
            canonical_partition_fallback("Xx", 1)
            canonical_partition_fallback("Xx", 1)
        warnings = [rec for rec in caplog.records if "Xx" in rec.message]
        assert len(warnings) == 1

    def test_warn_false_suppresses_logging(self, caplog):
        with caplog.at_level(logging.WARNING):
            canonical_partition_fallback("Xx", 1, warn=False)
        assert not caplog.records


class TestLookupPartitionFunction:
    """Dict-lookup wrapper used by the inversion call sites."""

    def test_present_entry_returned(self):
        assert lookup_partition_function({"Fe": 42.5}, "Fe", 1) == 42.5

    def test_missing_entry_routes_to_ladder(self):
        assert lookup_partition_function({}, "Na", 2) == 1.0

    def test_missing_open_shell_gets_generic_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            value = lookup_partition_function({}, "Xx", 1)
        assert value == 25.0
        assert any("Xx" in rec.message for rec in caplog.records)
