"""Unit tests for ALIAS_PRESETS + alias_preset() (arch candidate 2).

The presets capture the cocktails rediscovered three times via
bead-driven debugging (jbfg.2, n3rf.2, n3rf.4). These tests pin the
kwargs of each preset, the override-merge semantics, and the negative
case so the table can't silently drift.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from cflibs.inversion.identify.alias import ALIAS_PRESETS, ALIASIdentifier, alias_preset

# ---------------------------------------------------------------------------
# Expected canonical preset bodies — must match the table in alias.py.
# If the table changes intentionally, update this fixture deliberately.
# ---------------------------------------------------------------------------

_EXPECTED: Dict[str, Dict[str, Any]] = {
    "strict": {
        "r2_gate_mode": "fixed",
        "relative_cl_per_ion_stage": False,
        "high_recall": False,
        "intensity_threshold_factor": 3.0,
        "detection_threshold": 0.02,
        "chance_window_scale": 0.4,
        "max_lines_per_element": 30,
        "boltzmann_r2_min": 0.85,
    },
    "v2": {
        "r2_gate_mode": "adaptive_t",
        "relative_cl_per_ion_stage": True,
        "high_recall": False,
        "chance_window_scale": 0.4,
        "max_lines_per_element": 30,
    },
    "high_recall_v2": {
        "r2_gate_mode": "adaptive_t",
        "relative_cl_per_ion_stage": True,
        "high_recall": True,
        "chance_window_scale": 0.4,
        "max_lines_per_element": 30,
    },
    "consensus_voter": {
        "r2_gate_mode": "adaptive_t",
        "relative_cl_per_ion_stage": True,
        "high_recall": False,
        "chance_window_scale": 0.4,
        "max_lines_per_element": 30,
    },
}


def test_all_expected_presets_registered() -> None:
    """Every documented cocktail is present in the registry."""
    assert set(ALIAS_PRESETS) == set(_EXPECTED), (
        f"ALIAS_PRESETS keys drifted: registry={sorted(ALIAS_PRESETS)} "
        f"expected={sorted(_EXPECTED)}"
    )


@pytest.mark.parametrize("name", sorted(_EXPECTED))
def test_preset_kwargs_match_expected(name: str) -> None:
    """Each preset's kwargs are exactly the documented set."""
    assert ALIAS_PRESETS[name] == _EXPECTED[name]


@pytest.mark.parametrize("name", sorted(_EXPECTED))
def test_alias_preset_round_trip(name: str, mock_atomic_db) -> None:
    """``alias_preset(name)`` produces an ``ALIASIdentifier`` whose attributes
    match ``ALIAS_PRESETS[name]`` for the kwargs the constructor exposes."""
    ident = alias_preset(
        name,
        atomic_db=mock_atomic_db,
        elements=["Fe", "Cu"],
        resolving_power=5000.0,
    )
    assert isinstance(ident, ALIASIdentifier)
    for key, expected in ALIAS_PRESETS[name].items():
        # ALIASIdentifier stores each cocktail kwarg as an attribute of
        # the same name (verified by inspection of __init__). Compare
        # field-by-field rather than full-dict so future-added kwargs
        # don't break us.
        actual = getattr(ident, key)
        assert actual == expected, (
            f"alias_preset({name!r}).{key} = {actual!r}, "
            f"expected {expected!r} from ALIAS_PRESETS"
        )


def test_unknown_preset_raises_keyerror(mock_atomic_db) -> None:
    """Asking for an unregistered preset raises ``KeyError`` with the
    valid names in the message — fail-fast for typos."""
    with pytest.raises(KeyError, match="Unknown ALIAS preset 'definitely_not_a_preset'"):
        alias_preset(
            "definitely_not_a_preset",
            atomic_db=mock_atomic_db,
            elements=["Fe"],
            resolving_power=5000.0,
        )


def test_override_upgrades_v2_to_high_recall(mock_atomic_db) -> None:
    """``alias_preset("v2", high_recall=True)`` upgrades the v2 cocktail
    to high-recall — its attribute set equals ``high_recall_v2`` on the
    overlapping keys. This is the canonical "preset + override = different
    named preset" composability check from the arch review."""
    upgraded = alias_preset(
        "v2",
        high_recall=True,
        atomic_db=mock_atomic_db,
        elements=["Fe"],
        resolving_power=5000.0,
    )
    target = alias_preset(
        "high_recall_v2",
        atomic_db=mock_atomic_db,
        elements=["Fe"],
        resolving_power=5000.0,
    )
    for key in ALIAS_PRESETS["high_recall_v2"]:
        assert getattr(upgraded, key) == getattr(
            target, key
        ), f"v2+high_recall override drift on {key}"


def test_overrides_win_over_preset(mock_atomic_db) -> None:
    """A custom override beats the preset default, e.g. pinning
    ``chance_window_scale=0.6`` over ``v2``'s ``0.4``."""
    ident = alias_preset(
        "v2",
        chance_window_scale=0.6,
        atomic_db=mock_atomic_db,
        elements=["Fe"],
        resolving_power=5000.0,
    )
    assert ident.chance_window_scale == 0.6


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_atomic_db():
    """A minimal AtomicDatabase stand-in for constructor-only tests.
    ``alias_preset`` doesn't query atomic data, so a sentinel works."""

    class _Stub:
        """Empty stub; the ALIAS constructor stores the reference but
        doesn't dereference it until identify() is called."""

        def __repr__(self) -> str:  # pragma: no cover - debug aid only
            return "<AtomicDatabaseStub>"

    return _Stub()
