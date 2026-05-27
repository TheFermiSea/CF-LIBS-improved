"""Unit tests for ALIAS_PRESETS + alias_preset() (arch candidate 2).

The presets capture the cocktails rediscovered three times via
bead-driven debugging (jbfg.2, n3rf.2, n3rf.4). These tests pin the
kwargs of each preset, the override-merge semantics, and the negative
case so the table can't silently drift.
"""

from __future__ import annotations

import pytest

from cflibs.inversion.identify.alias import ALIAS_PRESETS, ALIASIdentifier, alias_preset

# ---------------------------------------------------------------------------
# Defining properties — the kwargs that make each cocktail architecturally
# distinct. Pinning these (not every byte) catches identity drift without
# duplicating the full ALIAS_PRESETS table.
# ---------------------------------------------------------------------------

_EXPECTED_NAMES = frozenset({"strict", "v2", "high_recall_v2", "consensus_voter"})


def test_all_expected_presets_registered() -> None:
    """Every documented cocktail is present in the registry."""
    assert set(ALIAS_PRESETS) == _EXPECTED_NAMES, (
        f"ALIAS_PRESETS keys drifted: registry={sorted(ALIAS_PRESETS)} "
        f"expected={sorted(_EXPECTED_NAMES)}"
    )


def test_strict_preset_uses_fixed_gate() -> None:
    """The strict preset is the precision-king baseline."""
    p = ALIAS_PRESETS["strict"]
    assert p["r2_gate_mode"] == "fixed"
    assert p["relative_cl_per_ion_stage"] is False
    assert p["high_recall"] is False


def test_v2_preset_uses_adaptive_gates() -> None:
    """The v2 cocktail bakes in the PR #175 + #176 winners."""
    p = ALIAS_PRESETS["v2"]
    assert p["r2_gate_mode"] == "adaptive_t"
    assert p["relative_cl_per_ion_stage"] is True
    assert p["high_recall"] is False


def test_high_recall_v2_is_v2_plus_high_recall() -> None:
    """high_recall_v2 = v2 + high_recall=True (n3rf.2 fix)."""
    p = ALIAS_PRESETS["high_recall_v2"]
    assert p["r2_gate_mode"] == "adaptive_t"
    assert p["relative_cl_per_ion_stage"] is True
    assert p["high_recall"] is True


def test_consensus_voter_matches_v2_physics() -> None:
    """consensus_voter is v2 physics with pinned thresholds (jbfg.2 / n3rf.4)."""
    p = ALIAS_PRESETS["consensus_voter"]
    assert p["r2_gate_mode"] == "adaptive_t"
    assert p["relative_cl_per_ion_stage"] is True
    assert p["high_recall"] is False


@pytest.mark.parametrize("name", sorted(_EXPECTED_NAMES))
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
    assert ident.chance_window_scale == pytest.approx(0.6)


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
