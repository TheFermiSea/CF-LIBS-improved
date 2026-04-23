"""Tests for the physics-grounding prompt primitives."""

from __future__ import annotations

import pytest

from cflibs.evolution.evaluator import FORBIDDEN_PREFIXES
from cflibs.evolution.prompts import (
    ALLOWED_PRIMITIVES,
    FORBIDDEN_LIBRARIES,
    ensure_consistency,
    render_preamble,
)


def test_forbidden_list_matches_scanner_blocklist() -> None:
    """The prompt's deny-list must stay in lockstep with the AST scanner's."""
    assert tuple(FORBIDDEN_LIBRARIES) == tuple(FORBIDDEN_PREFIXES)


def test_ensure_consistency_passes_in_tree() -> None:
    ensure_consistency()


def test_render_preamble_mentions_every_forbidden_library() -> None:
    text = render_preamble()
    for name in FORBIDDEN_PREFIXES:
        assert name in text, f"missing forbidden entry {name!r} in rendered preamble"


def test_render_preamble_mentions_allowed_primitives() -> None:
    text = render_preamble()
    for name in ALLOWED_PRIMITIVES:
        # Check only the module root (first word) to be robust to the
        # verbose descriptions attached to some entries.
        token = name.split()[0].rstrip(",")
        assert token in text, f"missing allowed entry {token!r} in rendered preamble"


def test_render_preamble_states_the_hard_constraint() -> None:
    text = render_preamble()
    assert "HARD CONSTRAINT" in text
    assert "physics-based only" in text
    assert "fitness = -inf" in text


def test_render_preamble_clarifies_ml_is_ok_in_optimization() -> None:
    text = render_preamble()
    assert "optimization process" in text
    assert "NEVER" in text


def test_ensure_consistency_raises_on_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate drift by pointing the prompt module's list at a different tuple.
    import cflibs.evolution.prompts as prompts_mod

    monkeypatch.setattr(prompts_mod, "FORBIDDEN_LIBRARIES", ("sklearn",))
    with pytest.raises(RuntimeError):
        prompts_mod.ensure_consistency()
