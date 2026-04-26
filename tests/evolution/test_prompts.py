"""Tests for the physics-grounding prompt primitives."""

from __future__ import annotations

import pytest

from cflibs.evolution.evaluator import FORBIDDEN_PREFIXES
from cflibs.evolution.prompts import ALLOWED_PRIMITIVES, render_preamble

pytestmark = pytest.mark.unit


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
