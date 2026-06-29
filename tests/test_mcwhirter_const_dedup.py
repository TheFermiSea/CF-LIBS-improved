"""Regression: McWhirter prefactor imported from core.constants, not inlined (M1-7).

physical_consistency.py's standalone copy is intentional (minimal-env importability)
and must remain a literal — guarded by the last test.
"""

import ast
from pathlib import Path

from cflibs.core.constants import MCWHIRTER_CONST

_ROOT = Path(__file__).parent.parent


def _literal_count(relpath, value=1.6e12):
    tree = ast.parse((_ROOT / relpath).read_text())
    return [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == value]


def test_line_selection_prefactor_equals_core_const():
    from cflibs.inversion.physics.line_selection import MCWHIRTER_PREFACTOR_CM3_K

    assert MCWHIRTER_PREFACTOR_CM3_K == MCWHIRTER_CONST


def test_temporal_uses_const_not_literal():
    src = (_ROOT / "cflibs/inversion/runtime/temporal.py").read_text()
    assert "MCWHIRTER_CONST" in src
    assert not _literal_count("cflibs/inversion/runtime/temporal.py")


def test_line_selection_no_bare_literal():
    assert not _literal_count("cflibs/inversion/physics/line_selection.py")


def test_physical_consistency_still_self_contained():
    # intentional standalone copy must remain
    assert _literal_count("cflibs/benchmark/physical_consistency.py")
