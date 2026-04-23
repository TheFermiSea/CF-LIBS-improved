"""Tests for the ``python -m cflibs.evolution`` CLI wrapper."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from cflibs.evolution.__main__ import main


def test_clean_file_exits_zero(tmp_path: Path) -> None:
    clean = tmp_path / "clean.py"
    clean.write_text("import numpy as np\nfrom jax import jit\n", encoding="utf-8")
    assert main([str(clean)]) == 0


def test_forbidden_file_exits_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("import torch\n", encoding="utf-8")
    rc = main([str(bad)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "torch" in captured.err
    assert "bad.py" in captured.err


def test_multiple_files_aggregate_violations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = tmp_path / "good.py"
    good.write_text("import numpy as np\n", encoding="utf-8")
    bad = tmp_path / "bad.py"
    bad.write_text("import sklearn\n", encoding="utf-8")
    rc = main([str(good), str(bad)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "sklearn" in captured.err
    assert "good.py" not in captured.err


def test_missing_file_is_violation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([str(tmp_path / "does_not_exist.py")])
    captured = capsys.readouterr()
    assert rc == 1
    assert "not found" in captured.err


def test_stdin_input(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("import torch\n"))
    rc = main(["-"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "<stdin>" in captured.err
    assert "torch" in captured.err


def test_syntax_error_is_violation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    broken = tmp_path / "broken.py"
    broken.write_text("def f(:\n", encoding="utf-8")
    rc = main([str(broken)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "syntax error" in captured.err
