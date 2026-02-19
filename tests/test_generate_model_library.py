"""CLI validation tests for scripts/generate_model_library.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_model_library.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - defensive test guard
        pytest.fail(f"generate_model_library.py timed out: {exc}")


def test_chunk_mode_rejects_non_positive_n_chunks(tmp_path: Path):
    result = _run(
        "chunk",
        "--chunk-id",
        "0",
        "--n-chunks",
        "0",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 1
    assert "n_chunks must be >= 1" in (result.stdout + result.stderr)


def test_chunk_mode_rejects_out_of_range_chunk_id(tmp_path: Path):
    result = _run(
        "chunk",
        "--chunk-id",
        "5",
        "--n-chunks",
        "3",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 1
    assert "chunk_id must be in range [0, 3)" in (result.stdout + result.stderr)


def test_submit_mode_rejects_negative_max_concurrent(tmp_path: Path):
    pytest.importorskip("cflibs.hpc")

    result = _run(
        "submit",
        "--n-chunks",
        "8",
        "--output-dir",
        str(tmp_path),
        "--max-concurrent",
        "-1",
    )
    assert result.returncode == 1
    assert "max_concurrent must be >= 0" in (result.stdout + result.stderr)
