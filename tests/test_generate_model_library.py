"""CLI validation tests for scripts/generate_model_library.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_model_library.py"
pytestmark = pytest.mark.integration


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


def test_chunk_mode_generates_real_spectra(tmp_path: Path, temp_db: str):
    h5py = pytest.importorskip("h5py")

    result = _run(
        "chunk",
        "--chunk-id",
        "0",
        "--n-chunks",
        "1",
        "--output-dir",
        str(tmp_path),
        "--db-path",
        temp_db,
        "--elements",
        "Fe",
        "--n-spectra",
        "2",
        "--lambda-min",
        "370",
        "--lambda-max",
        "380",
        "--delta-lambda",
        "0.05",
    )

    assert result.returncode == 0, result.stdout + result.stderr

    chunk_file = tmp_path / "chunk_0000.h5"
    assert chunk_file.exists()

    with h5py.File(chunk_file, "r") as f:
        spectra = f["spectra"][:]
        compositions = f["compositions"][:]
        elements = [value.decode("utf-8") for value in f.attrs["elements"]]

    assert spectra.shape[0] == 2
    assert np.any(spectra > 0.0)
    assert compositions.shape == (2, 1)
    assert elements == ["Fe"]


def test_build_index_mode_creates_artifacts(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("faiss")

    library_file = tmp_path / "model_library.h5"
    spectra = np.array(
        [
            [1.0, 0.0, 0.0, 0.5],
            [0.9, 0.1, 0.0, 0.4],
            [0.0, 1.0, 0.2, 0.0],
            [0.1, 0.8, 0.3, 0.1],
        ],
        dtype=np.float32,
    )
    with h5py.File(library_file, "w") as f:
        f.create_dataset("spectra", data=spectra)

    result = _run(
        "build-index",
        "--output-dir",
        str(tmp_path),
        "--n-components",
        "2",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "model_library.index.h5").exists()
    assert (tmp_path / "model_library.embedder.npz").exists()