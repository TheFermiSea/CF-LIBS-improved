"""Smoke tests for the NFS-shared benchmark data directory (T1.3).

These tests verify the canonical NFS path
(`/cluster/shared/cf-libs-bench/data/`) documented in
`docs/nfs-shared-data.md` is reachable and populated. They are designed
to be skipped cleanly on machines that aren't part of the cluster
(developer laptops, GitHub Actions runners, etc.) so they never break
local-dev workflows — but they do exercise the same env-var contract
(`CFLIBS_DATA_DIR`) that the production scripts honor.

Issue: `CF-LIBS-improved-mg58` (T1.3).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

CANONICAL_NFS_PATH = Path("/cluster/shared/cf-libs-bench/data")

# Datasets we expect to find under the canonical path. This list is
# intentionally small — adding more would couple the test to dataset
# refresh cadence. The two below are load-bearing for the headline
# benchmarks (Vrabel + BHVO-2) and have stable schemas.
REQUIRED_SUBDIRS = (
    "vrabel2020_soil_benchmark",
    "bhvo2_usgs",
)

# Files inside REQUIRED_SUBDIRS we expect to find. Like above, this is a
# minimal set — just enough to catch a half-populated mount.
REQUIRED_FILES = (
    "vrabel2020_soil_benchmark/train.h5",
    "vrabel2020_soil_benchmark/test.h5",
)


def _resolve_data_dir() -> Path:
    """Mirror the resolution logic used by `scripts/run_unified_benchmark.py`:

    Honor `CFLIBS_DATA_DIR` if set; otherwise fall back to the canonical
    NFS path.
    """
    env_override = os.environ.get("CFLIBS_DATA_DIR")
    if env_override:
        return Path(env_override)
    return CANONICAL_NFS_PATH


@pytest.fixture(scope="module")
def data_dir() -> Path:
    candidate = _resolve_data_dir()
    if not candidate.exists():
        pytest.skip(
            f"benchmark data dir not available at {candidate} — "
            "this test only runs on cluster nodes with the NFS mount or "
            "with CFLIBS_DATA_DIR pointed at a local copy. "
            "See docs/nfs-shared-data.md."
        )
    return candidate


def test_data_dir_is_a_directory(data_dir: Path) -> None:
    """The resolved data path must be a directory, not a file or symlink-to-nothing."""
    assert data_dir.is_dir(), f"{data_dir} exists but is not a directory"


def test_data_dir_is_non_empty(data_dir: Path) -> None:
    """A populated NFS export should contain at least one entry."""
    entries = list(data_dir.iterdir())
    assert entries, f"{data_dir} exists but is empty"


@pytest.mark.parametrize("subdir", REQUIRED_SUBDIRS)
def test_required_subdir_present(data_dir: Path, subdir: str) -> None:
    """Each headline-benchmark subdirectory must be visible."""
    target = data_dir / subdir
    assert target.is_dir(), (
        f"Required dataset subdirectory {target} is missing or not a "
        "directory; re-run the rsync from docs/nfs-shared-data.md."
    )


@pytest.mark.parametrize("rel_path", REQUIRED_FILES)
def test_required_file_present(data_dir: Path, rel_path: str) -> None:
    """Headline H5 files must be present and non-zero-sized."""
    target = data_dir / rel_path
    assert target.is_file(), f"Required file {target} is missing"
    assert target.stat().st_size > 0, f"Required file {target} is empty"


def test_canonical_path_is_documented() -> None:
    """`docs/nfs-shared-data.md` must reference the canonical path so the
    docs stay in lockstep with this test.
    """
    repo_root = Path(__file__).resolve().parents[2]
    doc = repo_root / "docs" / "nfs-shared-data.md"
    assert doc.is_file(), f"Expected documentation at {doc}"
    text = doc.read_text()
    assert "/cluster/shared/cf-libs-bench/data" in text, (
        f"{doc} does not reference the canonical NFS path; the test "
        "and docs have drifted."
    )
    assert "CFLIBS_DATA_DIR" in text, (
        f"{doc} does not document the CFLIBS_DATA_DIR override env var."
    )
