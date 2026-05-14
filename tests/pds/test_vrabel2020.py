"""Unit tests for cflibs.pds.vrabel2020.

Synthetic HDF5 fixtures (built per-test) keep the suite hermetic — no
dependence on the 10.75 GB real dataset. The smoke test at the end
opens the real files only if they're present on disk, so the suite
also acts as an integration check when run on a workstation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
openpyxl = pytest.importorskip("openpyxl")

# Import placed after both importorskip calls so module collection
# short-circuits cleanly when either optional dep is missing.
from cflibs.pds.vrabel2020 import (  # noqa: E402
    ELEMENTS,
    N_CHANNELS,
    N_CLASSES,
    N_TEST_SPECTRA,
    SHOTS_PER_TRAIN_SAMPLE,
    VrabelSampleComposition,
    VrabelTestSplit,
    VrabelTrainSplit,
    load_compositions,
    load_test,
    load_test_iter,
    load_test_labels,
    load_train,
    load_wavelengths,
)


# ─── Fixture builders ────────────────────────────────────────────────────
def _make_train_h5(path: Path, *, n_samples: int = 3, shots: int = 4) -> None:
    """Build a tiny synthetic train.h5 with `n_samples` × `shots` spectra.

    Schema mirrors the real file exactly so loaders touch the same paths.
    """
    with h5py.File(path, "w") as f:
        # Wavelengths/1: shape (1, N_CHANNELS), float64
        wl = np.linspace(200.0, 800.0, N_CHANNELS).reshape(1, -1)
        f.create_group("Wavelengths").create_dataset("1", data=wl)

        # Spectra/{001..00N}: each (N_CHANNELS, SHOTS_PER_TRAIN_SAMPLE)
        spectra_grp = f.create_group("Spectra")
        for sid in range(1, n_samples + 1):
            data = np.full(
                (N_CHANNELS, SHOTS_PER_TRAIN_SAMPLE),
                fill_value=float(sid),
                dtype=np.float64,
            )
            spectra_grp.create_dataset(f"{sid:03d}", data=data)

        # Class/1 holds n_samples * SHOTS_PER_TRAIN_SAMPLE entries — one
        # per spectrum slot, even unused ones. The class for sample i is
        # ((i - 1) % N_CLASSES) + 1.
        class_data = np.empty(
            n_samples * SHOTS_PER_TRAIN_SAMPLE, dtype=np.int32,
        )
        for i in range(n_samples):
            cls = (i % N_CLASSES) + 1
            class_data[i * SHOTS_PER_TRAIN_SAMPLE :
                       (i + 1) * SHOTS_PER_TRAIN_SAMPLE] = cls
        f.create_group("Class").create_dataset("1", data=class_data)


def _make_test_h5(path: Path, *, chunks: int = 2, per_chunk: int = 5) -> None:
    """Build a tiny synthetic test.h5."""
    with h5py.File(path, "w") as f:
        wl = np.linspace(200.0, 800.0, N_CHANNELS).reshape(1, -1)
        f.create_group("Wavelengths").create_dataset("1", data=wl)
        unk = f.create_group("UNKNOWN")
        for i in range(1, chunks + 1):
            data = np.full(
                (N_CHANNELS, per_chunk),
                fill_value=float(10 + i),
                dtype=np.float64,
            )
            unk.create_dataset(str(i), data=data)


def _make_support_tables(path: Path, *, n_samples: int = 5) -> None:
    """Build a tiny synthetic support_tables.xlsx."""
    wb = openpyxl.Workbook()
    # default sheet → MIXED_composition
    ws_comp = wb.active
    ws_comp.title = "MIXED_composition"
    header = ["Sample ID", "Class ID"] + list(ELEMENTS)
    ws_comp.append(header)
    for sid in range(1, n_samples + 1):
        cls = ((sid - 1) % N_CLASSES) + 1
        # composition: each element gets value sid * 0.1
        vals = [sid, cls] + [sid * 0.1 for _ in ELEMENTS]
        ws_comp.append(vals)

    ws_unc = wb.create_sheet("MIXED_uncertainty")
    ws_unc.append(header)
    for sid in range(1, n_samples + 1):
        cls = ((sid - 1) % N_CLASSES) + 1
        vals = [sid, cls] + [sid * 0.01 for _ in ELEMENTS]
        ws_unc.append(vals)

    # Add an unrelated sheet so the loader filters by name (regression).
    ws_other = wb.create_sheet("OREAS")
    ws_other.append(["Sample*", "ignored"])

    wb.save(path)


# ─── Tests ───────────────────────────────────────────────────────────────
def test_load_wavelengths(tmp_path: Path) -> None:
    p = tmp_path / "train.h5"
    _make_train_h5(p)
    wl = load_wavelengths(p)
    assert wl.shape == (N_CHANNELS,)
    assert wl[0] == pytest.approx(200.0)
    assert wl[-1] == pytest.approx(800.0)


def test_load_train_full_shots(tmp_path: Path) -> None:
    p = tmp_path / "train.h5"
    _make_train_h5(p, n_samples=3)

    split = load_train(p)
    assert isinstance(split, VrabelTrainSplit)
    assert split.spectra.shape == (3 * SHOTS_PER_TRAIN_SAMPLE, N_CHANNELS)
    assert split.wavelengths.shape == (N_CHANNELS,)
    assert split.sample_ids.shape == (3 * SHOTS_PER_TRAIN_SAMPLE,)
    assert split.class_ids.shape == (3 * SHOTS_PER_TRAIN_SAMPLE,)

    # Per fixture: sample i has all-`i` values
    for sid in (1, 2, 3):
        mask = split.sample_ids == sid
        assert split.spectra[mask].shape[0] == SHOTS_PER_TRAIN_SAMPLE
        assert np.all(split.spectra[mask] == sid)


def test_load_train_thinned_shots(tmp_path: Path) -> None:
    """shots_per_sample < 500 should subsample row count linearly."""
    p = tmp_path / "train.h5"
    _make_train_h5(p, n_samples=2)

    split = load_train(p, shots_per_sample=10)
    assert split.spectra.shape == (2 * 10, N_CHANNELS)
    assert split.sample_ids.tolist() == [1] * 10 + [2] * 10


def test_load_train_invalid_shots(tmp_path: Path) -> None:
    p = tmp_path / "train.h5"
    _make_train_h5(p, n_samples=1)
    with pytest.raises(ValueError, match="shots_per_sample"):
        load_train(p, shots_per_sample=0)
    with pytest.raises(ValueError, match="shots_per_sample"):
        load_train(p, shots_per_sample=999)


def test_load_test_full(tmp_path: Path) -> None:
    p = tmp_path / "test.h5"
    _make_test_h5(p, chunks=3, per_chunk=4)
    split = load_test(p)
    assert isinstance(split, VrabelTestSplit)
    assert split.spectra.shape == (3 * 4, N_CHANNELS)
    # chunk 1 → 11.0, chunk 2 → 12.0, chunk 3 → 13.0 per fixture
    assert np.all(split.spectra[0] == 11.0)
    assert np.all(split.spectra[4] == 12.0)
    assert np.all(split.spectra[8] == 13.0)


def test_load_test_iter_yields_in_order(tmp_path: Path) -> None:
    p = tmp_path / "test.h5"
    _make_test_h5(p, chunks=2, per_chunk=3)

    seen_offsets: list[int] = []
    total_rows = 0
    for offset, chunk in load_test_iter(p, chunk_size=2):
        seen_offsets.append(offset)
        total_rows += chunk.shape[0]
    assert total_rows == 6
    # Offsets should be monotonically non-decreasing
    assert seen_offsets == sorted(seen_offsets)


def test_load_test_labels(tmp_path: Path) -> None:
    p = tmp_path / "test_labels.csv"
    p.write_text("\n".join(["1", "5", "12", "1", "8"]))
    labels = load_test_labels(p)
    assert labels.tolist() == [1, 5, 12, 1, 8]
    assert labels.dtype == np.int32


def test_load_test_labels_rejects_out_of_range(tmp_path: Path) -> None:
    p = tmp_path / "test_labels.csv"
    p.write_text("1\n13\n")  # 13 is > N_CLASSES
    with pytest.raises(ValueError, match=r"outside \[1, 12\]"):
        load_test_labels(p)


def test_load_compositions(tmp_path: Path) -> None:
    p = tmp_path / "support_tables.xlsx"
    _make_support_tables(p, n_samples=4)
    samples = load_compositions(p)
    assert len(samples) == 4
    assert isinstance(samples[1], VrabelSampleComposition)
    assert samples[3].sample_id == 3
    assert samples[3].class_id == 3
    # Per fixture: composition[Fe] for sample 3 = 3 * 0.1 = 0.3
    assert samples[3].composition["Fe"] == pytest.approx(0.3)
    assert samples[3].uncertainty["Fe"] == pytest.approx(0.03)
    # All elements present
    assert set(samples[1].composition.keys()) == set(ELEMENTS)


def test_load_compositions_missing_sheet_raises(tmp_path: Path) -> None:
    """If neither MIXED_composition nor MIXED_uncertainty are present,
    the loader must raise a clear ValueError rather than KeyError."""
    p = tmp_path / "broken.xlsx"
    wb = openpyxl.Workbook()
    wb.active.title = "OREAS"
    wb.save(p)
    with pytest.raises(ValueError, match="MIXED_composition"):
        load_compositions(p)


# ─── Smoke test against the real dataset (skipped when files absent) ─────
_REAL_ROOT = Path(__file__).resolve().parents[2] / "data" / "vrabel2020_soil_benchmark"


@pytest.mark.skipif(
    not (_REAL_ROOT / "test.h5").exists(),
    reason="real test.h5 not present (10 GB dataset; skip on CI)",
)
def test_real_test_dataset_smoke() -> None:
    """When the real test.h5 is on disk, verify schema + label alignment.

    Asserts the structural contract that downstream benchmarks rely on:
    20,000 spectra of N_CHANNELS each, labels file is the same length.
    """
    test_h5 = _REAL_ROOT / "test.h5"
    labels_csv = _REAL_ROOT / "test_labels.csv"
    assert labels_csv.exists(), "test_labels.csv must accompany test.h5"

    # Use streaming to avoid loading 6 GB
    total = 0
    for _, chunk in load_test_iter(test_h5, chunk_size=2000):
        total += chunk.shape[0]
        assert chunk.shape[1] == N_CHANNELS
    assert total == N_TEST_SPECTRA

    labels = load_test_labels(labels_csv)
    assert labels.size == N_TEST_SPECTRA
    assert set(labels.tolist()).issubset(set(range(1, N_CLASSES + 1)))


@pytest.mark.skipif(
    not (_REAL_ROOT / "support_tables.xlsx").exists(),
    reason="real support_tables.xlsx not present",
)
def test_real_support_tables_smoke() -> None:
    """Real composition file: should have ≥130 samples, all elements
    present in at least one row, all values finite."""
    samples = load_compositions(_REAL_ROOT / "support_tables.xlsx")
    assert len(samples) >= 130, (
        f"expected ≥130 samples, got {len(samples)}"
    )
    # Every element must be present in at least one sample's keys
    seen_elements: set[str] = set()
    for s in samples.values():
        seen_elements.update(s.composition.keys())
    assert set(ELEMENTS).issubset(seen_elements), (
        f"missing elements: {set(ELEMENTS) - seen_elements}"
    )
    # All composition values are finite floats
    for s in samples.values():
        for elt, v in s.composition.items():
            assert np.isfinite(v), f"sample {s.sample_id} {elt} is not finite"
            assert v >= 0.0, f"sample {s.sample_id} {elt} is negative"
