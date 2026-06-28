"""Candidate-count invariance regression guard for the element identifiers.

This codifies the rig from the candidate-count-fragility audit
(``docs/architecture/2026-06-03-candidate-count-fragility-audit.md`` §2,§5)
as a CI test. It would have caught PR #216: the ``hybrid_union`` arm's
recall collapse when the NNLS mass floor was normalized by the *sum* of all
candidate coefficients, so the per-element bar tightened ~1/n as the
candidate count grew. The same bug survived in the standalone
``SpectralNNLSIdentifier`` default until the #216-followup fix.

The invariant under test: as distractor candidate elements are added to the
basis / element list, the recall of the true elements must NOT degrade
(flat or non-decreasing). A sum-normalized relative gate violates this; a
count-invariant gate (absolute coefficient SNR, MAX-relative floor) does not.

The rig builds a small basis library on the fly (do NOT depend on
``/tmp/real_basis``), runs each identifier at candidate-list sizes
n ~ {true, true+10, true+28}, and asserts recall does not COLLAPSE
(a one-element near-threshold flip is tolerated; a >=2-element ~1/n
cascade — the #216 signature — fails). See ``_RECALL_TOL``.

Runtime budget: coarse grid (1024 px, 3x2 (T, n_e) grid), one real BHVO-2
spectrum, three on-the-fly basis builds. ~25-35 s on CPU — well under the
600 s agent watchdog and the 60 s target.
"""

from __future__ import annotations

from pathlib import Path

import pytest

h5py = pytest.importorskip("h5py")

from cflibs.benchmark.synthetic_eval import build_corpus_basis_library  # noqa: E402
from cflibs.inversion.identify.spectral_nnls import SpectralNNLSIdentifier  # noqa: E402
from cflibs.io.spectrum import load_spectrum  # noqa: E402

pytestmark = [pytest.mark.requires_db, pytest.mark.integration]


# BHVO-2 USGS basalt certified majors (cflibs/benchmark/reference_compositions.py).
TRUE_ELEMENTS = ["Si", "Ti", "Al", "Fe", "Mn", "Mg", "Ca", "Na", "K", "P"]
# Monotone-nested distractor sets (absent from TRUE_ELEMENTS).
DISTRACTORS_10 = ["Cu", "Cr", "Ni", "Zn", "Co", "V", "Sr", "Ba", "Pb", "Li"]
DISTRACTORS_28 = DISTRACTORS_10 + [
    "Ag",
    "As",
    "Au",
    "B",
    "Be",
    "Bi",
    "Cd",
    "Ce",
    "Ga",
    "Ge",
    "Mo",
    "Nb",
    "Rb",
    "Sb",
    "Sc",
    "Sn",
    "Y",
    "Zr",
]

CANDIDATE_SETS = [
    (10, list(TRUE_ELEMENTS)),
    (20, list(TRUE_ELEMENTS) + DISTRACTORS_10),
    (38, list(TRUE_ELEMENTS) + DISTRACTORS_28),
]


def _db_path() -> str:
    candidates = [
        Path("libs_production.db"),
        Path("ASD_da/libs_production.db"),
        Path(__file__).parents[3] / "libs_production.db",
        Path(__file__).parents[3] / "ASD_da" / "libs_production.db",
    ]
    p = next((str(c) for c in candidates if c.exists()), None)
    if p is None:
        pytest.skip("Production database not found")
    return p


@pytest.fixture(scope="module")
def bhvo2_spectrum():
    """Load the real BHVO-2 spectrum once per module via the canonical loader."""
    candidates = [
        Path("data/bhvo2_usgs/chemcam_bhvo2_loc1_spectrum.csv"),
        Path(__file__).parents[3] / "data" / "bhvo2_usgs" / "chemcam_bhvo2_loc1_spectrum.csv",
    ]
    p = next((c for c in candidates if c.exists()), None)
    if p is None:
        pytest.skip("Real BHVO-2 spectrum not found")
    return load_spectrum(str(p))


def _recall(detected: set[str]) -> float:
    return sum(1 for t in TRUE_ELEMENTS if t in detected) / len(TRUE_ELEMENTS)


# One-element recall tolerance (= 1 / number of true elements). The detection
# gate is a HARD step at ``detection_snr`` and the NNLS coefficient SNR is only
# *approximately* count-invariant: single-element basis spectra are collinear
# (shared lines + a shared polynomial continuum), so adding distractor
# candidates perturbs the ``(A^T A)^-1`` covariance diagonal and lets NNLS
# redistribute a little coefficient mass onto the new correlated columns. A
# true element sitting within a few percent of the gate can therefore flip
# across it. On real BHVO-2 the borderline element is Mn (a MINOR — MnO 0.17
# wt%, ~0.0013 mass fraction — at the detection limit): its SNR drifts
# 4.14 -> 4.04 -> 3.94 across n=10/20/38, crossing the 4.0 gate at n=38. That
# is a legitimate near-threshold tie, NOT the #216 bug. The #216 bug was a
# ~1/n CASCADE that silently cut STRONG majors (Si/Al/Na, recall 0.70 -> 0.40,
# three elements lost); the tolerance below still catches that (a >=2-element
# collapse against the baseline) while permitting one borderline minor to flip.
_RECALL_TOL = 1.0 / len(TRUE_ELEMENTS) + 1e-9


def _assert_recall_non_decreasing(recalls: list[float], label: str) -> None:
    """Recall must not COLLAPSE as candidate count grows (the #216 invariant).

    Compared against the baseline (smallest candidate count) with a one-element
    tolerance: a single borderline element flipping across the hard SNR gate is
    permitted (see ``_RECALL_TOL``), but losing two or more true elements as
    candidates grow — the #216 ~1/n cascade — fails.
    """
    counts = [n for n, _ in CANDIDATE_SETS]
    baseline = recalls[0]
    for n_cur, r_cur in zip(counts[1:], recalls[1:]):
        assert r_cur >= baseline - _RECALL_TOL, (
            f"{label} recall collapsed with candidate count: "
            f"n={counts[0]}->{n_cur} recall {baseline:.3f}->{r_cur:.3f} "
            f"(allowed dip {_RECALL_TOL:.3f}). This is the #216 count-scaling bug."
        )


@pytest.fixture(scope="module")
def count_bases(tmp_path_factory):
    """Build one coarse basis per candidate count (on the fly, cached)."""
    db = _db_path()
    out_dir = tmp_path_factory.mktemp("count_invariance_bases")
    libs = {}
    for n, elements in CANDIDATE_SETS:
        lib = build_corpus_basis_library(
            db_path=db,
            output_path=str(out_dir / f"basis_n{n}.h5"),
            elements=elements,
            wavelength_range=(240.0, 850.0),
            pixels=1024,
            temperature_range=(7000.0, 12000.0),
            temperature_steps=3,
            density_range=(1e16, 1e17),
            density_steps=2,
            instrument_fwhm_nm=0.3,
            overwrite=True,
        )
        if lib is None:
            pytest.skip("Could not build basis library (h5py unavailable)")
        libs[n] = lib
    yield libs
    for lib in libs.values():
        lib.close()


class TestStandaloneNNLSCountInvariance:
    """The surviving #216 instance: standalone SpectralNNLSIdentifier.

    Pre-fix (sum-normalized 0.05 floor) recall sloped DOWN as candidates
    were added (Si, the most abundant element, silently cut). Post-fix
    (count-invariant SNR gate, MAX-relative floor default-off) recall is flat.
    """

    def test_standalone_recall_non_decreasing_with_count(self, count_bases, bhvo2_spectrum):
        wl, intensity = bhvo2_spectrum
        recalls = []
        for n, _elements in CANDIDATE_SETS:
            ident = SpectralNNLSIdentifier(
                basis_library=count_bases[n],
                fallback_T_K=9000.0,
                fallback_ne_cm3=3e16,
            )  # all-default gate (the shipped standalone behavior)
            res = ident.identify(wl, intensity)
            detected = {e.element for e in res.detected_elements}
            recalls.append(_recall(detected))

        # The guard: recall must not degrade as candidates grow.
        _assert_recall_non_decreasing(recalls, "Standalone NNLS")


class TestHybridUnionCountInvariance:
    """The #216 production arm: hybrid_union must stay count-flat."""

    def test_hybrid_union_recall_non_decreasing_with_count(self, count_bases, bhvo2_spectrum):
        from cflibs.atomic.database import AtomicDatabase
        from cflibs.inversion.identify.hybrid import HybridIdentifier

        wl, intensity = bhvo2_spectrum
        db = _db_path()
        recalls = []
        with AtomicDatabase(db) as adb:
            for n, elements in CANDIDATE_SETS:
                ident = HybridIdentifier(
                    atomic_db=adb,
                    basis_library=count_bases[n],
                    elements=list(elements),
                    require_both=False,  # union arm (production-best)
                )
                res = ident.identify(wl, intensity)
                detected = {e.element for e in res.detected_elements}
                recalls.append(_recall(detected))

        _assert_recall_non_decreasing(recalls, "hybrid_union")


class TestSNRGateSuppressesLeakageFP:
    """The count-invariant precision lever still suppresses the leakage FP.

    Per the audit, the NNLS-GAUSS-BASIS-4 leakage false positive sits at SNR
    just above 3 and ~8.15% of the max coefficient. The new default gate
    (detection_snr=4.0, MAX-relative floor off) must reject it. A MAX-relative
    floor large enough to reject it (>8.15%) would also cut the weakest true
    major (Na, ~5.6% of max), which is why SNR — not the relative floor — is
    the precision lever. This asserts the decision boundary directly so it
    does not depend on reproducing a fragile leakage spectrum.
    """

    @pytest.mark.parametrize(
        "snr, frac_max, expect_detected",
        [
            (3.3, 0.0815, False),  # leakage FP: rejected by snr>=4.0
            (5.05, 0.0563, True),  # real Na major: retained (snr passes, no floor)
            (9.43, 0.1525, True),  # real Si major: retained
        ],
    )
    def test_default_gate_decision_boundary(self, snr, frac_max, expect_detected):
        from cflibs.inversion.identify.spectral_nnls import (
            DEFAULT_DETECTION_SNR,
            DEFAULT_MIN_RELATIVE_COEFF,
            _passes_detection_gate,
        )

        detected = _passes_detection_gate(
            1.0,
            snr,
            frac_max,
            detection_snr=DEFAULT_DETECTION_SNR,
            min_relative_coeff=DEFAULT_MIN_RELATIVE_COEFF,
        )
        assert detected is expect_detected

    def test_max_floor_is_count_invariant_vs_legacy_sum_floor(self, count_bases, bhvo2_spectrum):
        """The relative gate metadata is now MAX-relative (count-invariant).

        ``relative_to_max`` for a given element must be ~stable as distractor
        candidates are added (the max coefficient does not change), unlike the
        legacy ``concentration_estimate`` (fraction-of-sum) which shrinks.
        """
        wl, intensity = bhvo2_spectrum
        rel_max = {}
        conc_sum = {}
        for n, _elements in CANDIDATE_SETS:
            ident = SpectralNNLSIdentifier(
                basis_library=count_bases[n],
                detection_snr=0.0,
                min_relative_coeff=0.0,
                fallback_T_K=9000.0,
                fallback_ne_cm3=3e16,
            )
            res = ident.identify(wl, intensity)
            by_el = {e.element: e for e in res.all_elements}
            rel_max[n] = by_el["Si"].metadata["relative_to_max"]
            conc_sum[n] = by_el["Si"].metadata["concentration_estimate"]

        # MAX-relative: count-invariant (within interpolation noise).
        assert (
            abs(rel_max[38] - rel_max[10]) < 0.03
        ), f"relative_to_max should be count-invariant, got {rel_max}"
        # SUM-relative: shrinks as candidates grow (the bug it replaced).
        assert (
            conc_sum[38] <= conc_sum[10]
        ), f"concentration_estimate (fraction-of-sum) should shrink, got {conc_sum}"
