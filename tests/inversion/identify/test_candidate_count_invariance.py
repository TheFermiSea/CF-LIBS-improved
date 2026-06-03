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
n ~ {true, true+10, true+28}, and asserts non-decreasing recall.

Runtime budget: coarse grid (1024 px, 3x2 (T, n_e) grid), one real BHVO-2
spectrum, three on-the-fly basis builds. ~25-35 s on CPU — well under the
600 s agent watchdog and the 60 s target.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from cflibs.benchmark.synthetic_eval import build_corpus_basis_library  # noqa: E402
from cflibs.inversion.identify.spectral_nnls import SpectralNNLSIdentifier  # noqa: E402

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


def _bhvo2_spectrum():
    candidates = [
        Path("data/bhvo2_usgs/chemcam_bhvo2_loc1_spectrum.csv"),
        Path(__file__).parents[3] / "data" / "bhvo2_usgs" / "chemcam_bhvo2_loc1_spectrum.csv",
    ]
    p = next((c for c in candidates if c.exists()), None)
    if p is None:
        pytest.skip("Real BHVO-2 spectrum not found")
    data = np.loadtxt(str(p), delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def _recall(detected: set[str]) -> float:
    return sum(1 for t in TRUE_ELEMENTS if t in detected) / len(TRUE_ELEMENTS)


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

    def test_standalone_recall_non_decreasing_with_count(self, count_bases):
        wl, intensity = _bhvo2_spectrum()
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
        counts = [n for n, _ in CANDIDATE_SETS]
        series = list(zip(counts, recalls))
        for (n_prev, r_prev), (n_cur, r_cur) in zip(series, series[1:]):
            assert r_cur >= r_prev - 1e-9, (
                f"Standalone NNLS recall degraded with candidate count: "
                f"n={n_prev}->{n_cur} recall {r_prev:.3f}->{r_cur:.3f}. "
                f"This is the #216 count-scaling bug."
            )

    def test_sum_normalized_floor_mechanism_is_count_scaling(self, count_bases):
        """Demonstrate the #216 mechanism the fix removes, grid-independently.

        The legacy #215 gate compared the fraction-of-SUM
        (``concentration_estimate = coeff / sum(all_candidate_coeffs)``)
        against a fixed 0.05 bar. Because the denominator grows with the
        candidate count while a true element's own coefficient is ~constant,
        that fraction STRICTLY SHRINKS as distractors are added — so a fixed
        bar tightens ~1/n and eventually drops the element. That is the
        count-scaling bug; whether it crosses 0.05 at a given n is grid-
        dependent, but the monotone shrink is the invariant defect.

        We assert the shrink on the most-abundant true element (Si). If this
        ever stops shrinking, the sum-normalized denominator is gone and the
        rig has lost the #216 mechanism it is meant to pin.
        """
        wl, intensity = _bhvo2_spectrum()
        si_conc_sum = []
        si_snr = []
        for n, _elements in CANDIDATE_SETS:
            ident = SpectralNNLSIdentifier(
                basis_library=count_bases[n],
                detection_snr=0.0,
                min_relative_coeff=0.0,
                fallback_T_K=9000.0,
                fallback_ne_cm3=3e16,
            )
            res = ident.identify(wl, intensity)
            si = next(e for e in res.all_elements if e.element == "Si")
            si_conc_sum.append(si.metadata["concentration_estimate"])
            si_snr.append(si.metadata["nnls_snr"])

        # Fraction-of-sum strictly shrinks as candidates grow (the bug).
        assert si_conc_sum[0] > si_conc_sum[1] > si_conc_sum[2], (
            "Expected Si fraction-of-sum to shrink monotonically with "
            f"candidate count (the #216 mechanism); got {si_conc_sum}."
        )
        # ...while Si's own SNR is essentially constant (count-invariant) —
        # so the recall loss is an artifact of the denominator, not the data.
        assert max(si_snr) - min(si_snr) < 1.0, f"Si SNR should be count-invariant; got {si_snr}."


class TestHybridUnionCountInvariance:
    """The #216 production arm: hybrid_union must stay count-flat."""

    def test_hybrid_union_recall_non_decreasing_with_count(self, count_bases):
        from cflibs.atomic.database import AtomicDatabase
        from cflibs.inversion.identify.hybrid import HybridIdentifier

        wl, intensity = _bhvo2_spectrum()
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

        counts = [n for n, _ in CANDIDATE_SETS]
        for (n_prev, r_prev), (n_cur, r_cur) in zip(
            list(zip(counts, recalls)), list(zip(counts, recalls))[1:]
        ):
            assert r_cur >= r_prev - 1e-9, (
                f"hybrid_union recall degraded with candidate count: "
                f"n={n_prev}->{n_cur} recall {r_prev:.3f}->{r_cur:.3f}. "
                f"This is the #216 count-scaling bug."
            )


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
        )

        # Reproduce the shipped default gate logic exactly.
        coeff = 1.0  # positive
        detected = (
            coeff > 1e-10
            and snr >= DEFAULT_DETECTION_SNR
            and frac_max >= DEFAULT_MIN_RELATIVE_COEFF
        )
        assert detected is expect_detected

    def test_max_floor_is_count_invariant_vs_legacy_sum_floor(self, count_bases):
        """The relative gate metadata is now MAX-relative (count-invariant).

        ``relative_to_max`` for a given element must be ~stable as distractor
        candidates are added (the max coefficient does not change), unlike the
        legacy ``concentration_estimate`` (fraction-of-sum) which shrinks.
        """
        wl, intensity = _bhvo2_spectrum()
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
