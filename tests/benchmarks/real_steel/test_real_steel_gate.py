"""Regression gate for the best real-steel lever combination.

Asserts the combined config (L2 neutral-anchor line selection + L1 fixed optimal-T + L3
one-point-calibration ``F``, see :mod:`tests.benchmarks.real_steel.best_config`) keeps its
honest **held-out** ``rmsep_overall`` below a guard threshold set just above the achieved value.

Honesty: ``T*`` and ``F`` are derived from ONE matrix-matched standard and applied unchanged to
the 35 held-out samples; the asserted number is the held-out RMSEP (un-overfittable).

Baseline (no levers) is 39.04 wt%; the combination achieves ~16.5 wt% held-out. The guard is
``RMSEP_GUARD`` — a regression trips it, an improvement passes (tighten the guard when it lands).

Marked ``slow`` (the held-out sweep solves ~40 spectra) and skipped when the real-steel parquet
is absent (it is a large data artifact, not checked into every checkout).
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from tests.benchmarks.real_steel.harness import PARQUET  # noqa: E402

# A-priori-selected standard (lowest in-sample uncorrected-L2 RMSEP across the 36 samples; see
# best_config.select_standard_index). Pinned here so the gate is deterministic and fast — it
# skips the 36-solve selection scan and reproduces the headline held-out result.
STANDARD_INDEX = 18
# Guard set just above the achieved held-out rmsep_overall (~16.48 wt%). Tighten on improvement.
RMSEP_GUARD = 18.0


@pytest.mark.slow
@pytest.mark.skipif(
    not os.path.exists(PARQUET), reason=f"real-steel parquet not present: {PARQUET}"
)
def test_best_config_heldout_rmsep_below_guard():
    from tests.benchmarks.real_steel.best_config import run_best

    out = run_best(std_index=STANDARD_INDEX)
    held = out["held_out"]
    overall = held["rmsep_overall"]
    assert overall < RMSEP_GUARD, (
        f"real-steel best-combo held-out rmsep_overall regressed: "
        f"{overall:.3f} >= guard {RMSEP_GUARD} (baseline 39.04). per-element: "
        + ", ".join(f"{k}={held[k]:.2f}" for k in sorted(held) if k.startswith("rmsep_"))
    )
    # The combination must also beat every lever alone and the baseline.
    assert overall < 29.98, f"combined ({overall:.3f}) must beat best single lever (L1 29.98)"
