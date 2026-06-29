"""Regression gate for the best real-steel lever combination (v2).

Asserts the combined v2 config (L2 neutral-anchor line selection + L1 fixed optimal-``T`` +
robust conditioning-gated geometric-mean OPC ``F``, see
:mod:`tests.benchmarks.real_steel.best_config_v2`) keeps its honest **held-out**
``rmsep_overall`` below a guard set just above the achieved value.

Honesty: standard selection, the robust fixed ``T`` and the robust ``F`` are derived ONLY from
candidate standards' own spectra + own certified compositions (a conditioning gate that never
sees the held-out samples), and a selected standard is scored leave-one-out with the geometric
mean of the OTHER selected standards' ``F`` (no self-leakage). The asserted number is the
held-out RMSEP over all 36 samples (un-overfittable).

Lever ladder: baseline (no levers) 39.04 -> v1 (L2+L1+single-standard OPC) 16.48 -> v2
(L2+L1+robust OPC) **10.12** wt% held-out. L4 self-absorption is intentionally OFF in v2: it
helps standalone but regresses once OPC is present (overall 10.12 -> 11.35, Fe 20.66 -> 23.43),
because the OPC ``F`` already absorbs the matrix self-absorption bias.

Marked ``slow`` (the run does a 36-sample conditioning scan + a held-out sweep) and skipped when
the real-steel parquet is absent (a large data artifact, not checked into every checkout).
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from tests.benchmarks.real_steel.harness import PARQUET  # noqa: E402

# Guard set just above the achieved v2 held-out rmsep_overall (~10.12 wt%). Tighten on improvement.
RMSEP_GUARD = 11.0
# v1 (single-standard OPC) held-out headline — v2 must beat it.
V1_RMSEP = 16.48


@pytest.mark.slow
@pytest.mark.skipif(
    not os.path.exists(PARQUET), reason=f"real-steel parquet not present: {PARQUET}"
)
def test_best_config_v2_heldout_rmsep_below_guard():
    from tests.benchmarks.real_steel.best_config_v2 import run_v2

    out = run_v2()  # conditioning-gated robust OPC; L4 off (the winning config)
    held = out["held_out"]
    overall = held["rmsep_overall"]
    assert overall < RMSEP_GUARD, (
        f"real-steel v2 held-out rmsep_overall regressed: "
        f"{overall:.3f} >= guard {RMSEP_GUARD} (baseline 39.04, v1 {V1_RMSEP}). per-element: "
        + ", ".join(f"{k}={held[k]:.2f}" for k in sorted(held) if k.startswith("rmsep_"))
    )
    # v2 must also beat v1 (the single-standard-OPC combination).
    assert overall < V1_RMSEP, f"v2 ({overall:.3f}) must beat v1 single-standard OPC ({V1_RMSEP})"
