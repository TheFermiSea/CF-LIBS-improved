"""Run the real-steel composition benchmark for a given lever config (baseline by default)."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import warnings
warnings.filterwarnings("ignore")

from tests.benchmarks.real_steel.harness import run_benchmark  # noqa: E402
from tests.benchmarks.ded_precision.line_lists import select_lines  # noqa: E402
from tests.benchmarks.ded_precision.benchmark_runner import extract_line_intensities  # noqa: E402
from tests.benchmarks.ded_precision.solver_runner import run_constrained_solver, recovered_wt  # noqa: E402


def baseline_solve(db, wl, inten, truth):
    """Current constrained approach (the gate baseline)."""
    els = list(truth.keys())
    window = (float(wl.min()), float(wl.max()))
    specs = [s for e in els for s in select_lines(db, e, window, 8, prefer_spread=False)]
    obs = extract_line_intensities(wl, inten, specs, instrument_fwhm_nm=0.2)
    res = run_constrained_solver(db, obs, 1e17)
    return recovered_wt(res)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    r = run_benchmark(baseline_solve, limit=a.limit)
    print("BASELINE real-steel RMSEP (wt%):")
    for k in sorted(r):
        v = r[k]
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
