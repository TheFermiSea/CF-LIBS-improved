"""DED real-goal metrics on the complete DB: ratio precision + drift sensitivity.

The DED benchmark reports absolute wt% RMSE, but the actual DED goal is real-time
drift tracking on a known set (Ti/Al/V): what matters is (a) does recovered
composition TRACK a true drift (slope ~1, the drift sensitivity), and (b) are the
element RATIOS (Al/Ti, V/Ti) recovered precisely. This extracts those from the
existing Al-scan composition series.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from tests.benchmarks.ded_precision.benchmark_runner import run_composition_series

DB = "ASD_da/libs_production.db"
print("Running Ti-6Al-4V Al-scan (clean) on the complete DB for ratio/drift metrics...")
df = run_composition_series(DB, "Ti-6Al-4V", "Al", clean=True)

# pivot to per-composition-point: rows are (sample/point, element) with target_value (the
# scanned Al target), pred_wt, error. Group by the Al target value.
pts = sorted(df["target_value"].unique())
print(f"\ncomposition points (Al target wt%): {[round(p,2) for p in pts]}")

# --- (a) DRIFT SENSITIVITY: slope of recovered Al vs true Al across the scan ---
al = df[df["element"] == "Al"]
true_al = al["target_value"].to_numpy(dtype=float)
pred_al = al["pred_wt"].to_numpy(dtype=float)
if len(set(true_al)) > 1:
    slope, intercept = np.polyfit(true_al, pred_al, 1)
    ss_res = np.sum((pred_al - (slope * true_al + intercept)) ** 2)
    ss_tot = np.sum((pred_al - pred_al.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"\nDRIFT SENSITIVITY (recovered Al vs true Al): slope={slope:.3f} (ideal 1.0), "
          f"intercept={intercept:+.2f} wt%, R2={r2:.3f}")
    print("  -> slope~1 + high R2 = drift is tracked even if absolute is biased")

# --- (b) RATIO PRECISION: Al/Ti and V/Ti recovered vs true, per point ---
print("\nRATIO recovery (recovered vs true, per Al-target point):")
print(f"  {'Al_tgt':>7} {'Al/Ti true':>10} {'Al/Ti rec':>10} {'V/Ti true':>10} {'V/Ti rec':>10}")
albyti_err, vbyti_err = [], []
for p in pts:
    sub = df[df["target_value"] == p]
    def m(el, col):
        r = sub[sub["element"] == el]
        return float(r[col].mean()) if len(r) else float("nan")
    # reconstruct per-element truth from error = pred - true -> true = pred - error
    truth = {el: m(el, "pred_wt") - m(el, "error") for el in ("Ti", "Al", "V")}
    pred = {el: m(el, "pred_wt") for el in ("Ti", "Al", "V")}
    alti_t, alti_r = truth["Al"] / truth["Ti"], pred["Al"] / pred["Ti"]
    vti_t, vti_r = truth["V"] / truth["Ti"], pred["V"] / pred["Ti"]
    albyti_err.append(abs(alti_r - alti_t) / alti_t)
    vbyti_err.append(abs(vti_r - vti_t) / vti_t)
    print(f"  {p:7.2f} {alti_t:10.4f} {alti_r:10.4f} {vti_t:10.4f} {vti_r:10.4f}")

print(f"\nRATIO mean |rel error|: Al/Ti {100*np.mean(albyti_err):.1f}%, V/Ti {100*np.mean(vbyti_err):.1f}%")
print("(For DED drift tracking, ratio rel-error + drift slope matter more than absolute wt%.)")
