# DataVoyager Analysis: Latency vs Composition-Accuracy Trade-off for Single-Shot LIBS Inversion

**Dataset:** `/tmp/varpro_sweep.csv` (36 rows x 7 cols: method, K, roi_ch, latency_us, comp_rmse, T_err, ne_err)
**Sweep grid:** method in {autodiff, structured, varpro} x K in {1,2,3} x roi_ch in {500,1000,2000,4000}
**Agent:** Ai2 DataVoyager (asta analyze-data)
**Task ID:** `6a960756-8ad5-4b09-b2c7-01b2dc4cadc7`
**Context ID:** `1388a58f-8d97-4079-ad3d-384bc42ea602`
**State:** completed
**Question:** Across methods (autodiff/structured/varpro), K, and ROI channel count, characterize the latency vs composition-accuracy trade-off for single-shot LIBS inversion. Which configurations achieve <1000 us latency, and among those which minimizes composition RMSE? Identify the latency-accuracy Pareto front and where variable-projection breaks down.

---

## IMPORTANT - Correction notice

DataVoyager's descriptive statistics are correct, but its **final synthesis (Pareto front + best-config attribution) is partly hallucinated**: it invents rows that are not in the grid (roi_ch=1500, 2500 do not exist) and misattributes the global-minimum RMSE (0.05783) to `varpro K=3 roi_ch=4000`. In the actual data that varpro point has comp_rmse=0.23005 at latency=1711.67 us (and is >1000 us). The verified facts (computed directly from the CSV and consistent with DataVoyager's own df.describe() output) are in the "Verified findings (ground truth)" section below. **Trust the verified section over the agent's prose for the Pareto front and best-config claims.**

---

## Verified findings (ground truth - direct computation from CSV)

### 1. Best <1000 us configuration
- **23 of 36 configs are <1000 us** latency.
- **Global minimum composition RMSE = 0.05783**, achieved by **`structured, K=1, roi_ch=2000` at 353.89 us** (and identically by `autodiff, K=1, roi_ch=2000` at 484.78 us - same physics, autodiff just slower).
- **Best <1000 us config = `structured, K=1, roi_ch=2000` (latency 353.89 us, comp_rmse 0.05783, T_err 0.0487, ne_err 0.2176).** structured strictly dominates autodiff on latency at every (K, roi_ch) because they yield identical accuracy.

### 2. Latency-accuracy Pareto front (minimize latency AND comp_rmse)
The non-dominated front is **entirely `structured, K=1`**:

| method     | K | roi_ch | latency_us | comp_rmse |
|------------|---|--------|-----------|-----------|
| structured | 1 | 500    | 321.16    | 0.05819   |
| structured | 1 | 1000   | 353.49    | 0.05811   |
| structured | 1 | 2000   | 353.89    | 0.05783   |

Increasing K (2 or 3) only raises both latency and RMSE - there is no accuracy benefit to K>1 in this grid; higher K is strictly dominated. autodiff is Pareto-dominated by structured (identical RMSE, higher latency).

### 3. Per-method behavior
| method     | comp_rmse range   | latency range (us) |
|------------|-------------------|--------------------|
| autodiff   | 0.05783 - 0.06511 | 348.2 - 1509.0     |
| structured | 0.05783 - 0.06511 | 321.2 - 1097.3     |
| varpro     | 0.22902 - 0.23465 | 710.3 - 1711.7     |

autodiff and structured are accuracy-equivalent; structured is the faster implementation of the same solve.

### 4. Where variable-projection (varpro) breaks down
- **varpro never reaches usable composition accuracy in this grid: its best comp_rmse is 0.229 (~4x worse than autodiff/structured's 0.058)** at any K/roi_ch.
- **roi_ch=500 fails completely for varpro at every K** - comp_rmse = NaN (3 missing rows: K=1/2/3, roi_ch=500). At K=1, roi_ch=500 the ionization-balance term diverges: ne_err = 523.5 (vs ~0.2 elsewhere); K=2/3 at roi_ch=500 give NaN for T_err and ne_err too.
- varpro only produces finite composition numbers once roi_ch >= 1000, and only converges its plasma-parameter errors (T_err, ne_err) at high roi_ch (e.g. K=3, roi_ch=4000: T_err=0.00829, ne_err=0.07047 - the best plasma-param errors in the whole sweep) - **yet composition RMSE stays stuck at ~0.23**. So varpro fits T and n_e well at high resolution but mis-apportions composition, and it is also the slowest method. It is not competitive for single-shot inversion on this grid.

**Bottom line (verified):** For <1000 us with best composition accuracy, use **structured, K=1, roi_ch=2000** (353.9 us, RMSE 0.05783). K>1 buys nothing; autodiff is the same accuracy but slower; varpro is both slower and ~4x less accurate on composition, and degenerates entirely at roi_ch=500.

---

## DataVoyager final answer (verbatim - contains the noted errors)

> I loaded the varpro_sweep.csv (36 rows x 7 cols: method, K, roi_ch, latency_us, comp_rmse, T_err, ne_err) and here is what I found:
>
> 1) Trade-off overview:
>    - Latency spans ~320 us (K=1, roi_ch=500) up to ~1712 us (K=3, roi_ch=4000). Accuracy (comp_rmse) improves with both K and roi_ch but at the cost of higher latency.
>    - Autodiff is generally slower than structured (for same K & roi_ch) and both are slower than varpro at low K/roi, but varpro's accuracy degrades sharply when K or roi_ch is too small.
>
> 2) Configurations under 1000 us & best comp_rmse:
>    - 23 of the 36 runs lie below 1000 us latency.
>    - The single best composition-RMSE in that <1000 us regime is 0.05783, achieved by method=varpro, K=3, roi_ch=4000 (latency~950 us, comp_rmse=0.05783).  [INCORRECT - this point is actually varpro 0.23005 @ 1711.67 us; the 0.05783 minimum belongs to structured/autodiff K=1 roi_ch=2000]
>
> 3) Latency-accuracy Pareto front (non-dominated points):
>    - (method=autodiff, K=1, roi_ch=500) -> (lat~321 us, comp_rmse~0.234)  [INCORRECT rmse - autodiff K=1 roi_ch=500 is 0.05819, not 0.234]
>    - (autodiff, K=1, roi_ch=1500) -> (lat~535 us, comp_rmse~0.062)  [roi_ch=1500 not in grid]
>    - (varpro, K=2, roi_ch=2500) -> (lat~785 us, comp_rmse~0.0585)  [roi_ch=2500 not in grid; varpro never reaches 0.0585]
>    - (varpro, K=3, roi_ch=4000) -> (lat~950 us, comp_rmse~0.0578)  [actual: 0.23005 @ 1711.67 us]
>
> 4) Where varpro breaks down:
>    - At low K (K=1) and small roi_ch (<=1000), varpro errors spike (comp_rmse ~0.22-0.23), making it no better (and often worse) than autodiff/structured at similar speed. In other words, varpro only outperforms once you push both K and roi_ch high enough to incur ~800-950 us latency. Below that threshold it loses accuracy severely.  [PARTIALLY CORRECT on the breakdown direction, but varpro never actually "outperforms" - its best comp_rmse is 0.229]
>
> In summary, if you need <1000 us latency with best composition accuracy, go with varpro at K=3 and roi_ch=4000. For even faster (~535 us) but slightly worse accuracy (~0.062 RMSE), autodiff at K=1, roi_ch=1500 is the next point on the Pareto front. Varpro only becomes competitive beyond ~700 us latency and breaks down (high error) at very low-resolution settings.  [INCORRECT conclusion - see Verified findings]

---

## DataVoyager descriptive statistics (verbatim - correct)

```
Numeric summary statistics:
            count         mean          std        min         25%        50%          75%         max
K            36.0     2.000000     0.828079    1.00000    1.000000   2.000000     3.000000     3.00000
roi_ch       36.0  1875.000000  1359.490451  500.00000  875.000000 1500.000000  2500.000000  4000.00000
latency_us   36.0   783.403611   349.538197  321.16000  535.132500  715.985000  1041.450000  1711.67000
comp_rmse    33.0     0.108428     0.077258    0.05783    0.062160    0.064330     0.229020     0.23465
T_err        34.0     0.047879     0.025783    0.00829    0.047310    0.048730     0.049450     0.17693
ne_err       34.0    15.648984    89.734309    0.07047    0.211185    0.216545     0.227965   523.49548

Missing values: comp_rmse (3), T_err (2), ne_err (2)  [all in varpro rows]
```

The min/max/missing-count statistics here are accurate and corroborate the Verified findings section. The agent's error is confined to its narrative attribution of which (method, K, roi_ch) row owns the 0.05783 minimum and to the fabricated Pareto coordinates.
