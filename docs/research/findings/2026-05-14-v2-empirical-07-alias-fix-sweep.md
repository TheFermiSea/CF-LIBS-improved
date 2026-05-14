# Empirical 07: alias-fix sweep (Phase C of jaunty-weaving-mist)

- Cells: 8 (2³ combinations of ftp1/762f/dj6y)
- Seeds per cell: 5
- Dataset shard: `1/3`
- Promotion target: `vrabel2020_soil_benchmark`
- Promotion guards: 'aalto_libs', 'bhvo2_usgs' (no metric regression > 0.02)

## Overall metrics (mean ± stdev across seeds)

| cell | macro_f1 | macro_precision | macro_recall | false_positives_per_spectrum |
|---|---|---|---|---|
| baseline | 0.1111 ± 0.0000 | 0.1818 ± 0.0000 | 0.1023 ± 0.0000 | 0.0909 ± 0.0000 |
| ftp1 | 0.2819 ± 0.0000 | 0.4318 ± 0.0000 | 0.2321 ± 0.0000 | 0.3636 ± 0.0000 |
| 762f | 0.1111 ± 0.0000 | 0.1818 ± 0.0000 | 0.1023 ± 0.0000 | 0.0909 ± 0.0000 |
| dj6y | 0.1111 ± 0.0000 | 0.1818 ± 0.0000 | 0.1023 ± 0.0000 | 0.0909 ± 0.0000 |
| ftp1+762f | 0.1111 ± 0.0000 | 0.1818 ± 0.0000 | 0.1023 ± 0.0000 | 0.0909 ± 0.0000 |
| ftp1+dj6y | 0.3092 ± 0.0000 | 0.4318 ± 0.0000 | 0.2700 ± 0.0000 | 0.4545 ± 0.0000 |
| 762f+dj6y | 0.1111 ± 0.0000 | 0.1818 ± 0.0000 | 0.1023 ± 0.0000 | 0.0909 ± 0.0000 |
| all_three | 0.1111 ± 0.0000 | 0.1818 ± 0.0000 | 0.1023 ± 0.0000 | 0.0909 ± 0.0000 |

## Per-dataset macro_f1 (mean ± stdev)

| cell | vrabel2020_soil_benchmark | aalto_libs | bhvo2_usgs |
|---|---|---|---|
| baseline | n/a | n/a | n/a |
| ftp1 | n/a | n/a | n/a |
| 762f | n/a | n/a | n/a |
| dj6y | n/a | n/a | n/a |
| ftp1+762f | n/a | n/a | n/a |
| ftp1+dj6y | n/a | n/a | n/a |
| 762f+dj6y | n/a | n/a | n/a |
| all_three | n/a | n/a | n/a |

## Per-element f1 delta vs baseline (focus: Si, Mg, Al, Ti)

| cell | Si f1 (delta) | Mg f1 (delta) | Al f1 (delta) | Ti f1 (delta) |
|---|---|---|---|---|
| baseline | n/a | n/a | n/a | n/a |
| ftp1 | n/a | n/a | n/a | n/a |
| 762f | n/a | n/a | n/a | n/a |
| dj6y | n/a | n/a | n/a | n/a |
| ftp1+762f | n/a | n/a | n/a | n/a |
| ftp1+dj6y | n/a | n/a | n/a | n/a |
| 762f+dj6y | n/a | n/a | n/a | n/a |
| all_three | n/a | n/a | n/a | n/a |

## Promotion rule decision

> Pick the cell with the highest macro_f1 lift on
> `vrabel2020_soil_benchmark` subject to no regression
> > 0.02 on any metric on `aalto_libs` / `bhvo2_usgs`.
> Ties within 0.005 macro_f1 → prefer simpler (fewer flags).

**Result:** undetermined — see log below.

### Decision log

```
!! baseline lacks macro_f1 on 'vrabel2020_soil_benchmark'; cannot evaluate promotion rule.
```
