# ExoJAX-grade CF-LIBS reference forward + structured-GN inversion harness

This is the reproducibility code behind the committed v4 realtime plan:
[`docs/research/realtime/2026-06-20-realtime-plan-v4-real-data-accuracy.md`](../../../docs/research/realtime/2026-06-20-realtime-plan-v4-real-data-accuracy.md).
It is an **independent ExoJAX-based reference** forward model + structured-Jacobian
K=1 Gauss-Newton inversion harness used to **validate** the production core-JAX-only
`cflibs/jitpipe` forward — it is NOT a duplicate of the shipped pipeline.

> **Isolation / dependency note.** ExoJAX is NOT a declared CF-LIBS dependency, and
> importing it pollutes the test suite. These scripts therefore live under
> `scripts/research/rtval/` and are deliberately kept out of pytest collection
> (`pytest.ini` sets `testpaths = tests`, so `scripts/` is never collected; none of
> these files are named `test_*.py`). Run them only in the dedicated ExoJAX venv
> described below — never inside the normal test environment.

> **Paths are examples.** Every `/tmp/rtval/...` path below is a vasp-01-specific
> EXAMPLE, not a committed location. Point `--db-path`, `--bundle`, `--output`,
> `--out-dir`, and `--real` at wherever you stage the bundle/data on your host.

Original staging (vasp-01, for reference): `vasp-01:/tmp/rtval/`;
ai-proxy mirror: `/tmp/validate.py` + `/tmp/rtval_local/*`.

## What this is

- **`reference_forward.py`** — ExoJAX-grade REFERENCE forward model. Single-zone LTE
  plasma EMISSION computed with ExoJAX's atomic-line opacity engine
  (`exojax.opacity.lpf.lpf.xsvector` / `vvoigt`: Voigt-Hjerting line profiles +
  line-strength summation, Faddeeva-based, custom-JVP) driven by OUR Saha-Boltzmann
  level populations (ExoJAX ships NO Saha; we supply it in log-space via `lax.logistic`).
  Instrument FWHM is folded EXACTLY into the Voigt Gaussian core (sigma added in
  quadrature) — Gaussian(x)Voigt is a Voigt — so no separate convolution. jit/vmap-clean,
  differentiable. `forward_with_basis(T,ne)` returns the (G, E) per-species basis B so
  the spectrum is LINEAR in composition (S = B @ comp).
- **`reference_inversion.py`** — STRUCTURED-Jacobian K=1 Gauss-Newton (the varpro_bench
  "structured" strategy). Concentration columns are the basis B (no autodiff); the 2
  nonlinear (T, log_ne) columns use FORWARD DIFFERENCES (a single JVP through ExoJAX
  `hjert` is ~3.4 ms — 9x a forward — so FD with 2 extra basis evals is the lever).
  Marquardt relative (diagonal) damping. theta=[T, log10 ne, raw...], comp=softmax(raw),
  peak-normalized target.
- **`build_atomic_bundle.py`** — builds the pluggable static atomic-line bundle
  (lambda, gA=gk*Aki, E_k, ion_stage, Stark, partition polys, IP, mass) from the cflibs
  production SQLite DB. Bundle is a single `.npz` consumed by the forward model.
- **`validate.py`** — driver with `--synth` (round-trip a known T,ne,comp) and
  `--real <npz>` (invert a real-spectra-agent spectrum from `/tmp/rtval/data/`).

## Environment

- GPU: Tesla V100S-PCIE-32GB, vasp-01.
- venv: `/tmp/exojax_venv` (ExoJAX 2.5.0 + jax/jaxlib 0.9.2 [cuda12]). JAX sees
  `[CudaDevice(id=0)]`, `default_backend()=="gpu"`.
- Run with `bash -lc 'unset LD_LIBRARY_PATH; /tmp/exojax_venv/bin/python ...'`
  (login shell / stale LD_LIBRARY_PATH force CPU fallback).

## Build + run

Run these from inside `scripts/research/rtval/` (so `validate.py`'s flat
`from reference_forward import ...` sibling imports resolve), with `cflibs`
importable on the venv path. `export_truth_datasets.py` needs `cflibs`; run it as
`PYTHONPATH=<repo-root> python export_truth_datasets.py ...` to avoid the worktree
import trap.

```bash
cd scripts/research/rtval

# 1. atomic bundle (9 ChemCam elements, 405 real NIST lines, 240-850 nm)
python build_atomic_bundle.py \
  --db-path /cluster/shared/cf-libs-bench/jitpipe-m3/repo/ASD_da/libs_production.db \
  --elements Si Ca Fe Al Mg Na K Ti Mn --wl-min 240 --wl-max 850 \
  --lines-per-elem 24 --output /tmp/rtval/atomic/bundle.npz

# 2. synthetic round-trip (recover known comp; sub-ms at ~1000-ch ROI)
python validate.py --synth --bundle /tmp/rtval/atomic/bundle.npz --k 1 --channels 1000

# 3. real ChemCam spectrum
python validate.py --real /tmp/rtval/data/chemcam_calib.npz \
  --bundle /tmp/rtval/atomic/bundle.npz --k 1 --index 0 --wl-min 240 --wl-max 300
```

## Measured (V100S, float32, K=1, median of 120 single-shot calls)

| case | channels | latency median | sub-ms | notes |
|------|---------:|---------------:|:------:|-------|
| synth | 4000 | 2065 us | no | full ExoJAX Voigt path; dense Voigt scales w/ grid |
| synth | 1000 | 708 us | YES | T_err ~5%, comp_rmse ~0.10 (30-case avg) |
| synth | 500  | 564 us | YES | min 472 us |
| real ChemCam (240-300nm) | 1163 | 906 us | YES | T=8830 K, ne=1.0e17 recovered |
| real ChemCam (240-850nm) | 5860 | 2092 us | no | T=8688 K, ne=1.0e17 |

Sub-ms is reached at LIBS-realistic ROI (~1000 channels). The forward is the FULL
ExoJAX opacity engine — NOT a hand-rolled Voigt fallback.

## Known limitations / honest caveats

- ne is the weakest-recovered direction (~29% rel err at K=1): Stark broadening only
  weakly modulates the spectrum at ne~1e17 — a real CF-LIBS degeneracy, not a bug.
- K=1 from a FLAT composition prior (raw=0) gives comp_rmse ~0.10 on a 9-elem simplex.
  Production use should warm-start composition from a Boltzmann-plot/NNLS prior; K>1
  did not improve here from the flat start (overshoot on the ill-conditioned ne axis).
- Real-mode composition comparison is number-fraction(recovered) vs wt%(truth) over
  shared elements, both renormalized — diagnostic, NOT a calibrated CF-LIBS quant.
