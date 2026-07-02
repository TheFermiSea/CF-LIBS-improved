# Strict Single-Shot <1 ms CF-LIBS Composition Pipeline — Plan v2 (ExoJAX2-evaluated)

**Date:** 2026-06-19
**Supersedes:** `/tmp/realtime_plan.md` (v1).
**New inputs folded in this revision:**
- `/tmp/exojax_latency.json` — ExoJAX2 v2.5.0 single-shot GPU benchmark (same V100S, float32, batch=1).
- `/tmp/exojax_capability.md` — ExoJAX2 data-stack + physics-fit audit (what to reuse vs not).
- `/tmp/asta_redo.md` — re-verified asta literature on sub-ms physics inversion + atomic data beyond NIST ASD.
- (carried) `/tmp/rt_latency.json` — the hand-rolled single-shot V100S latency measurement.

**One-line ExoJAX2 verdict:** **OFFLINE.** ExoJAX2 is for offline manifold/reference-accuracy work,
**not** the <1 ms hot path — measured K=3 Gauss-Newton with ExoJAX is ~11.3 ms (7.3x over the
hand-rolled 1.54 ms), and *no* trimmed config crossed under 1 ms for K=3.

---

## 0. Executive summary

The v1 feasibility result is **unchanged and re-confirmed**: a differentiable fixed-step Gauss-Newton
single-shot CF-LIBS inversion hits **strict <1 ms on the measured V100S in a bounded configuration**
(K=3 GN on a <=1000-channel ROI = **684 us**, recommended default; K=3 @ 500 ch = 571 us; K=2 @
<=1000 ch ~= 550 us), while the demanding **full config (K=3 @ 4000 ch) is 1.46-1.54 ms, ~50% over
budget**. The regime is **dispatch/launch-bound, not compute-bound** (trivial-op floor 126 us; K=3 @
256 ch still 502 us), so the winning levers are **fewer kernel launches** — manifold warm-start to keep
K tiny, fixed-K `lax.scan`, ROI windowing, and CUDA-graph capture — not fewer FLOPs.

The **new decision** this revision adds is ExoJAX2's role. We benchmarked ExoJAX2 v2.5.0 on the same
V100S at the CF-LIBS target (300 lines x 4000 channels, 17 params, batch=1, float32). The result is
decisive: ExoJAX's **forward** is 390 us (2.7x the hand-rolled 142 us) and its forward-mode **Jacobian**
is 3386 us (6.5x the hand-rolled 520 us), giving an **estimated K=3 inversion of ~11.3 ms** — 7.3x over
the hand-rolled 1.54 ms and an order of magnitude over the <1 ms budget. Critically, ExoJAX's forward is
**not** launch-limited (34 jaxpr eqns / ~16 fused HLO kernels); the killer is the **(N_line x N_chan) =
1.2M-element Voigt-Hjerting numatrix** OpaDirect materializes, whose cost the 17-input jacfwd multiplies
~9x. Even the most aggressive trim (100 lines @ 1000 ch) still yields K=3 ~ 1.19 ms, and **no measured
configuration crossed under 1 ms for K=3.** ExoJAX2 is therefore confirmed **offline-only**: use it for
manifold pre-computation and reference-accuracy cross-checks; **keep the hand-rolled fused float32
forward + GN kernel on the hot path**.

What we **do** adopt from the ExoJAX ecosystem is its **physics-only data plumbing** (it is jax + radis
+ numba with **zero** banned imports — passes our TID251 ban and AST blocklist): VALD3/Kurucz line
ingestion, Barklem & Collet 2016 / Irwin 1981 partition functions, Voigt/Hjerting profile primitives,
and H-/Rayleigh continuum. These feed **offline** into the baked per-line static arrays the hot-path
kernel reads at zero runtime cost. We do **not** reuse ExoJAX's multi-layer atmospheric RT, and we keep
CF-LIBS's own `(T, n_e)` Saha/Boltzmann solver (ExoJAX has no Saha; `n_e` enters it only as Stark
broadening width).

---

## 1. <1 ms FEASIBILITY (UNCHANGED — restated crisply)

Device: **Tesla V100S-PCIE-32GB**, JAX 0.9.2, **float32**, warm (compile-once then median of >=200
per-call `block_until_ready`). Problem: 4000 channels, 300 lines, 15 elements, 17 params, GN damping
lambda=0.01. Single-shot = batch=1 with each call individually blocked (true latency, not throughput).

| Quantity | median (us) | min (us) |
|---|---:|---:|
| forward `S(theta)` | 142.1 | 116.6 |
| Jacobian `dS/dtheta` (jacfwd, 4000x17) | 520.0 | 461.1 |
| one GN step (fwd + jac + 17x17 solve), fused jit | 657.1 | 614.5 |
| **K=3 fused GN (the actual single-shot inversion)** | **1543.5** | **1459.2** |
| K=2 fused GN | 1139.0 | — |
| trivial 1-op jit (launch-overhead floor) | 125.9 | 91.7 |
| compile (one-time) | 2292 ms | — |

**Configs that meet strict <1 ms (measured lever tables):**

| Config | Measured median | Verdict |
|---|---:|---|
| K=3 + ROI <=1000 ch | **684 us** | **PASS (recommended default)** |
| K=3 + ROI 500 ch | 571 us | PASS (aggressive ROI) |
| K=2 + ROI <=1000 ch | ~550 us (interp) | PASS |
| K=1 + 4000 ch | 605 us | PASS on time, risky on convergence |
| K=2 + 4000 ch | 1013 us | BORDERLINE FAIL (13 us over) |
| K=3 + 4000 ch | 1496 us | FAIL (the demanding config) |

**Why launch-bound, not compute-bound (3 independent signals):**
1. The trivial 1-op jit floors at **126 us median / 92 us min** — ~1100x more than the FLOPs warrant;
   pure Python->XLA dispatch + kernel-launch + sync tax.
2. K=3 at **256 channels** still floors at **502 us** — collapsing channels 16x (4000->256) only drops
   K3 from 1496->502 us; the residual ~500 us is *stacked op launches across 3 GN steps*, not arithmetic.
3. Async-pipelined K=3 is 1262 us vs 1543 us blocking — the ~280 us gap is the per-call sync tax.

**Implication (unchanged):** the path to <1 ms is **fewer kernel launches**. Reduce K (fewer stacked op
groups), fuse the whole K-step loop into one `lax.scan` region, ROI-window to <=1000 channels, and
CUDA-graph the loop. **Ship the bounded ROI config as the strict-<1 ms product; treat full-spectrum
K=3 @ 4000 ch as a CUDA-graph stretch goal that must be benchmark-gated before any <1 ms claim
(plausible landing ~0.7-1.0 ms, not proven).**

---

## 2. ExoJAX2 ROLE — hot-path forward vs OFFLINE (decided from measurement)

### 2.1 The head-to-head benchmark (same V100S, batch=1, float32, 300 lines x 4000 ch, 17 params)

| Quantity | Hand-rolled (rt_latency) | ExoJAX2 OpaDirect (exojax_latency) | Ratio |
|---|---:|---:|---:|
| forward `S(theta)` | 142.1 us | 389.6 us | **2.7x slower** |
| Jacobian `dS/dtheta` | 520.0 us (jacfwd) | 3386.4 us (jacfwd; jacrev OOMs @ 17.95 GiB) | **6.5x slower** |
| **K=3 GN inversion** | **1543.5 us (measured)** | **~11328 us (estimated 3x(fwd+jac))** | **7.3x slower** |
| forward op count | (fused) | 34 jaxpr eqns / ~16 fused HLO kernels | forward NOT launch-bound |

ExoJAX ROI sweep (jacfwd; K3 = 3*(fwd+jacfwd)) confirms the structural problem rather than a tuning miss:

| ROI | ExoJAX fwd | ExoJAX jacfwd | ExoJAX K3 |
|---|---:|---:|---:|
| 256 ch | 215 us | 266 us | **1443 us** |
| 500 ch | 203 us | 312 us | **1545 us** |
| 1000 ch | 208 us | 345 us | **1659 us** |
| 2000 ch | 323 us | 1814 us | 6413 us |
| 4000 ch | 370 us | 3177 us | 10639 us |

And the line-count sweep at 1000 ch: **100 lines** -> K3 = **1187 us**; 300 lines -> 1839 us. So even the
single most aggressive trim measured (100 lines @ 1000 ch) is **still 1.19 ms** — over the strict <1 ms
budget. **No measured ExoJAX configuration crossed under 1 ms for K=3.**

### 2.2 Why ExoJAX loses on the hot path (root cause, not a tuning artifact)

The forward op-launch count is **fine** (~16 fused kernels, 34 jaxpr eqns) — ExoJAX is *not*
launch-limited the way our hand-rolled batch=1 path is. The killer is **compute structure**: OpaDirect
(LPF line-by-line) materializes a full **(N_line x N_chan) = 300 x 4000 = 1.2M-element Voigt-Hjerting
`numatrix`** per evaluation. That is fine for ExoJAX's *batched, broadband, accuracy-first* design point,
but at batch=1 it is heavier than our quadrature-folded single-Gaussian-scatter forward, and the
**17-input forward-mode Jacobian replicates that 1.2M-element matrix across 17 tangents (~9x blowup)** —
which is exactly the ~6.5x Jacobian / ~7.3x K3 penalty we measured. This is intrinsic to a faithful
Voigt-Hjerting LPF opacity engine; trimming lines/channels does not buy a sub-ms K=3.

### 2.3 Decision: ExoJAX2 is OFFLINE; the hot path stays custom

**REUSE from ExoJAX (OFFLINE only):**
- **Atomic-line ingestion:** `database/vald` (`AdbVald`/`AdbSepVald`) and `database/kurucz`
  (`AdbKurucz`) — deeper metal/iron-group line lists than NIST ASD, plus per-line radiative/Stark/vdW
  damping constants (`gamRad`, `gamSta`, `vdWdamp`) ASD does not carry.
- **Partition functions:** `database/core_atom/pf.py` — Barklem & Collet 2016 U(T) on a 42-pt grid for
  284 species (`load_pf_Barklem2016`, `interp_QT_284`) + Irwin-1981 polynomial `log U = sum a_n (log T)^n`
  (the *same* polynomial form CF-LIBS already uses) — drop-in for the offline DB-build partition layer.
- **Voigt / Hjerting / Faddeeva primitives:** `special/faddeeva.py`, `erfcx.py`, `expn.py`,
  `opacity/lpf/` — as the **reference-accuracy** line-shape used to validate our cheaper folded-Gaussian
  hot-path approximation, and as the opacity engine inside the offline manifold generator.
- **Boltzmann line strength + T-reweight** (`core_atom/line_strength.py`, PreMODIT `f(E,T)=
  exp(-c2 E (1/T - 1/Tref))`) — for offline cross-checks of our Boltzmann population physics.
- **PreMODIT precompute-LBD-once / reweight-per-T pattern** (`opacity/premodit`) — the natural template
  for the offline `cflibs/manifold/` grid generator (amortize line work once; cheap per-(T) evaluation).
- **Continuum H-/Rayleigh** (`opacity/opacont.py:OpaHminus`, `opacity/rayleigh.py`) — differentiable
  JAX `(T, nu)` functions usable in the offline forward for LIBS continuum components.

**KEEP CUSTOM (HOT PATH, the <1 ms kernel):**
- **Single-zone LTE emission with our own `(T, n_e)` Saha/Boltzmann solver.** ExoJAX has **no Saha
  ionization-balance solver** anywhere in `src/` — populations are a function of **T only** (Boltzmann x
  per-species U(T)); ion-stage ratios come from external chemistry, and `n_e` enters ExoJAX **only** as
  the `Nelec` term in Stark broadening (`gamma_vald3`), a line *width*, not a population driver. CF-LIBS
  must own its `SahaBoltzmannSolver` (the `(T,n_e)` -> ion-ratio + level-population core).
- **The fused float32 GN kernel** (`forward` + `jacfwd` + damped 17x17 solve under `lax.scan`), with the
  **quadrature-folded single Gaussian-scatter matmul** (Doppler+Stark+instrument folded into one
  effective sigma; deliberately avoiding the cuDNN conv path AND avoiding the (N_line x N_chan)
  Voigt-Hjerting numatrix that sinks ExoJAX at batch=1). This is the 142 us forward / 1.54 ms K3 path.
- **Manifold nearest-neighbor warm-start** (physics grid + argmin/FAISS), CUDA-graph capture, resident
  device buffers — all the launch-reduction machinery from Section 1.

**What CF-LIBS does NOT reuse from ExoJAX:** everything in `rt/` (multi-layer `dtau` (N_layer, N_nu),
two-stream/n-stream layer solvers, chord/transmission geometry, Toon reflection) and the `(T,P)`
atmospheric/hydrostatic driving convention. CF-LIBS is single-/two-zone optically-thin/thick LTE plasma
emission, not a layered atmosphere.

**Integration shape (offline build, hot-path read):**
`ExoJAX (VALD3/Kurucz ingestion + B&C/Irwin U(T) + Voigt reference + H-/Rayleigh) -> offline DB build ->
baked per-line static arrays (line_center, line_Ek, line_gA + data-quality weight, line_stark coeffs,
U_coeffs) in PipelineSnapshot/BatchAtomicData -> CF-LIBS hot-path kernel: Saha(T,n_e) -> Boltzmann ->
emissivity -> folded-Gaussian scatter -> instrument convolution.` No ExoJAX code on the hot path; no
layered RT in the loop.

---

## 3. ATOMIC / DATA STRATEGY beyond NIST ASD (via ExoJAX ingestion + asta-verified sources)

NIST ASD is the gold standard for *evaluated* gA but is incomplete (weak/UV/iron-group lines absent or
high-uncertainty), carries **no Stark-width data**, and has limited high-T partition functions. ExoJAX's
ingestion layer (VALD3/Kurucz/B&C/Irwin) is the plumbing that lets us pull these in **offline**, baking
the result into static arrays. All sources below are physics data (no ML) — fully physics-only-compliant.

### 3.1 gA / oscillator strengths — by element/regime

| Element / regime | Primary source | Why (asta-verified) | Role |
|---|---|---|---|
| **Common/light elements, low-charge ions** (the analytic spine) | **NIST ASD** (graded) + **VALD3** curated, NIST-preferred provenance | VALD picks best gA per line (NIST > experimental > HFR > Kurucz) + carries Stark/vdW constants; Ryabchikova 2015 DOI `10.1088/0031-8949/90/5/054005`; VALD3 arXiv `1710.10854`. | **Trusted spine**: prefer ASD-grade gA for the strong analytic lines a calibration-free method trusts. |
| **Transition metals / iron group / UV-blue line forest** (Fe, Ni, Cr, Ti, Mn) | **Kurucz** line lists (~10^7-10^8 lines) for **completeness**, **down-weighted** where it disagrees with NIST/VALD/OP | Completeness-over-accuracy trade; Kurucz gA replaced by Opacity Project in WR opacity work (CorpusId 12736045); MAFAGS-OS / halo-star cross-checks (CorpusId 16044163). | Populate weak/blended lines for forward-model fidelity; **per-line data-quality weight** down-ranks unreliable Kurucz gA in the GN residual. |
| **Rare earths / lanthanides, complex spectra** | **Cowan/HFR** (semi-empirical, fitted to observed levels) and **FAC (Gu)** relativistic-CI for any missing/highly-ionized lines | HFR ~10-20% on strong lines, often the *only* source for complex spectra; FAC fills completeness for arbitrary ions but gA is *calculated* (10-30%). Gu, *Can. J. Phys.* 86, 675 (2008), DOI `10.1139/p07-197` [reference-known]; FAC-in-use: Del Zanna ArXIII DOI `10.1016/J.ADT.2004.07.002`. | Synthesize *missing* lines only; defer to NIST/VALD where graded. Apply Slater-Condon scaling (~0.85-0.90) for transition-metal radial integrals. |
| **gA accuracy audit (all of the above)** | **BRASS** benchmark cross-check | Quality-controlled cross-validation of Kurucz/VALD/NIST gA against benchmark stellar spectra; Lobel et al., *Atoms* 7(4):105 (2018), DOI `10.3390/atoms7040105`. Fe I specifically: *Solar Physics* (2019) DOI `10.1007/s11207-019-1543-2`. | Sets the **per-line data-quality weight** baked into the residual (down-weight low-grade lines). |

ExoJAX ingests VALD3 (`AdbVald`) and Kurucz (`AdbKurucz`) directly to HDF5 caches; FAC/Cowan outputs are
merged at DB-build. **All gA is a frozen per-line constant** -> enters the kernel as a static `line_gA`
array at **zero runtime cost**.

### 3.2 Partition functions (high-T Saha-Boltzmann)

- **Primary:** **Barklem & Collet 2016** U(T) for all neutral + singly/doubly-ionized atoms (Z<=92) +
  291 diatomics, consistent level cutoffs. A&A 588, A96 (2016), arXiv `1602.03304`, DOI
  `10.1051/0004-6361/201526961`. (Repo already has a B&C-2016 patch script.) Loaded via ExoJAX
  `load_pf_Barklem2016` / `interp_QT_284` at DB-build.
- **Fast evaluator / fallback:** **Irwin 1981** polynomial `log U = sum a_n (log T)^n` for 344 species —
  the exact form CF-LIBS already uses; one einsum at runtime. Irwin, *ApJS* 45, 621 (1981), DOI
  `10.1086/190731` [reference-known]. Bake as `U_coeffs` static array; B&C as primary, Irwin polynomial
  as the cheap hot-path evaluator.

### 3.3 Molecular bands (oxides / radicals) — offline forward option

ExoJAX's molecular stack (**ExoMol** native + **HITRAN/HITEMP via RADIS**) can model LIBS molecular
emission bands (CN, C2, TiO, AlO, CaO/CaOH, oxide tails in geological/SuperCam spectra) *consistently
with the atomic lines* — something a pure NIST-ASD atomic pipeline cannot. **Offline only** (RADIS is a
heavy transitive stack; molecular bands are not in the primary <1 ms analytic-line path). Use for
geological/SuperCam regimes where oxide tails matter; bake resulting band opacity into the offline
forward / manifold, not the hot-path kernel.

### 3.4 Continua — for which regimes

- **H- (negative-hydrogen bound-free + free-free, John 1988)** and **Rayleigh** — ExoJAX `OpaHminus` /
  `xsvector_rayleigh_gas`, differentiable `(T,nu)` JAX functions; relevant to LIBS plasma continuum
  (Rayleigh at the blue end; H- in cooler/recombining plasma). **Reusable offline.**
- **NOT in ExoJAX:** LIBS continuum is dominated by **bremsstrahlung (free-free) + radiative
  recombination**, which ExoJAX (cool-atmosphere target) does not provide. CF-LIBS must supply its own
  free-free + recombination continuum. **CIA** (ExoJAX `OpaCIA`) is high-pressure molecular gas — **not
  relevant** to optically-thin hot LIBS plasma.

### 3.5 Stark widths (NOT in NIST ASD — needed for n_e from broadening)

- **STARK-B database** (Sahal-Brechot, Dimitrijevic et al.; VAMDC node) — theoretical Stark FWHM/shifts
  vs T and n_e; ~20-30% accuracy (semiclassical perturbation theory). *Phys. Scr.* 90, 054008 (2015),
  DOI `10.1088/0031-8949/90/5/054008`. The standard source for the data NIST entirely lacks.
- **Recent Dimitrijevic/Djurovic measurements** for LIBS-relevant high densities — esp. **Al II/III + He
  I 388.86 nm at high n_e**, *Spectrochim. Acta B* (2020), DOI `10.1016/j.sab.2020.105816`; plus O I
  (2025) DOI `10.3390/galaxies13050116`, N V (2025) DOI `10.3390/data10090140`, N VI (2024) DOI
  `10.3390/data9060077`. Use to back **strong diagnostic lines** used for n_e.
- **Hot-path form:** analytic, n_e-linear (`stark_fwhm = line_stark * (n_e/1e17)`); per-line Stark
  coefficient is a baked static array; the n_e dependence is the one live term. VALD3 also carries a
  per-line `gamSta` Stark constant that ExoJAX exposes (`gamma_vald3`'s `gamStark = 10^gamSta * Nelec *
  (T/1e4)^(1/6)` — note it **does** take electron density), usable as a fallback Stark coefficient where
  STARK-B lacks the transition.

### 3.6 Level energies

NIST-shift any ab-initio (FAC/Cowan) levels at DB-build for blend-assignment accuracy; bakes into
`line_center` / `line_Ek` static arrays at zero runtime cost.

**Net:** every accuracy gain beyond ASD is **per-line static data computed offline** and baked into
`PipelineSnapshot` / `BatchAtomicData`. The only *live* hot-path physics terms remain Saha (n_e), Boltzmann
(T), Stark (n_e), and an optional flagged escape-factor self-absorption term.

---

## 4. RISKS + the physics-only-vs-ML tension + dependency weight

### 4.1 Latency / engineering risks (carried from v1, still load-bearing)
- **CUDA-graph capture is the load-bearing assumption for full-spectrum K=3.** If it does not deliver
  ~0.7-1.0 ms, the product *must* fall back to the bounded ROI config. Benchmark-gate before any claim.
- **float64 -> float32 forward port.** Shipped `single_spectrum_forward` is float64; the latency numbers
  are float32. Risk: conditioning in Saha `exp(-Eion/kT)` and partition functions. Keep the 17x17 GN
  solve in float32 + damping; validate round-trip vs float64 reference on golden spectra.
- **K too small -> under-convergence.** K=1 @ 4000 ch passes on time (605 us) but may under-converge.
  Manifold warm-start makes small K viable; gate on composition-recovery error, not just latency.
- **ROI windowing can drop diagnostic lines** for unexpected elements -> biased T/n_e. Choose ROI from
  the candidate element set (presumes ID ran) or use a fixed super-ROI covering all target elements.

### 4.2 NEW — ExoJAX-specific risks
- **ExoJAX on the hot path is the wrong tool, measured.** ~11.3 ms K=3 (7.3x over budget) and no trim
  reaches <1 ms K=3. The risk would be *adopting* it for inference; the measurement closes that door.
  Mitigation: ExoJAX is offline-only by decision.
- **Jacobian memory:** ExoJAX `jacrev` **OOMs** at 4000 ch (tries 17.95 GiB); `jacfwd` works but is the
  ~3.4 ms cost. Even offline manifold/Jacobian work with ExoJAX must use forward-mode or chunked
  reverse-mode and watch device memory.
- **gA accuracy regression risk:** Kurucz/FAC are *completeness over accuracy*; for a calibration-free
  method that trusts absolute line intensities, naively swapping in Kurucz gA for strong analytic lines
  would degrade quantification. Mitigation: VALD/NIST-preferred provenance + BRASS-derived per-line
  data-quality weights; **benchmark-gate every atomic-data swap** against the synthetic corpus + NIST
  parity before it lands (per project memory: identifier/forward changes are benchmark-gated).

### 4.3 Physics-only vs ML tension (stated plainly)
**HARD CONSTRAINT (CLAUDE.md):** shipped CF-LIBS must not import sklearn/torch/tensorflow/keras/flax/
equinox/transformers/`jax.nn`/`jax.experimental.stax`; ML only in `cflibs/evolution/`.

- **ExoJAX does NOT add tension — it is physics-only.** A full banned-import scan of all 183 `src/exojax/`
  `.py` files returned **ZERO** matches; ExoJAX is jax + numba + radis + numpy/pandas/astroquery. The
  subset we'd import (database/, opacity/, special/, partition functions) **passes** our Ruff TID251 ban
  and the AST blocklist scanner.
- **The real tension is the *accuracy-at-<1 ms* gap, unchanged from v1.** If the bounded physics config
  (K=3 @ <=1000-ch ROI, 684 us) meets the accuracy bar, **there is no tension** — ship a fully
  physics-only differentiable-GN pipeline. The manifold nearest-neighbor warm-start is **physics, not
  ML** (precomputed grid + argmin/FAISS) and is the escape hatch that likely makes a learned surrogate
  unnecessary. Only if the bounded config fails accuracy AND full-spectrum K=3 cannot be graph-captured
  under 1 ms would high-accuracy-<1ms force an **ADR-level** revisit of the constraint (learned
  warm-start/surrogate) — a gated decision, not a default. `jax.nn` is banned, so the closure softmax
  stays the manual `exp/sum` form (`cflibs/inversion/physics/softmax_closure.py`).

### 4.4 Dependency weight of adopting exojax/radis
- **Do NOT take the full `exojax` PyPI dependency on the shipped library.** Its stack pulls **radis
  (>=0.15.2)** (heavy: astropy, scipy, transitive), **numba** (LLVM compiler dep), **astroquery,
  PyMieScatt, hitran-api, vaex, zarr, tables, h5py** — most needed only for clouds/molecular/remote-fetch
  paths CF-LIBS does not run at inference.
- **Recommended adoption mode: vendor/extract the targeted physics-only subset** —
  `database/core_atom/`, `database/kurucz/`, `database/vald/`, `database/hminus.py`, `opacity/rayleigh.py`,
  `opacity/lpf/`, `special/`, and the partition-function code — into an **offline DB-build tool**, so the
  shipped CF-LIBS install stays lean and never drags radis/astroquery/PyMieScatt into the runtime. The
  atomic-only ingestion path should be checked to confirm it can avoid the radis transitive import (or
  vendor only `core_atom/` + `opacity/lpf/`). All of that subset is physics-only and ban-compliant.
- **astroquery** is only needed for *fetching* line lists; once cached to HDF5 it is not an inference-time
  dependency. **Net:** exojax/radis are **build-time/offline** dependencies, never hot-path runtime ones.

---

## 5. BUILD SEQUENCE (reuse jitpipe forward/autodiff/manifold; ExoJAX offline; fused GN hot path)

Reuse map (verified in repo):
- **Differentiable forward:** `cflibs/manifold/batch_forward.py::single_spectrum_forward` (Saha ->
  Boltzmann -> emissivity -> Voigt, JAX, **float64 — needs float32 RT variant**); `batch_forward_model`,
  `forward_from_snapshot`.
- **Atomic-data packing:** `cflibs/manifold/batch_forward.py::pack_atomic_data` / `BatchAtomicData`
  (struct-of-arrays pytree); `cflibs/jitpipe/snapshot.py::build_snapshot` / `PipelineSnapshot`
  (host-built, `.npz`-cached, pytree-registered) — the baked static arrays from Section 3 live here.
- **jit/vmap + fixed-iteration solve:** `cflibs/jitpipe` (`run_one`, `run_batch`, `StaticConfig`,
  `PipelineParams`); `cflibs/jitpipe/solve.py::scan_solve` (already a `lax.scan` fixed-iteration solver —
  the template for the fixed-K GN loop).
- **Manifold warm-start:** `cflibs/manifold/{generator,loader,vector_index,basis_index}.py` +
  `cflibs/inversion/solve/coarse_to_fine.py::HybridInverter.invert` (coarse manifold lookup -> fine
  optimize). Reuse the coarse lookup; replace fine L-BFGS-B with fixed-K GN.
- **Closure:** `cflibs/inversion/physics/softmax_closure.py` (manual `exp/sum`, physics-only-safe).
- **Real-time harness + latency SLA:** `cflibs/inversion/runtime/streaming.py` (`FastAnalyzer`,
  `StreamingAnalyzer`, `LatencyStats`, `LatencyMonitor`, `create_streaming_pipeline`).

**Step 0 (NEW) — ExoJAX-fed offline DB-build tool (NO runtime dep; gate at DB build).** Stand up an
offline tool that uses the vendored ExoJAX subset to ingest VALD3/Kurucz, compute B&C-2016/Irwin U(T),
and (optionally) generate Voigt/H-/Rayleigh reference opacity. Output: baked per-line static arrays
(`line_center`, `line_Ek`, `line_gA` + data-quality weight, `line_stark` coeffs, `U_coeffs`) merged into
`PipelineSnapshot` / `BatchAtomicData`. **Gate:** ban-scan the vendored subset (TID251 + AST blocklist);
confirm radis is not pulled into the shipped runtime; benchmark-gate the new atomic data vs synthetic
corpus + NIST parity before promoting any gA/level/Stark swap. *Reuses:* ExoJAX (offline only),
`pack_atomic_data`, `build_snapshot`.

**Step 1 — float32 RT forward kernel (gate: forward <150 us @ 4000 ch).** Port `single_spectrum_forward`
to a float32 RT variant (`cflibs/inversion/runtime/rt_kernel.py`). Keep the quadrature-folded single
Gaussian-scatter matmul (no cuDNN conv; **NO** (N_line x N_chan) Voigt-Hjerting numatrix — that is the
ExoJAX cost we are avoiding). Validate round-trip vs the float64 reference (and, offline, vs the ExoJAX
Voigt reference) on golden spectra. *Reuses:* `single_spectrum_forward`, `pack_atomic_data`.

**Step 2 — fixed-K GN inversion as one `lax.scan` fused jit (gate: K=3 @ 1000 ch <1 ms blocking on
V100S).** Implement the GN step (fwd + jacfwd + damped 17x17 solve), wrap K of them in `lax.scan` (mirror
`scan_solve`), manual softmax closure. Reproduce the **684 us @ 1000 ch / K=3** acceptance number.
*Reuses:* `jitpipe/solve.scan_solve` pattern, `softmax_closure`.

**Step 3 — manifold warm-start hookup (gate: warm-start lookup <100 us; total still <1 ms).** Wire the
coarse manifold nearest-neighbor (`HybridInverter` coarse stage / `vector_index`) to produce theta0
feeding Step 2; confirm K stays <=3 with the warm start on the synthetic corpus. *Reuses:*
`coarse_to_fine`, `manifold/vector_index`, `manifold/loader`.

**Step 4 — CUDA-graph capture + resident buffers (gate: per-shot replay latency; attempt full 4000-ch
K=3 < 1 ms).** Capture the Step-2+3 fused fn as a CUDA graph; pin theta/S_meas/atomic arrays on device
with `donate_argnums`. Re-run the Section 1 lever table; record whether full-spectrum K=3 crosses under
1 ms. **This gate decides whether bounded ROI is the product or full-spectrum is reachable.**

**Step 5 — accuracy gate on the physics path (gate: composition recovery vs synthetic corpus + NIST
parity).** Run the bounded config end-to-end on `output/synthetic_corpus` / golden spectra; measure
T/n_e/C error vs ground truth. **This gate decides the Section 4.3 fork:** physics-only bounded config
passes -> ship it; only if it fails do we open the constraint-revisit ADR. *Reuses:* `cflibs/validation`
(GoldenSpectrum, NIST parity), `cflibs/benchmark`.

**Step 6 — offline manifold-gen via ExoJAX PreMODIT pattern (no runtime cost; gate at build).** For the
offline `cflibs/manifold/` grid generator, adopt ExoJAX's precompute-LBD-once / reweight-per-T pattern
(and optionally its Voigt opacity) for **reference-accuracy** manifold nodes, while the hot-path warm
start still reads the cheap grid. *Reuses:* `cflibs/manifold/generator`, ExoJAX `opacity/premodit`
(offline).

**Step 7 — streaming integration + latency SLA (gate: `LatencyStats` p99 <1 ms over a shot stream).**
Drop the kernel into `runtime/streaming.FastAnalyzer`; assert p99 single-shot latency <1 ms via
`LatencyMonitor`. *Reuses:* `runtime/streaming`.

**Benchmark-gate discipline (project memory):** replicate the strict single-shot timing harness
(compile-once, >=200 per-call `block_until_ready`, median+min) in CI as the latency gate, and the
synthetic-corpus recovery + NIST parity test as the accuracy gate. **Never run the full pytest suite
inside a sub-agent** (watchdog); use narrow subsets or background from the parent.

---

## One-line ExoJAX2 verdict

**OFFLINE.** Keep the hand-rolled fused float32 forward + Gauss-Newton kernel on the <1 ms hot path
(ExoJAX K=3 measured ~11.3 ms = 7.3x over budget, and no trim reaches sub-ms K=3 because OpaDirect's
1.2M-element Voigt-Hjerting numatrix x 17-tangent Jacobian is ~9x too heavy at batch=1); **use ExoJAX2
offline** for atomic-line ingestion (VALD3/Kurucz), partition functions (B&C 2016 / Irwin), Voigt/H-/
Rayleigh reference accuracy, and the PreMODIT manifold-gen pattern — it is physics-only (zero banned
imports), so adopt it as a vendored offline DB-build/manifold tool, never a hot-path or shipped-runtime
dependency.
