# Empirical Analysis of ADR-0001 Prioritization
## Grounded in Exp 1 Smoke Results (2026-05-12 / 13)

**Status:** Analysis only — no implementation. This document maps measured
behavior of the identifier pipeline on real benchmark data (Aalto LIBS,
BHVO-2 USGS, Vrabel 2020 *Scientific Data*) to the pattern catalogue in
`ADR-0001-radis-jaxrts-pattern-survey.md`. The goal is to inform the
prioritization of follow-on ADRs (0002+), not to change ADR-0001 itself.

**Author:** autonomous benchmark session 2026-05-12 / 13. All cited numbers
are from `/tmp/smoke-{correlation,comb,alias,nnls,hybrid}.log` and the live
telemetry stream at `/cluster/shared/cf-libs-bench/telemetry/`. All literature
citations are inherited from ADR-0001 §10 — no new claims made.

---

## 1. Empirical observations (measurements only)

### 1.1 Identifier smoke leaderboard
Methodology: 1 identifier × `--vrabel-max-shots 1 --dataset-shard 1/3 --sections id` × cluster GPU. n=11 scored spectra (post-LOOCV filter); identical data slice for all 5 identifiers; JAX persistent cache shared via NFS (`/cluster/shared/jax-cache`).

| Rank | Identifier | Wall time | F1 (micro) | Precision | Recall | FP/spec |
|---|---|---|---|---|---|---|
| 1 | hybrid_union | 5 m 10 s | **0.715** | 0.690 | 0.742 | 2.0 |
| 2 | spectral_nnls | 5 m 22 s | 0.442 | 0.371 | 0.545 | 5.5 |
| 3 | correlation | 3 m 25 s | 0.177 | 0.538 | 0.106 | 0.5 |
| 4 | alias | 8 m 25 s | 0.141 | **1.000** | 0.076 | **0.0** |
| 5 | comb | 5 m 59 s | 0.028 | 0.200 | 0.015 | 0.4 |

**Sample-size caveat:** n=11 is small. Bootstrap CI bands will be wide. Exp 1 v5 (in flight) will lift to n ≈ 33 × 3 shards = ~100 scored per identifier with multiple iters.

### 1.2 Pipeline cost telemetry
From the cluster-wide telemetry sampler (PR #145, 5-second cadence):

- **GPU utilization during identifier sweep:** 0 – 24 %, mean ~3 % (vasp-01 / 02 / 03). JAX preallocated ~24 GB GPU memory but the kernels are brief microsecond bursts; most of each polling interval lands on CPU-side work (NNLS solvers, scipy.optimize, BoltzmannPlotFitter, numpy preprocessing). Fresh JAX probe confirmed `jax.devices() == [CudaDevice(id=0)]` and arithmetic correctly executes on `cuda:0`, so this is *not* silent CPU fallback — it is **Amdahl-bound on the non-JAX fraction of the per-spectrum loop**.
- **JIT compile-cache file count:** 7,394 files in 44 MB after ~10 min of mixed-cell load on vasp-01 (telemetry sample 2026-05-12T23:13Z). Each unique spectrum shape (varying `n_lines_detected` per spectrum) forces a new compile in the variable-shape paths of `alias.py`.
- **BoltzmannPlotFitter warning rate:** 21 lines/s (default) → 2 lines/s with `CFLIBS_USE_JAX_BOLTZMANN_COMPOSITION=1`. ~10× compute-rate reduction in the per-element T-fit hot path is consistent with most fits now routing through the JAX-vectorized path.
- **Per-iter wall (composition phase, `--sections all`, `--vrabel-max-shots 2`):** ≥ 38 minutes per iter on shard 1 — *did not complete in 1 hour*. This is what forced the pivot to `--sections id` for Exp 1 v5.

### 1.3 Failure-mode signature
From identifier smoke output annotations:

- **`alias`:** `false_positives_per_spectrum = 0.0` with `recall = 0.076`. The output filter is strict — never wrong, often silent.
- **`spectral_nnls`:** `false_positives_per_spectrum = 5.55` with `recall = 0.545`. The output filter is permissive — high recall at the cost of large overprediction.
- **`hybrid_union`:** `false_positives_per_spectrum = 2.0`, `recall = 0.742`. The union recovers nearly all of `spectral_nnls`'s recall while clipping ~64 % of its false positives. This is the empirical signature of two mostly-independent error distributions combining; not a generic ensemble averaging effect.

---

## 2. Mapping observations to ADR-0001 Tier-1 / Tier-2 patterns

### 2.1 Patterns already merged on `feat/adr-0001-pattern-survey-impl`
Verified by inspection of branch tip (2026-05-13 ~ 01:00 UTC):

| ADR ID | Commit | Targets | Addresses measured bottleneck? |
|---|---|---|---|
| T1-3 (lax.while_loop for outer Saha-Boltzmann) | `ad85f85` | `cflibs/inversion/solve/iterative.py` outer loop | **Partially.** The outer `(T, n_e)` iterate is now jit-able and vmap-able. But the dominant per-spectrum cost I measured is in the **per-element WLS Boltzmann fit** (inside step 4 of the loop body), not the outer iterate. T1-3 alone does not eliminate the 21 lines/s slope-positive failure rate; that is `BoltzmannPlotFitter._fit_common_boltzmann_plane` and `_fit_per_element` failing physics gate per element. |
| T1-5 (chunked lax.scan + checkpoint over wavelength) | `b8ba040` | `cflibs/manifold/generator.py`, `cflibs/radiation/spectrum_model.py` | **Not the identifier bottleneck.** This addresses the manifold pre-compute / forward-model OOM problem (cited in ADR-0001 §5.5). My measured per-iter cost is dominated by identification, not forward generation. |
| T1-6 (bayesian.py decomposition + `forward_models/` registry) | `837ab24` | `cflibs/inversion/solve/bayesian.py` | **Relevant to Exp 2 (composition shootout).** Indirectly affects this experiment only when bayesian composition runs. Bead `CF-LIBS-improved-36f6` already records that bayesian runs at 0 % GPU; the decomposition does not by itself fix that. |
| T1-2 (unified forward-model kernel) | `0d74ddd` | `cflibs/radiation/spectrum_model.py` ↔ `cflibs/manifold/batch_forward.py` ↔ `bayesian.py` | **Composition-side win.** Same scope as T1-5. Does not touch identifier path. |
| T1-4 (LDM/DIT Gaussian broadening) | `e640ef2` | `cflibs/manifold/generator.py` | **Forward / manifold pre-compute only.** Not on the identifier hot path. |

**Pattern:** the merged Wave-0 through Wave-3 work is overwhelmingly composition-side and forward-model-side. None of it directly removes the recompile-per-shape pressure or scipy.optimize-per-element pressure that I measured as the identifier wall-time dominator.

### 2.2 Tier-1 / Tier-2 patterns *not yet* merged that **directly** address measured identifier bottlenecks

| ADR ID | Description | Why empirically prioritized |
|---|---|---|
| **T1-1** | `host.py` / `kernels.py` split + shared JAX fallback decorator. Listed targets include `cflibs/inversion/identify/`. | Per ADR-0001 §3.7 #10 + §7.1, this is "upstream of nearly every other Tier-1/2 item." For the identifier path specifically, the host/kernel split unblocks shape-padding and `register_pytree_node` retrofits in `alias.py` / `hybrid_union.py`. Without it, every identifier-side JAX-ification pattern (T2-3, etc.) lives in `*Jax` shadow files. |
| **T2-3** | Manual `register_pytree_node` on `SingleZoneLTEPlasma` + `InstrumentModel`. ADR explicitly states: *"eliminates jit recompile on element-list change"*. | **This is exactly the problem.** 7,394 JIT cache entries in 44 MB on a single shard's smoke = strong evidence that pytree-instability of the plasma-state container is forcing recompiles. Effort listed as Low (~1 day each). |
| **T2-4** | Two-stage line DataFrame (`lines_raw` / `lines_scaled`). ADR states: *"eliminates redundant SQLite hits inside the solver loop"*. | The BoltzmannPlotFitter loop is per-element, per-spectrum. If each call refetches atomic lines from SQLite, the CPU-bound fraction I measured includes pure I/O. T2-4 would isolate that. |
| **T2-7** | `lax.while_loop`-based bisection on `log n_e` (replaces `scipy.optimize.brentq`). | Another scipy.optimize call inside the per-spectrum loop. Bisection-via-while-loop is jit/vmap-compatible. Effort: Low (~1 day). |
| **T2-1** | `@custom_jvp` on Voigt / Stark piecewise cutoffs. | Not a wall-time fix, but a *correctness* fix needed before L-BFGS-B (joint mode) and HMC (bayesian) gradients are trustworthy. Bead `CF-LIBS-improved-8yuo` already records the rename-or-fix problem for `bayesian` plasma-param marginalization. |

### 2.3 Tier-1 / Tier-2 patterns *not yet* merged that are **secondary** to measured bottlenecks
| ADR ID | Reason for deferral |
|---|---|
| T2-2 (frozen-dataclass snapshots, MemoryPolicy) | Low-impact for current GPU 0–3 % regime. Useful once T1-1 + T2-3 close the JIT-recompile loop. |
| T2-5 (PartitionFunctionSource ABC) | Not on the per-spectrum hot path. |
| T2-6 (named-model template registry) | Orchestration polish. |
| T2-8 (input/params/misc namespacing on `SpectrumModel`) | Hashable for memoization is downstream of T2-3. |
| All Tier-3 (T3-1 through T3-7) | Polish + dev-experience; defer. |

### 2.4 Tier-4 (research-grade / speculative)
| Pattern | Empirical posture |
|---|---|
| `lax.custom_root` over the Saha-Boltzmann fixed-point | ADR-0001 §5.4 candidate 3. **Genuinely novel** — neither jaxrts nor exojax does this. With T1-3 already merged, this is the natural next step *if* the outer-loop wall is shown to dominate. **My measurements do not yet show this.** The current dominator is per-element fits (step 4), not the outer iterate count. I would not promote `custom_root` until the per-element bottleneck is addressed (T2-3 + T2-4 + T2-7). |

---

## 3. Proposed adjustments to follow-on ADR ordering

Strictly grounded in measured data; effort estimates inherited from ADR-0001 §8.

### 3.1 Reorder the next 3 ADRs to address the identifier bottleneck before more composition work

1. **ADR-0002 (NEXT): T1-1 host/kernel split applied to `cflibs/inversion/identify/`** *only*. Scope-limit the split to the identifier package before the broader 31-file rollout. Justification: the empirically-measured identifier wall-time (8 m 25 s for alias on 11 spectra) is currently the gating cost for Exp 1, Exp 2, Exp 4, and every future identifier-shootout benchmark. The full host/kernel split is large and risks merge conflicts with in-flight identifier JAX-port work; a scoped identifier-only split is the minimum-viable form.
2. **ADR-0003: T2-3 register_pytree_node on `SingleZoneLTEPlasma` (and any other identifier-touching containers)**. Cited evidence: 7,394 JIT cache entries on a small shard. Direct ADR-0001 quote: *"eliminates jit recompile on element-list change"*. Effort: Low (~1 day per container).
3. **ADR-0004: T2-7 bisection on `log n_e`** (jaxrts pattern, `helpers.py::bisection` cited at ADR-0001 §5.4#2). Replaces `scipy.optimize.brentq` in the inner loop. Low effort. Compounds with T1-3's outer-loop jit.

### 3.2 Defer

- **T1-1 broad rollout** (all 31+ files). The scoped identifier-only T1-1 above can ship first; the rest can wait for an explicit cost/benefit ADR.
- **All Tier-3 polish patterns** until at least the three above land. The dev-experience wins do not move the F1 / wall-time needle.

### 3.3 Promote a missing item not currently in ADR-0001

The ensemble finding (hybrid_union F1=0.715 > spectral_nnls 0.442 > alias 0.141) suggests a **research direction the ADR does not cover**: *ensemble identification design*. The existing bead `CF-LIBS-improved-7nmw` proposes a 2-of-3 confirmation rule across alias / comb / correlation. My data suggests the *union* pattern works *better* than the *intersection* / *confirmation* pattern, because the component identifiers have **orthogonal error distributions** (alias under-predicts; spectral_nnls over-predicts). This is an algorithmic-research question, not a JAX-port question, and belongs in the GPD-Research-Publication-Pipeline epic (`CF-LIBS-improved-mm2`, currently frozen).

I am **not proposing a new ADR for this** until the Exp 1 v5 sweep gives a bootstrap CI on the rankings. The single-iter smokes are an underpowered signal.

---

## 4. Explicit unknowns

Things I cannot defensibly claim without more data — listed so they don't get smuggled into future ADRs:

1. **Whether T1-3's "~10× faster on GPU" claim** (ADR-0001 §5.4 candidate 1, cited from jaxrts) materialized for CF-LIBS' inversion loop. The branch shipped `ad85f85` but I have not measured a before / after wall on identical data. The merged tests assert numerical parity, not performance. **Action item:** Exp 4 (final-number run) should include both `ad85f85` and `pre-ad85f85` runs on identical input.
2. **Whether GPU utilization can reasonably exceed 5 %** for the identification pipeline as currently written. My measurements show 0–24 %. The Amdahl ceiling depends on the non-JAX fraction; ADR-0001 §5.3 enumerates the JAX migrations needed but does not state a target utilization. Until T1-1 + T2-3 + T2-7 land, the per-spectrum loop is still dominated by Python overhead and scipy calls.
3. **Whether the hybrid_union F1=0.715 generalizes** beyond the n=11 smoke. Bead `CF-LIBS-improved-r4qz` (PAPER) is contingent on Exp 1 v5 bootstrap CI confirming the ranking is stable.
4. **Whether T1-6 (bayesian decomposition)** improves bayesian-workflow GPU utilization. Bead `CF-LIBS-improved-36f6` reports 0 % GPU on heavy tier; T1-6 is structural (split monolith) but does not by itself change device placement. **Action item:** Exp 2 (composition shootout) should include `jax.devices()` probe + telemetry check at the start of each bayesian iter.

---

## 5. Citations
All sources inherited from ADR-0001 §10. The peer-reviewed claims this analysis depends on:

- Lütgert, B. et al. (2026) "jaxrts: A code for analyzing X-ray Thomson scattering spectra with autodifferentiation." *Comput. Phys. Commun.* 110173. (ADR Stream C primary.)
- Kawahara, H. et al. (2022) "Exoplanetary Spin Inference Using Stellar Spectrum Templates." *ApJS* 258 31. (ADR Stream C primary.)
- Kawahara, H. et al. (2025) "ExoJAX2." *ApJ* 985 263; arXiv:2410.06900. (ADR Stream C extended.)
- Mollière, P. et al. (2019) "petitRADTRANS." *A&A* 627 A67. (ADR Stream D primary.)
- Malik, M. et al. (2017) "HELIOS." *AJ* 153 56. (ADR Stream D primary.)
- van der Walt, T. (Radis): peer-reviewed and unrefereed sources catalogued in ADR-0001 §10.1.

No new literature claims are made in this analysis. All ADR-0001 citations were taken at face value; no separate verification was attempted in this session.

---

## 6. Recommended next data step (not implementation)

Before *any* of the ADR-0002 / 0003 / 0004 work is committed, the in-flight Exp 1 v5 sweep must complete and produce a bootstrap CI on the F1 leaderboard above. If the CI on hybrid_union vs spectral_nnls overlaps substantially (e.g. > 30 %), the prioritization in §3.1 should be re-examined: the identifier-bottleneck path is only worth optimizing if the identifier ranking is itself stable. If the CI does *not* overlap (hybrid_union remains the clear winner), the prioritization in §3.1 is the right next step.

Estimated time-to-Exp-1-v5-completion at write-time: ~1.5 hours wall (parallel across 3 shards on vasp-01 / 02 / 03; warm JAX cache from smokes).
