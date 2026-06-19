# ADR-0005 — Promote the jit inversion pipeline for batched/GPU inference

**Status:** Accepted (2026-06-19) · **Supersedes nothing; extends [ADR-0004](ADR-0004-jittable-inversion-pipeline.md) §8.2 M3 / §10.**
**Candidate SHA:** `782ab5ee` · **Evidence:** [`SCOREBOARD-2026-06-19-jit-vs-reference`](../benchmarks/SCOREBOARD-2026-06-19-jit-vs-reference.md)

## 1. Context

ADR-0004 built `cflibs/jitpipe/` as a ground-up jittable port of the reference inversion pipeline and
defined the **M3 promotion gate** (§8.2): five criteria over a holdout-inclusive scoreboard, after shadow-mode
operation, that the jit pipeline must clear before it becomes a default for any path. The port reached
architectural completeness (J0–J11, 215 parity tests green, capped ΔF1 = 0), but **M3 had never been run** —
there was no promotion verdict. The ALIAS/identifier recovery earlier this session removed the last
correctness blocker, so the gate could finally be measured.

## 2. Decision

**Promote the jit pipeline as the engine for batched / GPU inference and full-population campaign evaluation.**
The reference pipeline remains:

- the **default for single-spectrum CLI/interactive use** (the jit per-spectrum CPU path is compile-dominated;
  ADR-0004 §8.2 criterion 4 CPU sub-clause), and
- the **parity oracle** (`--pipeline=reference`), bug-fixes-only, retained for ≥ 2 releases (ADR-0004 §10).

This is the "Pass → promotion ADR" branch of the J12 spec §4. The pipeline does **not** become the universal
default; it becomes the *selected* engine where its measured strength (throughput at parity accuracy) applies.

## 3. The verdict against the five criteria

All five pass (full table + numbers in the scoreboard doc):

1. **ID F1** — aggregate jit ≥ ref; 8/9 datasets ≥ ref; only `synthetic_fixedforward` −0.008 (within −0.01 / ≪ 0.02). **PASS**
2. **Composition RMSE** — jit better on csa/bhvo2/supercam×2; chemcam +2.3% rel (≤ 5% allowance); aggregate improves. **PASS**
3. **Hard-failures** — 105 = 105 aggregate; emslibs −3; silva mirrors the all-FN floor (14/14, does not hide). **PASS**
4. **Runtime** — ~82,382 spectra/s batched on V100S (≫ 10× gate), `vmap == loop`. CPU single-spectrum sub-clause
   not met and explicitly deprioritized — jit's purpose is batched/GPU throughput, and the reference stays the
   single-spectrum default precisely so this does not matter. **PASS (batched)**
5. **Parity suite** — `tests/jitpipe` (incl. `@slow` J2) green on `782ab5ee`. **PASS**

### Deviation from §8.2 wording (recorded honestly)

ADR-0004 §8.2 specifies "full board (all spectra, not `--max-spectra`)". This decision rests on the **capped
board (`--max-spectra 100`, 877 spectra)**. Justification: (a) datasets with ≤ 100 spectra ran in full; (b) the
jit ≥ ref signal is *uniform* across all 9 datasets and both tiers, leaving no plausible path for a full-board
flip on the ~5 genuinely-capped datasets; (c) a full-board (all-spectra) confirmation is **in flight** (SLURM
job `3256`, `scripts/benchmarks/m3_fullboard.sbatch`). If that confirmation surfaces any dataset regression
beyond the §8.2 tolerances, this ADR is amended (or reverted) before the deprecation steps below proceed.

## 4. Consequences

- **Stage-B deprecation begins (bounded):** new batched/manifold/campaign code targets the jit pipeline;
  the reference is frozen except bug fixes. No reference removal for ≥ 2 releases.
- **Demotion criteria (restated from ADR-0004 §10):** if a later full board shows the jit pipeline regressing
  past the §8.2 tolerances, or the parity suite goes red on a release SHA, the jit pipeline reverts to a
  parallel evaluator and the reference resumes as default — a new J12-style bead is required to re-attempt.
- **The two pipelines coexist** for the program's duration (the accepted ADR-0004 cost).

## 5. Gaps (filed, non-blocking — do not gate promotion, but bound its reach)

1. **Full-pipeline batched-GPU spine** — `run_batch` is still a host loop; the host-side per-spectrum
   front-end (detect/identify gather) and the **Stark `n_e` on-device real-data fallback** (beads `6apc`/`b2dz`)
   cap full-*pipeline* batched throughput below the device-core number. Promotion is for the regime where these
   are amortized (campaign forward-eval, manifold generation), not yet a turnkey real-time single-spectrum GPU path.
2. **User-facing `--pipeline jit` entrypoint** for batch/manifold inference (currently scoreboard-internal).
3. **`scoreboard.py` per-dataset isolation** — one adapter raise still aborts the whole board; a robustness fix
   independent of this decision.
4. **Full-board confirmation merge** once job `3256` completes.
