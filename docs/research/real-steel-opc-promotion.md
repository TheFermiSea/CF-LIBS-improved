# Promoting the real-steel OPC config into the shipped CF-LIBS pipeline

**Status:** plan (2026-06-27). Companion to `docs/research/real-steel-accuracy-levers.md`.
**Goal:** promote the winning real-steel lever combination — neutral-anchor line selection +
fixed/optimal-`T` + matrix-matched One-Point-Calibration (OPC) `F` — from the benchmark layer
(`tests/benchmarks/real_steel/`) into the shipped `cflibs/` pipeline as an opt-in **known-matrix /
OPC mode**. The DED real goal *is* a known matrix (Ti-6Al-4V, steel), so this is the deployment
sweet spot, not a corner case.

## Why (the measured result)

Honest held-out RMSEP on the 36-sample real-steel gate (`data/real_steel/steel_266.parquet`):

| Config | overall | Fe | Cr | Ni | Si | Mn | Mo | Cu |
|---|---|---|---|---|---|---|---|---|
| baseline (no levers) | 39.04 | 79.8 | — | — | — | — | — | — |
| L2 (neutral-anchor lines) | 30.43 | 64.5 | | | | | | |
| L1 (L2 + fixed optimal-`T`) | 29.98 | | | | | | | |
| L4 (L2 + self-absorption) standalone | 20.78 | 42.7 | 12.7 | 13.7 | 2.7 | 13.4 | 4.1 | 25.4 |
| **v1** (L2+L1+single-standard OPC) | **16.48** | | | | | | | |
| **v2** (L2+L1+**robust** OPC, L4 off) | **10.12** | 20.66 | 2.56 | 3.43 | 7.20 | 1.50 | 10.81 | 10.07 |
| v2 + L4 (regression) | 11.35 | 23.43 | 3.24 | 3.10 | 6.45 | 7.25 | 11.43 | 10.38 |

**Winning config = v2** (`tests/benchmarks/real_steel/best_config_v2.py`, `run_v2`, `use_l4=False`):
baseline 39.04 → v1 16.48 → **v2 10.12 wt% held-out**, a 4x reduction. Three honesty rules are
baked in and must survive promotion:

1. Standard **selection** uses only each candidate standard's own data + own certified truth (a
   conditioning gate at fixed `T`), never the held-out samples.
2. The OPC `F` is a **geometric mean over all well-conditioned standards** (robust to one
   keystone-collapsed standard), with clamp-saturated per-element factors filtered out.
3. **Leave-one-out**: a standard that is itself scored never sees its own `F`.

**L4 self-absorption is intentionally dropped from v2.** It helps standalone (Fe 64.5 → 42.7) but
*regresses* once OPC is present (overall 10.12 → 11.35, Fe 20.66 → 23.43), because the OPC `F` —
derived on the same pipeline from a matrix-matched standard — already absorbs the systematic
matrix self-absorption bias as part of the relative-sensitivity factor. A second per-spectrum
self-absorption boost over-corrects. L4 stays an opt-in lever for the **OPC-free** regime only.

## What to promote (modules + API)

The benchmark already isolates the three primitives; promotion lifts them behind a clean,
physics-only, opt-in API. Nothing changes the default calibration-free path.

### 1. Line selection — `cflibs/inversion/physics/line_selection.py`
- Promote the **neutral-anchor wide-`E_k`** policy from `lever_l2_lines.select_l2_lines`
  (require a neutral anchor per element, spread `E_k`, drop ion-only-observed trace minors,
  avoid resonance/self-absorbed lines).
- API: extend the existing line-selection entry point with a `policy="neutral_anchor"` option (the
  current behavior stays the default). Pure DB + window inputs, no new deps.

### 2. Fixed / optimal temperature — `cflibs/inversion/solve/` (iterative solver)
- **Already shipped:** `IterativeCFLIBSSolver` accepts `fixed_temperature_K` (threaded through
  `run_constrained_solver`). The lever uses it directly — no new code, only documentation that the
  known-matrix mode supplies a calibrated `T*`.
- Promote `choose_optimal_T` (scan `T` to minimize a matrix-matched standard's composition error,
  Zhao 2018) as a **calibration-time** helper, e.g. `cflibs/inversion/physics/opc.py:choose_optimal_temperature(standard)`.

### 3. NEW OPC correction step — `cflibs/inversion/physics/opc.py`
A new physics module (the only genuinely new shipped code). Two phases:

```python
# --- calibration phase (offline, from certified standards) ---
@dataclass(frozen=True)
class OPCCalibration:
    robust_T_K: float                 # mean of selected standards' optimal T*
    F: dict[str, float]               # per-element relative-sensitivity factor (geomean)
    selected_standards: list[str]     # provenance
    conditioning_rule: str            # the exact a-priori gate used

def calibrate_opc(
    standards: Sequence[Standard],    # (spectrum, certified_composition) pairs
    *,
    cond_T_K: float = 9000.0,
    t_grid: Sequence[float] = ...,
) -> OPCCalibration: ...
    # 1. conditioning gate (in-sample only): converged, not degenerate,
    #    in-sample uncorrected RMSEP < threshold, no non-matrix element soaks closure
    # 2. robust_T = mean(optimal T* over selected)
    # 3. F = geomean_e( C_true_e / C_rec_e ) over selected, clamp-saturated values filtered

# --- inference phase (online, on unknowns) ---
def apply_opc(observations, calibration: OPCCalibration) -> None:
    # multiply each observation's intensity by F[element] in place, then solve at robust_T_K
```

- Wire `apply_opc` as an **optional pre-solve step** in the inversion pipeline
  (`cflibs/inversion/pipeline.py`): when an `OPCCalibration` is supplied, rescale the
  Boltzmann-plot intensities by `F` and pass `fixed_temperature_K=calibration.robust_T_K` to the
  solver. When absent, the pipeline is byte-for-byte the current calibration-free path.
- Persist `OPCCalibration` as JSON (it is tiny: one `T`, ~7 floats, provenance) via
  `cflibs/io/` so a matrix calibration is reusable across a DED build.

### 4. CLI / config surface
- `cflibs calibrate-opc <standards.yaml> --output steel_opc.json` — run the calibration phase.
- `cflibs invert spectrum.csv --opc steel_opc.json` — inference with a known-matrix calibration.
- Config: an `opc:` block (`calibration_path`, or inline `robust_T_K` + `F`) in the inversion YAML.

## Physics-only + gating plan

- **Physics-only constraint:** all of the above is pure NumPy + the existing Saha-Boltzmann solver
  — no `sklearn`/`torch`/`tensorflow`/`keras`/`flax`/`equinox`/`transformers`/`jax.nn`. The
  geometric mean, clamp, and conditioning gate are arithmetic. Ruff TID251 + the AST blocklist
  scanner already enforce this; the new `opc.py` adds no banned API.
- **Honesty is structural, not procedural:** `calibrate_opc` takes *only* standards; it cannot see
  unknowns, so held-out peeking is impossible by construction. Document and unit-test that
  `apply_opc` never reads a recovered composition (no positive feedback — the failure the
  2026-06-09 audit condemned).
- **Gates (in order) for the shipped change:**
  1. `ruff check` + `black --check` + `mypy` on the new/changed `cflibs/` files.
  2. **DED no-regression** (the real goal): `pytest tests/benchmarks/ded_precision -q` must stay
     green — OPC mode is opt-in, so the default DED path is unchanged; additionally verify a
     known-matrix Ti-6Al-4V OPC calibration does not regress the synthetic DED precision floor.
  3. **NIST parity** unchanged (OPC touches intensities post-extraction, not atomic data).
  4. **Real-steel gate** (`tests/benchmarks/real_steel/test_real_steel_gate.py`,
     `RMSEP_GUARD=11.0`): the promoted code, exercised through the shipped API, must reproduce the
     v2 held-out ≤ 11.0 wt% with the same a-priori conditioning rule (no held-out selection).
- **Roll-out:** land the benchmark v2 + this plan first (done here); then ship `opc.py` +
  pipeline wiring behind the opt-in flag in a follow-up PR gated on (1)-(4).

## Open items / next levers

- v2 Fe (20.66) and Mo/Cu (~10) are the next targets. Fe is OPC-corrected but still the largest
  residual; a CD-SB columnar-density treatment of the Fe major *inside* the OPC-calibrated solve
  (rather than the standalone L4 boost that double-corrects) is the principled next step.
- Stark `n_e` (L7) is independent of OPC and feeds Saha — combine once on-device Stark `n_e` lands.
- Multi-window / multi-standard `F(λ)` (wavelength-resolved OPC) for instruments with strong
  response curvature; the per-line `F` scaffold already exists (`per_line=True`).
