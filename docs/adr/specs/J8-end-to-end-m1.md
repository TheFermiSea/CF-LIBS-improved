# J8 Implementation Spec — End-to-End Single-Spectrum Graph + Scoreboard Flag → Milestone M1

**Bead:** J8 · **ADR:** [ADR-0004](../ADR-0004-jittable-inversion-pipeline.md) §8.1/§8.2 (M1), §5.4 · **Track:** spine · **Depends:** J1–J7 · **Estimated effort:** 5–8 pd / ~1.5 wk

## 1. Goals

- Compose J1–J7 into one jitted stage graph (`jitpipe/pipeline.py`): response multiply → calibrate → detect/identify → self-absorption → Stark → solve, mirroring the reference stage order of `run_pipeline` (`cflibs/inversion/pipeline.py:666`: `:697-706` response, `:709-733` detect_and_select via `:337`, `:761-795` Stark, `:797-818` solve).
- **Quality-flag failure semantics:** the reference raises `ValueError("No usable spectral lines detected for inversion.")` at zero observations (`pipeline.py:751-752`) and the board scores it all-FN (`benchmark/scoreboard.py:159-166`); the jit pipeline's masked equivalent (zero valid observations) must produce **the same all-FN record** — not NaN concentrations, not a crash. Kernels never raise; `host.py` interprets `failed` masks.
- Wire `--pipeline {reference,jit}` into the scoreboard: subparser (`cflibs/cli/main.py:1255-1290`) → `scoreboard_cmd` (`:1012`) → `run_scoreboard` (`scoreboard.py:274`) → `_score_spectrum` (`:132`). Reference path unchanged (`:153-158`). Everything score-defining is shared and untouched: candidate policy truth ∪ CONFOUNDER_ELEMENTS (`:142`), presence rule, seeded sampling (`:193-198`), micro-averaged confusion (`:214-217`), failure policy (`:159-166`).
- Emit the identical per-spectrum record schema (`scoreboard.py:143-185`) including `stage_timings_s` (single-spectrum mode: real timings; batched mode later in J9 reports batch_time/B with B recorded); `"pipeline": "jit"` recorded in board JSON + reproduce line (`:377`).

## 2. Scope notes

- Single-spectrum (`run_one`) end-to-end jit only; the batch axis is J9's job (the stage kernels are already vmap-tested individually).
- `host.py` unpacking to the reference result dataclasses is the integration surface: `WavelengthCalibrationResult`, `LineDetectionResult`, `CFLIBSResult` (`iterative.py:81`) — downstream consumers and the parity adapters see identical types.
- Stage-boundary outputs are first-class NamedTuples surfaced under `CFLIBS_JITPIPE_DEBUG` (ADR-0004 R5); this bead builds that switch.
- The divergence-ledger document is created here (`docs/jitpipe/divergence-ledger.md`) and seeded with any M1 adjudications.

## 3. Acceptance criteria — **Milestone M1 gate**

On the two M1 datasets — **bhvo2_chemcam** (4 spectra; headline; response-corrected) and **chemcam_calib** (240 spectra; known R≈2000; real spectral gaps exercising masking):

1. Element calls agree ≥95 % of spectra (Tier-D).
2. T within 2 %; n_e within 10 % (the Stark-stage tolerance dominates); concentrations rtol 5 % / atol 0.01.
3. 2-dataset scoreboard F1 delta ≥ −0.02 vs the frozen baseline protocol (`docs/benchmarks/SCOREBOARD-2026-06-10-baseline.md`, seed 20260610).
4. Failure-policy parity demonstrated: a fixture spectrum with zero usable lines produces the same all-FN scored record on both pipelines.
5. All per-stage Tier-K/S contracts green when the composed graph is stage-bisected via `parity.py` (no contract regressions from composition — catches inter-stage padding/mask handoff bugs).
6. End-to-end jit compiles once per bucket; no recompile on `PipelineParams` leaf changes (extends J0's AC-6 to the full graph).
7. CPU x64 full-graph parity test in CI (subset-sized per watchdog rules); GPU smoke on one vasp node.

## 4. Test plan

`tests/jitpipe/test_parity_pipeline.py` (end-to-end vs `run_pipeline` on M1 fixtures; failure-policy fixture; padding invariance at the graph level); `tests/jitpipe/test_scoreboard_dispatch.py` (flag threading, record-schema equality incl. `stage_timings_s` keys, `"pipeline"` field, reproduce line). Shadow-mode hook: both pipelines run on every scoreboard invocation with deltas logged (Tier-B harness, consumed by J12's ≥1-release shadow requirement).

## 5. Risks

- Inter-stage shape/mask handoffs (the composition is where mask bugs surface) — mitigated by AC-5 stage-bisect + graph-level padding invariance.
- The 23/76 sampled-spectrum hard-failure surface of the reference (`SCOREBOARD…md`) must map onto jit `failed` masks case-by-case; budget triage time for chemcam_calib's gap regions.
- Watchdog discipline: commit after each integration step; never run the full board inside a sub-agent (CLAUDE.md rules).

## 6. Dependencies / files

Depends J1–J7 (all stages). Enables J9, J12. Files: `cflibs/jitpipe/pipeline.py`, `host.py` completion, `parity.py` completion, `cflibs/cli/main.py` (flag), `cflibs/benchmark/scoreboard.py` (dispatch only — scoring logic untouched), tests, `docs/jitpipe/divergence-ledger.md`.
