# M7 — LTE Refuse-to-Report (Phase 1): Gate Findings

**Date:** 2026-06-20 · **Branch:** `v4/m7-refuse-to-report` · **Lever 6**, accuracy-first roadmap §4/§6/§7.
**Solver host:** legacy iterative path (`run_pipeline`, geological preset, `stark_ne=True`) — the 2×-more-accurate path the roadmap mandates for accuracy physics.

## What landed (3 commits, all gated / additive)

1. **Wire the dead `QualityAssessor.assess` into the solver** (`20ba121c`). The Cristoforetti
   multi-check (Saha-Boltzmann consistency, inter-element T scatter, closure → `quality_flag`) had
   **zero production callers**. Now called inside the shared `_assemble_quality_metrics`, so BOTH
   solve paths (python + lax) emit `quality_flag` / `saha_boltzmann_consistency` /
   `inter_element_t_std_frac`. Added `CFLIBSResult.overall_reliable = {McWhirter satisfied} AND
   {quality_flag ∈ (excellent, good, acceptable)}`. CLI refuse-to-report escalation gated behind
   `CFLIBS_REFUSE_TO_REPORT` (default OFF == byte-identical legacy CLI).
2. **Resonance-line McWhirter δE** (`3c4fc551`), gated behind `CFLIBS_MCWHIRTER_RESONANCE_DE`
   (default OFF). Replaces the physically-incorrect `max(E_k)` δE.
3. **Scoreboard reliability surface** (`132ce225`): per-spectrum `quality_flag`, `overall_reliable`,
   `lte_mcwhirter_satisfied`, `lte_n_e_required_cm3` for conditional-RMSE / confusion analysis.

## The δE correction (physics)

The roadmap flagged the McWhirter δE as a bug. The *current* shipped code (post-#300–305) was
**already** off the roadmap's stale premise: it used `max(E_k)` (highest observed upper level), not
the adjacent-observed gap. Both are wrong:

- `max(E_k)` (~6–8 eV) **over-inflates** the cubic n_e floor → **false-rejects valid LTE plasmas**.
- The largest *adjacent level* gap lands on low-lying **same-parity (forbidden)** terms that don't
  stress LTE (Fe I → 0.74 eV) → **too lax**.

The physically-correct McWhirter δE is the **resonance-line energy** — the dipole-allowed ground-state
transition whose fast radiative decay collisions must overcome (Cristoforetti et al. 2010,
Spectrochim. Acta B 65, 86-95; confirmed via the curated NotebookLM literature corpus). Implemented as
the strongest (max A_ki) `is_resonance` line's upper energy per species, max over species present.

**Oracle (executed, real DB `ASD_da/libs_production.db`)** — resonance δE vs literature resonance energies:

| Species | resonance δE (DB) | literature | n_e floor: `max(E_k)` → resonance |
|---|---|---|---|
| Ca I | 2.933 eV | 2.93 | 3.5e16 → 1.1e15 |
| Na I | 2.104 eV | 2.10 | 2.1e16 → 1.5e15 |
| Mg I | 4.346 eV | 4.35 | 6.8e16 → 3.2e15 |
| Si I | 4.920 eV | 4.93 | 7.8e16 → 1.8e15 |
| Fe I | 4.991 eV | (strongest resonance) | 6.8e16 → 6.4e13 (adjacent-gap trap would give 0.74 eV) |

**Real-world false-reject confirmed:** a scoreboard run surfaced the legacy `max(E_k)` δE flagging a
healthy **n_e = 2.7×10¹⁷** plasma as non-LTE (floor 7.0×10¹⁷, ratio 0.38). The resonance δE floor for
the same plasma is ~6×10¹⁵ → correctly PASSes. This is the exact failure the sub-lever removes.

## Verification gate (executed, not asserted)

| Check | Kind | Result |
|---|---|---|
| **Non-regression, assess-wiring** (baseline `d784d33c` vs branch, default env, aalto 30 spectra) | benchmark | **PASS** — composition/T/n_e **bit-identical** (0/30 differ) |
| **Non-regression, δE flag** (branch δE-off vs δE-on, aalto 30 spectra) | benchmark | **PASS** — bit-identical; only `lte_n_e_required_cm3` / `lte_n_e_ratio` change |
| **δE physics** (resonance δE vs literature resonance energies) | oracle | **PASS** — Ca/Na/Mg/Si exact |
| **`overall_reliable` logic** (McWhirter AND quality_flag≥acceptable, 5 cases) | cross-method | **PASS** — unit-tested |
| **Defensive annotation never raises** (mock/edge DB → `quality_flag='unknown'`) | limiting case | **PASS** |
| **Lax/python key-set parity** preserved after adding assess keys | consistency | **PASS** |
| **CLI gate**: OFF == legacy; ON refuses below-acceptable; hard gates fire regardless | benchmark | **PASS** |

18 unit tests in `tests/inversion/solve/test_refuse_to_report.py`; full regression set
(parity / LTE / quality / CLI / scoreboard) green.

## Deferred (follow-up bead)

The roadmap's **synthetic false-reject confusion matrix** and **conditional-RMSE ≤ unconditional-RMSE**
quantification require a synthetic corpus that recovers cleanly through the pipeline with composition
truth. A quick 24-spectrum probe corpus had detection/recovery issues (18/24 fail line detection — a
corpus-generation tuning matter, not an M7-code matter), and the existing real corpora don't fill the
gap (aalto carries no composition truth; bhvo2/SuperCam are holdout/placeholder). The scoreboard now
**captures the per-spectrum reliability surface** needed for this analysis once a clean corpus exists.
**Status:** the safety property (no composition regression) and the physics (δE) are proven; the
"proves-it-helps" conditional-RMSE quantification is the remaining gate step.

## Flag semantics (defaults preserve legacy exactly)

- `CFLIBS_REFUSE_TO_REPORT` (CLI only; default OFF): when ON, CLI marks RESULT UNRELIABLE if
  `overall_reliable` is False.
- `CFLIBS_MCWHIRTER_RESONANCE_DE` (solver; default OFF): when ON, McWhirter δE = resonance-line energy.
- assess-wiring + `overall_reliable` are **always computed** (pure annotation; never alter
  T/n_e/composition — proven bit-identical above).
