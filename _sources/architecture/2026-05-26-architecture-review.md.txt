# Architecture review — 2026-05-26

**Status:** session handoff for a Claude Agent Team. This document is the
authoritative brief; each candidate below is self-contained enough that an
agent can claim it without reading the prior transcript.

**Goal of the new session:** land the 5 architectural deepenings identified
below (or a subset, with explicit triage). Each candidate has a beads id,
a file:line target, a definition of done, and a validation command.

**Glossary** (use these terms exactly — do not substitute "service",
"component", "boundary"):

- **Module** = anything with an interface and an implementation
- **Interface** = everything a caller must know (types, invariants,
  ordering, error modes, perf characteristics)
- **Implementation** = body of code
- **Depth** = leverage at the interface (high = deep, low = shallow)
- **Seam** = location an interface lives
- **Adapter** = concrete satisfier of an interface
- **Leverage** = capability per unit of interface
- **Locality** = change/bug/knowledge concentrated in one module

**Source for full skill definitions:**
`/home/brian/.claude/skills/improve-codebase-architecture/LANGUAGE.md`.

---

## Quick-start for the agent team

```bash
# Working dir
cd /home/brian/code/CF-LIBS-improved
git checkout dev && git pull

# Tooling
ast-grep --lang python -p '<pattern>' cflibs/       # structural search
# Serena MCP via mcp__plugin_serena_serena__*       # symbol-level lookup
.venv/bin/python                                     # JAX, scipy, pyarrow
.venv/bin/python -m pytest <path> -q --timeout=60   # fast tests

# Cluster (V100S nodes vasp-01..03, ssh as briansquires@brians-...
# is NOT how to reach them — direct ssh to vasp-02, vasp-03)
ssh vasp-02 'squeue --states=all'                   # check before submit
ssh vasp-02 'cd ~brian/code/cflibs-bench && git pull && sbatch scripts/<x>.sh'

# Beads (don't use TodoWrite — use bd)
bd ready                # claimable
bd show <id>            # detail
bd update <id> --status=in_progress
bd close <id> --reason "..."

# Honor the user's cluster-coordination rule: another agent shares
# vasp-01..03. Do NOT pin --nodelist — let SLURM schedule.
```

**Cluster filesystem layout:**

- Repo on cluster: `~brian/code/cflibs-bench/` (note: different name from
  `CF-LIBS-improved` here). Sync via `git pull origin dev`.
- Data: `/cluster/shared/cf-libs-bench/data` (NFS-mounted)
- JAX compile cache: `~brian/jax-cache` (user-private — the shared one has
  uid skew and hangs jobs)
- Benchmark output: `~brian/code/cflibs-bench/output/<run>/`
- ai-proxy ↔ vasp-02 share `/home/brian/` via NFS, so files appear on
  both sides instantly.

**Bead JSONL is unstable across sessions:** `bd close` and `bd update`
get rolled back on the next `bd` invocation because the JSONL auto-imports
the on-disk state. Commit `.beads/issues.jsonl` immediately after bead
operations or the closes will revert.

**Worktree isolation does NOT work for the Agent tool** in this repo —
agents spawned with `isolation: worktree` wrote files to the main repo
working tree (their CWD ignored the worktree path). The work was correct,
just placed in the wrong directory. Either accept this and review the
main repo diff, or run agents without isolation and accept the
serialization.

---

## State at session end (2026-05-26)

### Current benchmark leaderboard (Phase 7, n=24, --quick)

```
hybrid_union                F1=0.619  R=0.625  P=0.679   ← king
alias_high_recall           F1=0.545  R=0.510  P=0.644   ← post-n3rf.2
hybrid_consensus_weighted   F1=0.491  R=0.422  P=0.701   ← post-jbfg.2
alias_v2                    F1=0.402  R=0.361  P=0.534   ← post-FISTA + n3rf.4
spectral_nnls               F1=0.399  R=0.490  P=0.374
alias                       F1=0.202  R=0.142  P=0.542
comb                        F1=0.014  R=0.042  P=0.008   ← s1qr.2 OPEN
```

Reference baseline (Phase 1b CPU, pre-fix): `alias_v2 F1=0.364`,
`alias_high_recall F1=0.083`. The leaderboard above is **post-all-fixes**.

### Open beads relevant to this review

- `s1qr.2` (P1) — comb wavelength_tolerance pivot. The amplitude-floor
  fix collides with the SNR=3 LIBS detection-limit convention; the fix
  needs to be shape-based (template-correlation discriminator), not
  amplitude. See `tests/inversion/test_comb_fp_reduction.py::test_tier2_strict_rejects_moderate_correlation_peaks`
  for the canary.
- `e7u3` (P2) — jaxopt Anderson acceleration for outer fixed-point.
- `nnci` (P3) — remove `object.__new__(SingleZoneLTEPlasma)` bypass.
- `w6bf` (P3) — comb default relaxation (benchmark required).
- `pp9i` (P3) — alias_boltzmann_r2_sweep workflow.
- `wph7` (P2) — spectral_nnls residual-variance cap inversion (behavior-
  changing, benchmark required).

### Recent epics with mostly-closed children

- `n3rf` (Post-ALIAS-fix followups): 3/4 closed
- `jbfg` (JAX/CPU parity + voting): 3/3 closed via current epic state
- `s1qr` (Cross-exam): 2/3 closed (s1qr.2 reopened for shape-based rework)

---

## Recommended sequencing

The 5 candidates are independent enough that 5 agents can claim them in
parallel, but #1 unblocks #2 (the preset table lives where the factory
lives). Suggested phasing:

**Wave 1 (parallel):**
- A1 — Candidate 1 (predictor-builder family)
- A3 — Candidate 3 (closure unification, ADR T1-3)
- A4 — Candidate 4 (PartitionFunction encapsulation, ADR B-P7)
- A5 — Candidate 5 (resonance filter seam)

**Wave 2 (after A1 lands):**
- A2 — Candidate 2 (ALIAS preset cocktails — depends on factory from #1)

**Validation gate:** after each merge, run Phase 8 cluster benchmark on
the n=24 cohort (24-spectrum Vrabel + Aalto + BHVO-2). Numbers must
not regress the current leaderboard.

---

## Candidate 1 · Collapse the predictor-builder family

**Strength:** Strong. **Beads:** new bead to file as `arch-predictors`.

### Files

- `cflibs/benchmark/unified.py:1270-1703`
- 6 builders × ~70 LOC = ~420 LOC of near-identical closure boilerplate:
  - `_build_alias_predictor`
  - `_build_alias_v2_predictor`
  - `_build_alias_high_recall_predictor`
  - `_build_hybrid_consensus_2of3_predictor`
  - `_build_hybrid_consensus_2of4_with_nnls_predictor`
  - `_build_hybrid_consensus_weighted_predictor`

### Problem

Each builder is a closure factory with identical outer shell. They
differ only in 2-5 kwargs passed to `ALIASIdentifier()` and (for
consensus variants) which sibling identifiers get composed.

**Cost evidence (deletion test passes):**
- Bead `jbfg.2` required `r2_gate_mode="adaptive_t"` + `relative_cl_per_ion_stage=True`
  inside **three** consensus builders (commit `06de07f`).
- Bead `n3rf.2` required the **same** two-kwarg fix inside the standalone
  `_build_alias_high_recall_predictor` (commit `b809d4b`).
- Bead `n3rf.4` required a third place to be aware of the same v2 cocktail.

Four discoveries of one logical decision. The "good cocktail" is implicit
knowledge scattered across docstrings and builder bodies.

### Solution

Extract a single factory:

```python
def _make_predictor(
    identifier_class,
    preset: str,                          # named cocktail, see #2
    voting: VotingConfig | None = None,    # for consensus variants
    sibling_identifiers: list[type] = (),  # for ensemble variants
    config_namer: Callable | None = None,
) -> Callable[..., ElementIdentificationResult]:
    """Single factory replacing 6 hand-rolled builders."""
```

The registry then becomes data:

```python
ID_WORKFLOW_PRESETS = {
    "alias": ("strict", None),
    "alias_v2": ("v2", None),
    "alias_high_recall": ("high_recall_v2", None),
    "hybrid_consensus_2of3": ("v2", BinaryVoting(2, ["alias", "comb", "correlation"])),
    "hybrid_consensus_2of4_with_nnls": ("v2", BinaryVoting(2, [..., "nnls"])),
    "hybrid_consensus_weighted": ("v2", WeightedVoting(weights={...}, threshold=0.40)),
}
```

### Definition of done

- 6 hand-rolled builders deleted; replaced by `_make_predictor` + registry
  table.
- `tests/benchmark/test_alias_sweep_workflows_registered.py` and the
  existing hybrid_consensus tests still pass.
- New unit test asserts that `_make_predictor(ALIASIdentifier, "v2")`
  produces a predictor functionally equivalent to the previous
  `_build_alias_v2_predictor` on at least one fixture spectrum.
- LOC delta ≥ -300 (net deletion).

### Validation

```bash
.venv/bin/python -m pytest tests/benchmark/ tests/test_hybrid_consensus_2of3.py -q --timeout=120
```

Then submit a Phase 8a cluster benchmark with the same 7-workflow list
as Phase 7 (`scripts/submit_post_alias_fix_benchmark.sh` after rebasing).
Numbers must match Phase 7 within 0.01 macro_F1 per workflow.

### Watch-outs

- `_build_alias_sweep_predictor_factory` at `unified.py:1094` is already
  the pattern to imitate. Don't reinvent.
- The `_jax_identifier_flags_for(cls)` helper at `unified.py:29` does
  x64 enablement as a side effect — preserve that semantics in the new
  factory (or extract it to a more honest place).

---

## Candidate 2 · Name the ALIAS configuration cocktails

**Strength:** Strong (after #1 lands). **Beads:** `arch-alias-presets`.

### Files

- `cflibs/inversion/identify/alias.py:820-920` — 30+ kwargs on `__init__`

### Problem

`ALIASIdentifier.__init__` takes 30 kwargs. Five coherent "cocktails"
have been discovered through bead-driven debugging. None of them are
named.

### The cocktails, discovered the hard way

| preset | r2_gate_mode | relative_cl_per_ion_stage | high_recall | found via |
|---|---|---|---|---|
| `strict` | `fixed` | False | False | baseline |
| `v2` | `adaptive_t` | True | False | PR #175 + #176 |
| `high_recall_v2` | `adaptive_t` | True | True | n3rf.2 |
| `consensus_voter` | `adaptive_t` | True | False | jbfg.2 |

Plus the JAX-flag bundle (`use_jax_*`) which is auto-injected.

### Solution

Define a preset table in `alias.py`:

```python
ALIAS_PRESETS: dict[str, dict[str, Any]] = {
    "strict": {"r2_gate_mode": "fixed", "relative_cl_per_ion_stage": False, ...},
    "v2":     {"r2_gate_mode": "adaptive_t", "relative_cl_per_ion_stage": True, ...},
    "high_recall_v2": {"r2_gate_mode": "adaptive_t", ..., "high_recall": True},
}

def alias_preset(name: str, **overrides) -> ALIASIdentifier:
    """Public constructor: select a known cocktail."""
    cfg = {**ALIAS_PRESETS[name], **overrides}
    return ALIASIdentifier(**cfg)
```

Existing `__init__` stays for experimentation; the predictor factory
from #1 calls `alias_preset()`.

### Definition of done

- `ALIAS_PRESETS` table lives in `alias.py` (or `alias_presets.py`).
- All predictor builders (or the unified factory from #1) consume presets
  by name.
- New unit test: each named preset round-trips through `__init__` and
  the resulting kwargs match the table.
- Docstring on `ALIASIdentifier.__init__` references `ALIAS_PRESETS` so
  the next maintainer doesn't have to discover the cocktails again.

### Validation

Same as #1, plus:
```bash
.venv/bin/python -m pytest tests/test_alias_unit.py -q --timeout=60
```

---

## Candidate 3 · Unify the two compositional closures

**Strength:** Strong. **ADR alignment:** ADR-0001 T1-3.
**Beads:** `arch-closure-unify`.

### Files

- `cflibs/inversion/physics/softmax_closure.py` — JAX, used by
  `joint_optimizer.py` + `coarse_to_fine.py`
- `cflibs/inversion/physics/closure.py` — numpy, used by
  `iterative.py` solver. Contains `ClosureEquation.apply_ilr` + PWLR
  (adaptive ridge).

### Problem

Two mechanisms implementing the same compositional-simplex math
(`sum C_i = 1`). A caller's choice depends on which solver was written
first, not on the LIBS regime they're modeling.

The detective on bead s1qr.3 documented that:
- softmax_closure has log-sum-exp stabilization but vanishing gradient ∝ C_i (locks small elements out)
- ILR has no vanishing gradient (singularity moved to ±∞) but is only
  available via the iterative solver
- PWLR is the adaptive-ridge variant in `closure.py` not used at all
  in JAX-side solvers

So JAX solvers can't use the closure mode that handles trace elements
best.

### Solution

```python
class ClosureStrategy(Protocol):
    name: str
    backend: Literal["jax", "numpy"]
    def apply(self, params: Array) -> Array: ...
    def gradient_check(self, c: Array) -> bool: ...  # for trace-element safety

# adapters
class SoftmaxClosure(ClosureStrategy): ...
class ILRClosure(ClosureStrategy): ...
class PWLRClosure(ClosureStrategy): ...
```

Solvers take a `ClosureStrategy` instance (not an import path).

### Definition of done

- `ClosureStrategy` protocol defined in `cflibs/inversion/physics/closure_strategy.py`.
- All three adapters (Softmax, ILR, PWLR) implemented.
- `joint_optimizer`, `coarse_to_fine`, `iterative` all take a
  `closure: ClosureStrategy` constructor kwarg (default Softmax for JAX,
  ILR for iterative, matching current behavior).
- Existing tests in `tests/test_softmax_closure.py` + iterative-solver
  tests still pass.
- New parity test: `SoftmaxClosure(theta).apply()` produces the same
  output as the pre-refactor `softmax_closure(theta)` to bit precision.

### Validation

```bash
.venv/bin/python -m pytest tests/test_softmax_closure.py \
  tests/test_solver.py tests/test_closed_form_solver.py \
  -q --timeout=120
```

### ADR alignment

ADR-0001 T1-3 (lax-while-iterative) already proposes pre-resolving
closure mode into a closed-over `closure_fn`. This deepening generalizes
T1-3 to all three solvers. Read `docs/adr/specs/T1-3-lax-while-iterative.md`
before starting.

---

## Candidate 4 · Encapsulate the partition function

**Strength:** Strong. **ADR alignment:** ADR-0001 B-P7.
**Beads:** `arch-partition-encapsulation` (also closes the partial fix
notes from s1qr.1).

### Files

- `cflibs/atomic/structures.py:118-145` — `PartitionFunction` dataclass
  exposes `(element, ionization_stage, coefficients, t_min, t_max, source)`
- 8 call sites consume the dataclass; 4 of them (post-fix `80bc338`) now
  pass `t_min`/`t_max` through; **4 still ignore the bounds**:
  - `cflibs/radiation/kernels.py:264`
  - `cflibs/manifold/generator.py:414-415`
  - `cflibs/manifold/batch_forward.py:50` (used at lines 256, 356, 457,
    613-614, 645)
  - `cflibs/plasma/anderson_solver.py:214`

### Problem

The `t_min/t_max` metadata exists in the dataclass but isn't part of the
contract — half the readers just take the coefficients. The deletion
test passes: encapsulating bounds inside a provider concentrates the
guard logic; today it's smeared across the partial-fix call sites.

### Solution

```python
class PartitionFunctionProvider(Protocol):
    def at(self, T_K: float | Array) -> float | Array: ...
    def valid_range(self) -> tuple[float, float]: ...
    @property
    def g0(self) -> float: ...

class PolynomialPartitionFunctionProvider:
    """Polynomial form with bounds clamping and g0 floor."""
    coefficients: tuple[float, ...]
    t_min: float
    t_max: float
    g0: float
    def at(self, T_K): ...  # clamp T to [t_min, t_max], floor at g0
```

`AtomicDatabase` vends sealed instances:

```python
provider = atomic_db.partition_function_for("Fe", 1)
U = provider.at(T_K)  # bounds + g0 enforced inside
```

The polynomial coefficients never leave the provider.

### Definition of done

- `PartitionFunctionProvider` protocol + concrete implementation in
  `cflibs/plasma/partition.py`.
- `AtomicDatabase.partition_function_for(element, stage)` returns a
  provider instance.
- All 8 call sites migrated to `provider.at(T)` instead of
  `polynomial_partition_function(T, coeffs, ...)`.
- The 4 batched-array call sites need a `BatchedPartitionFunctionProvider`
  with a `.at_batch(T_array, species_indices) -> Array` method — this
  is the trickier part. Schema-level work on snapshot containers
  (`TraceableAtomicSnapshot`, `AtomicDataContainer`) to carry
  per-species t_min/t_max/g0 arrays.
- `tests/test_partition_function_extrapolation.py` (5 existing tests
  + at least 3 new tests for the batched provider).

### Validation

```bash
.venv/bin/python -m pytest tests/test_partition_function_extrapolation.py \
  tests/test_plasma.py tests/test_atomic.py \
  tests/test_closed_form_solver.py tests/test_solver.py \
  tests/test_anderson_solver.py \
  -q --timeout=120
```

Phase 8 cluster benchmark; no alias_v2 macro_F1 regression vs Phase 7 (0.402).

### ADR alignment

ADR-0001 B-P7 specifies `RovibPartitionFunction` ABC with `at(T, **kwargs)`.
This is the canonical direction. Read `docs/adr/ADR-0001-RUNBOOK.md` §B-P7
before starting.

---

## Candidate 5 · Move the resonance filter behind the Boltzmann seam

**Strength:** Worth exploring. **Beads:** `arch-resonance-seam`.

### Files

- `cflibs/inversion/identify/alias.py:1528` — caller sets `cand["nnls_significant"]`
- `cflibs/inversion/identify/alias.py:1734` — caller threads it into the check
- `cflibs/inversion/identify/alias.py:2598-2750` — `_boltzmann_consistency_check`
  uses the flag to filter resonance lines

### Problem

Two checks (NNLS significance, Boltzmann linearity) couple through a
kwarg the caller has to thread. The caller must know both checks; the
flag rides through a stranger.

### Solution

Move the decision-to-filter inside `_boltzmann_consistency_check`:

```python
def _boltzmann_consistency_check(self, element, fused_lines, ..., candidate=None):
    # Internal seam: filter resonance lines if NNLS supports the candidate
    lines_to_fit = self._apply_resonance_filter(fused_lines, candidate)
    # ... existing logic ...

def _apply_resonance_filter(self, fused_lines, candidate):
    """Internal seam. Drops resonance lines when NNLS supports the
    candidate. Pre-scans to avoid stranding all-resonance elements
    (n3rf.4 lesson)."""
    if not candidate or not candidate.get("nnls_significant"): return fused_lines
    # ... pre-scan + filter logic ...
```

The caller passes only the candidate dict it already owns.

### Definition of done

- `_apply_resonance_filter` extracted as a private method.
- `_boltzmann_consistency_check` signature loses the `nnls_significant`
  kwarg (or marks it as deprecated for one release).
- All callers updated.
- The n3rf.4 pre-scan guard (don't strand all-resonance elements like
  Al I) is preserved.
- A new unit test exercises the filter directly on a synthetic candidate
  dict, not via the full `identify()` path.

### Validation

```bash
.venv/bin/python -m pytest tests/test_alias_unit.py tests/inversion/ -q --timeout=120
```

Phase 8 benchmark: alias_v2 macro_F1 must hold at 0.402 (Phase 7 baseline).

### Watch-outs

- The n3rf.4 fix specifically handles the case where all matched lines
  are resonance lines (Al I, no non-resonance alternatives). The
  pre-scan logic must be preserved verbatim in the extracted method.
  See commit `4794d04` for the exact pre-scan code.

---

## Cross-cutting context for the agent team

### Tooling preferences (the user has corrected twice — do NOT default to bash grep)

Ladder (try in order, fall back only when previous fails):

1. **Serena MCP** (LSP-grade symbol resolution) — `find_symbol`,
   `find_referencing_symbols`, `search_for_pattern`, `replace_symbol_body`,
   `rename_symbol`, `safe_delete_symbol`. Use this for any Python
   symbol-level question.
2. **colgrep skill** — semantic intent search ("where inversion line
   selection happens").
3. **ast-grep** at `/usr/local/bin/ast-grep` — structural Python search
   (e.g. `ast-grep --lang python -p 'def $F($$$): $$$ scipy.optimize.nnls($$$)'`).
4. **ripgrep / bash grep** — last resort, only for non-Python files
   (YAML, TOML, Markdown, Rust).

This is enforced by the CLAUDE.md at the repo root. Saved to memory at
`/home/brian/.claude/projects/-home-brian-code-CF-LIBS-improved/memory/feedback_use_code_intelligence_tools.md`.

### Physics-only constraint (HARD)

`cflibs/` must NOT import: `sklearn`, `torch`, `tensorflow`, `keras`,
`flax`, `equinox`, `transformers`, `jax.nn`, `jax.experimental.stax`.
Enforced by ruff TID251 (see `pyproject.toml`). ML is allowed only in
`cflibs/evolution/` (LLM-driven optimization tooling). Full spec:
bead `CF-LIBS-improved-3fy3`.

### JAX runtime gotchas

- `JAX_PLATFORMS=cpu` for local dev; `JAX_PLATFORMS=cuda` on cluster.
- `jax_enable_x64=True` is **required** for the identifier path (see
  `cflibs/benchmark/unified.py:_jax_identifier_flags_for` for the
  side-effect enablement). Without x64, FISTA NNLS at float32 silently
  drops 95% of coefficient precision on column-correlated matrices.
- `pytest tests/` forces CPU + x64 via `conftest.py` — but benchmark
  cluster runs need explicit enablement at the call site.

### Cluster benchmark conventions

- Phase numbering: this session's runs were Phase 1b through Phase 7.
  Continue with Phase 8 for the next validation cycle.
- Submit script: `scripts/submit_post_alias_fix_benchmark.sh` (edit
  `--experiment-label` and `ID_WORKFLOWS` per phase).
- Output dir: `~brian/code/cflibs-bench/output/post-alias-fix-d553/` —
  rsync down to local `output/post-alias-fix-d553-phase<N>/` after
  completion.
- Comparison: pull `id_summary.json` and diff against Phase 7 baseline.
- Honor the multi-agent cluster-coordination rule: DO NOT pin
  `--nodelist=`; let SLURM schedule.

### Beads workflow (this session learned the hard way)

- `bd close <id>` and `bd update <id> --status=X` work, BUT the JSONL
  rolls back on the next `bd` invocation due to auto-import.
- **Always `git add .beads/issues.jsonl && git commit` immediately after
  bead state changes.** Otherwise the next session's `bd ready` shows
  stale OPEN status.

### What didn't work this session

- The `isolation: worktree` flag on the Agent tool. Agents wrote to the
  main repo working tree instead of the assigned worktree. Either accept
  this and review the main repo diff, or serialize agents.
- The MAD-based 5σ noise floor for comb FP suppression (s1qr.2 v1). It
  collides with the LIBS detection-limit convention (SNR=3). The bug
  needs a shape-based fix, not amplitude. Reverted in commit `e382b05`.

### Useful commands the agent team will need

```bash
# Find all callers of a symbol (use Serena, not grep)
# In MCP: mcp__plugin_serena_serena__find_referencing_symbols(name_path="X", relative_path="cflibs/")

# Run focused subset of tests
.venv/bin/python -m pytest tests/<file>::<test> -q --timeout=60

# Lint + format
ruff check <files>
black --check <files>
black <files>  # apply

# Cluster submit
ssh vasp-02 'cd ~brian/code/cflibs-bench && git pull && sbatch scripts/submit_post_alias_fix_benchmark.sh'

# Cluster monitor
ssh vasp-02 'squeue -u brian; tail -20 ~brian/code/cflibs-bench/logs/slurm/post-alias-fix-<jobid>.out'

# Pull cluster results
rsync -aq vasp-02:/home/brian/code/cflibs-bench/output/post-alias-fix-d553/ output/post-alias-fix-d553-phase8/
```

---

## Recommendation for the team lead

Land Candidate 1 first. It's the highest-leverage, lowest-risk refactor,
and it unblocks #2 (the preset table lives where the factory lives).
Once Wave 1 lands, run Phase 8 cluster benchmark to confirm no
regression, then claim Wave 2.

The HTML version of this review (with diagrams) is at
`/tmp/architecture-review-20260526.html`. Open it for the visual
before/after diagrams that complement this prose.

---

## Provenance

- Generated 2026-05-26 by `/improve-codebase-architecture` skill
- Based on: two parallel `Explore`-agent walks of `cflibs/inversion/identify/`,
  `cflibs/benchmark/unified.py`, `cflibs/plasma/`, `cflibs/inversion/solve/`,
  `cflibs/inversion/physics/`, `cflibs/radiation/`.
- Friction signals come from this session's bead-driven fixes (jbfg.1, jbfg.2,
  n3rf.1, n3rf.2, n3rf.4, s1qr.1, s1qr.2, s1qr.3).
- Glossary terms per `/home/brian/.claude/skills/improve-codebase-architecture/LANGUAGE.md`.
