# CF-LIBS ast-grep invariant rule pack

Structural (AST-level) enforcement of the invariants this project spent weeks
establishing: the physics-only constraint, the silent-fallback audit, atomic-DB
immutability, and test-pollution hygiene. These rules complement (do not
replace) the ruff TID251 ban and the `cflibs/evolution/` AST blocklist scanner —
they add coverage ruff cannot express (except-body shape, string-literal schema
mutations, module-level side-effects) and act as a fast backstop.

## Running

```bash
ast-grep test                     # run the rule-pack self-tests (valid/invalid)
ast-grep scan                     # run all rules over the repo, human-readable
ast-grep scan --json=stream       # machine-readable
python scripts/ast_grep_gate.py   # CI gate: fail on any error-severity hit
```

Config lives in `sgconfig.yml` (`ruleDirs: rules`, `testConfigs: rule-tests`).

## The gate / advisory ratchet

`scripts/ast_grep_gate.py` implements a ratchet:

- **`error`**-severity rules are the **GATE**. They are *clean on the current
  tree*, so the gate exits 0 today. Any new match fails the gate.
- **`warning` / `hint`**-severity rules are **ADVISORY**. They carry legitimate
  pre-existing debt; their counts are reported but never fail the build, so CI
  does not break on day one.

Promote an advisory rule to a gate by cleaning its hits and flipping its
`severity` to `error`. CI wiring: `.github/workflows/ci.yml` runs `ast-grep
test` + the gate as a **continue-on-error advisory step** on Python 3.12.

## Rules

| id | sev | enforces | current hits |
|----|-----|----------|--------------|
| `physics-only-no-ml-imports` | error (gate) | no `sklearn/torch/tensorflow/keras/flax/equinox/transformers` import under `cflibs/` (except `cflibs/evolution/`) | **0** |
| `physics-only-no-jax-nn` | error (gate) | no `jax.nn.*` / `from jax import nn` / `jax.experimental.stax` under `cflibs/` (except `evolution/`) | **0** |
| `no-jax-clear-caches-in-tests` | error (gate) | no `jax.clear_caches()` in `tests/` (process-global cache wipe) | **0** |
| `no-default-physics-constants` | warning | no `ioniz*/partition*.get(key, <number>)` — the "IP=15.0" class | 2 |
| `no-silent-fallback-return` | warning | `except` handler that RETURNS a value (launders a default) under `cflibs/inversion/` | 40 |
| `no-except-pass` | warning | `except ...: pass` (swallowed failure) under `cflibs/` | 27 |
| `no-print-in-shipped` | warning | `print(...)` under `cflibs/` (except `cli/`) — use the logger | 11 |
| `no-module-level-setenv-in-tests` | warning | module-level `os.environ.setdefault(...)` in `tests/` | 30 |
| `no-stray-schema-mutation` | warning | `ALTER/DROP TABLE` literal outside sanctioned migration/builders | 5 |
| `standard-db-read-only` | hint | `sqlite3.connect(...)` without `mode=ro` under `scripts/` or `cflibs/atomic/` | 43 |
| `strict-mode-symmetric-assign` | hint | `if strict: VAR=A / else: VAR=B` (strict must raise, not recompute) | 0 |

### Why each rule (pointers)

- **physics-only-\*** — the HARD physics-only constraint (`CLAUDE.md`; bead
  `CF-LIBS-improved-3fy3`). Structural backstop to ruff TID251 and the evolution
  blocklist. ML lives ONLY under `cflibs/evolution/`.
- **no-default-physics-constants** — the "IP = 15.0" class from
  `docs/research/physics-first-principles-audit.md` (Cluster A, atomic data). A
  guessed ionization potential / partition value silently biases T, n_e and
  every concentration. Deliberately scoped to `ioniz*/partition*` dicts;
  abundance/concentration numeric defaults (`0.0` = absent, `1.0` = identity)
  are physically correct and excluded to avoid false positives.
- **no-silent-fallback-return / no-except-pass** — the ~102-site silent-fallback
  audit (`docs/research/physics-first-principles-audit.md` + the strict-mode /
  no-fallback-exploratory work). A caught exception that returns a warm-start /
  default reports a fabricated physics result.
- **no-print-in-shipped** — library code embedded in pipelines / MCP tools /
  cluster jobs must route through `logging`, not raw stdout.
- **no-module-level-setenv-in-tests / no-jax-clear-caches-in-tests** — project
  memory: `reference_test_pollution_collection_time` (a module-level
  `setdefault` leaked session-wide → 3 full-suite-only failures) and the CLAUDE.md
  JAX test-pollution rule (`jax.clear_caches()` is a global wipe).
- **standard-db-read-only / no-stray-schema-mutation** — standard-atomic-DB
  immutability mandate (a past ASD corruption cost months; memories
  `reference_atomic_db_incomplete_nist_ingest`, `project_atomic_db_complete_reset`).
  The gold-standard `ASD_da/*.db` must be opened read-only; schema mutations
  belong in the one sanctioned migration site or a `scripts/build_*` builder.

## Annotating a legitimate exception

- **Silent-fallback / except-pass**: add a trailing `# fallback-ok: <reason>`
  comment inside the `except` handler. Example:
  ```python
  try:
      probe_optional_capability()
  except Exception:
      return None  # fallback-ok: optional dependency probe
  ```
- **standard-db-read-only**: switch to
  `sqlite3.connect(f"file:{path}?mode=ro", uri=True)`, or leave it if the
  connection genuinely builds an overlay / derived DB (advisory only).
- **no-stray-schema-mutation**: legitimate derived-DB builders should move under
  `scripts/build_*.py` (auto-exempt) or be added to that rule's `ignores`.
- **no-print-in-shipped**: replace with `logger.info(...)` / `logger.debug(...)`.

## Current-tree triage (advisory hits are pre-existing debt, out of scope to fix here)

Genuine TODOs worth follow-up:

- **`no-default-physics-constants` (2, genuine — IP default class):**
  - `cflibs/inversion/physics/quality.py:348` — `ionization_potentials.get(obs.element, 15.0)`
  - `cflibs/inversion/runtime/streaming.py:1305` — `_ionization_potentials.get(element, 10.0)`
  Both substitute a fabricated IP when the atomic data is missing (should raise
  or record). File as follow-up beads.
- **`no-silent-fallback-return` (40)** and **`no-except-pass` (27)** — the
  silent-fallback debt across `cflibs/inversion/**` and `cflibs/**`. Each site
  should be reviewed: raise/record, log-and-continue, or annotate `# fallback-ok`.
- **`no-print-in-shipped` (11)** — genuinely dirty: `cflibs/hpc/slurm.py`
  (dry-run banners), `cflibs/benchmark/{checkpoint,synthetic_eval,composition_eval,unified}.py`
  (harness progress), `cflibs/atomic/database_generator.py` (generation progress).
  Convert to logger calls. (Note: docstring `print(...)` usage examples, e.g.
  `cflibs/inversion/physics/uncertainty.py:32`, are correctly NOT flagged — they
  are string literals, not calls.)
- **`no-module-level-setenv-in-tests` (30)** — mostly the benign
  `os.environ.setdefault("JAX_PLATFORMS", "cpu")` idiom (identical to conftest)
  plus `tests/conftest.py:20`. Benign but structurally the pollution pattern;
  prefer consolidating into a single conftest fixture.
- **`no-stray-schema-mutation` (5)** — `scripts/ingest_nist_lines_full.py`,
  `scripts/ingest_dump_levels.py`, `scripts/rebuild_lines_from_dump.py`. These
  are legitimate derived-DB / ingest builders; either rename to `build_*` or add
  to the rule's `ignores`.
- **`standard-db-read-only` (43)** — ingest/build/migration scripts opening the
  DB read-write to construct it. Legitimate for builders; advisory flag to keep
  read-only opens the default for *consumers*.

## The `strict-mode-symmetric-assign` residual gap (honest limitation)

ast-grep matches on syntax, not data/control flow, so it **cannot** track
variable identity across arbitrary control flow. This rule only catches the
**trivial adjacent case**:

```python
if strict:
    VAR = A          # single statement
else:
    VAR = B          # same VAR, single statement
```

It does **NOT** catch (all are real forms of the anti-pattern it targets):

- multi-statement `if`/`else` branches;
- augmented assignments (`VAR += ...`);
- a strict branch that assigns and a *fallthrough* (post-`if`) assignment;
- tuple / multiple-target unpacking (`a, b = ...`);
- a strict branch that calls a helper which internally computes the value;
- `elif` chains and nested conditionals.

Treat a clean scan as **necessary but not sufficient**. The robust check for
"strict must raise-or-record, never recompute" remains a code-review / targeted
unit-test responsibility. The rule ships at `hint` severity for this reason and
currently has 0 hits on the tree.

## Adding a rule

1. Write `rules/<id>.yml` — `id`, `language: python`, `severity`, a `message`
   (short, WHY + doc/memory pointer), an optional `note` (longer), the `rule:`
   matcher, and `files:`/`ignores:` scoping.
2. Write `rule-tests/<id>-test.yml` with `valid:` (must-NOT-match) and
   `invalid:` (MUST-match) snippets. Cover the false-positive shapes you
   deliberately excluded.
3. `ast-grep test --update-all` to generate the `__snapshots__/` baseline, then
   `ast-grep test` to confirm green.
4. `ast-grep scan` and triage every hit: real bug (record as TODO here),
   false positive (tighten the matcher / add a metavariable `regex` constraint),
   or annotate-worthy.
5. Start at `warning`/`hint`; promote to `error` (gating) only once the current
   tree is clean for that rule.
