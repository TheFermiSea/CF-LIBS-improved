---
slug: physics-only-constraint
title: "The Physics-Only Constraint"
chapter: architecture
order: 80
status: stable
register: handbook
summary: >
  The hard constraint that shipped cflibs/ must import NO ML library, its canonical banned-API list,
  and its two-point enforcement — Ruff TID251 static rule + the AST blocklist scanner in
  cflibs/evolution/evaluator.py. Includes the enforcement history: softmax->softmax_closure,
  sklearn->scipy L-BFGS-B, ml/ package deleted.
tags: [physics-only, banned-api, ruff, tid251, ast-scanner, evolution, constraint]
updated: 2026-07-02
sources:
  - CLAUDE.md
  - docs/v4/overhaul/ARCHITECTURE.md
  - cflibs/evolution/evaluator.py
  - docs/Evolution_Framework.md
related: [module-map, change-taxonomy, two-backend-pipeline, adr-index]
code_refs:
  - cflibs/evolution/evaluator.py::scan_source
  - cflibs/inversion/physics/closure.py::softmax_closure
---

*[Wiki home](../index.md) · [Glossary](../glossary.md) · [Bibliography](../bibliography.md)*

# The Physics-Only Constraint

CF-LIBS is a **physics** library, and that is enforced, not aspirational. The shipped algorithm must
not import or use any machine-learning library. This page states the canonical banned-API list, the
two independent enforcement mechanisms, and the enforcement history that shows the constraint has real
teeth. It is an architectural invariant on par with the two-backend split.

## The hard constraint

> [!IMPORTANT] Shipped `cflibs/` must not import or use: `sklearn`, `torch`, `tensorflow`, `keras`,
> `flax`, `equinox`, `transformers`, `jax.nn`, or `jax.experimental.stax`.

Machine learning is allowed in **exactly one** place: `cflibs/evolution/`, the LLM-driven algorithm
optimization tooling — and even `cflibs/evolution/` itself may not import ML. The distinction is: the
*external* driver that *calls* `cflibs/evolution/` (running the LLM, scoring candidates) is ML tooling;
the shipped package that the driver invokes is a pure blocklist scanner + ES config with no ML imports.
This nuance was a documentation correction: CLAUDE.md now reads "ML is allowed only in the external
evolution driver that calls `cflibs/evolution/`", not "in `cflibs/evolution/`".

Why the constraint exists: CF-LIBS must be a *physics solver*, not a learned surrogate. A learned
regressor for $(T, n_e, \text{composition})$ is fast but instrument-specific and non-transferable; the
whole value proposition is standardless, physics-grounded quantification with calibrated uncertainty.
Physics-only alternatives to ML surrogates exist and are used — e.g. the physics-forward Bayesian
estimator [@aguilera2007]-style multi-element fitting under NumPyro
([abstractions](abstractions.md)).

## Two-point enforcement

The constraint is enforced at two independent levels, so a violation must slip past **both** to ship.

### 1. Ruff TID251 static rule

`pyproject.toml` bans these APIs from the entire shipped codebase via
`[tool.ruff.lint.flake8-tidy-imports.banned-api]`. This runs in the quality gate
(`ruff check cflibs/ tests/`) and in CI. Because `jitpipe` lives under `cflibs/`, TID251 covers it
automatically — the jit backend's smooth relaxations must be explicit `jnp` expressions or reuse the
shipped `softmax_closure`, never `jax.nn.softmax` (ADR-0004 D7).

### 2. AST blocklist scanner (`cflibs/evolution/evaluator.py`)

The scanner parses evolved candidate code and rejects any that violates the ban **before** physics
evaluation (fitness $= -\infty$). Unlike a regex, it walks the AST, so it catches the dynamic-import
evasions a string match would miss. `scan_source` flags:

| AST form | Example caught |
|----------|----------------|
| `ast.Import` | `import sklearn`, `import jax.nn`, `import torch as t` |
| `ast.ImportFrom` | `from sklearn.linear_model import LinearRegression` |
| dynamic import calls | `__import__("sklearn")`, `importlib.import_module("torch")` |
| `exec` / `eval` / `compile` | `exec("import torch")`, `eval("__import__('sklearn')")` |
| attribute-chain builtins | `builtins.exec(...)` |

The banned single-token entries forbid the top-level module (`import sklearn` etc.); dotted entries
(`jax.nn`, `jax.experimental.stax`) forbid the specific submodule while leaving core JAX legal. The
scanner is exposed as a CLI: `python -m cflibs.evolution candidate.py` (or `-` for stdin).

## Enforcement history (the constraint has bitten real code)

These are not hypothetical — each is a concrete change made to satisfy the constraint:

| Change | What happened |
|--------|---------------|
| `softmax` → `softmax_closure` | The simplex/closure projection was reimplemented as an explicit `jnp` expression `cflibs.inversion.physics.softmax_closure` instead of `jax.nn.softmax`, so the closure math ships without touching the banned `jax.nn` namespace. |
| `sklearn` → scipy L-BFGS-B | Optimization that had reached for `sklearn` was rewritten on `scipy.optimize` (L-BFGS-B) — a physics-legal optimizer. The joint solver is scipy L-BFGS-B driving JAX gradients, never an ML framework. |
| `ml/` package deleted | An entire `ml/` package was removed from the shipped tree; ML lives only in the external evolution driver, not in `cflibs/`. |

The tooling-choice corollary from ADR-0004 §6.1: even *JAX-ecosystem* optimizers are banned when they
drag in a banned dep — `jaxopt` is unmaintained and its successors (`optimistix`/`lineax`) are
`equinox`-dependent, so the codebase uses core-JAX `custom_root`/hand-written `custom_vjp` instead.

## Where the line is drawn

| Allowed | Banned |
|---------|--------|
| `numpy`, `scipy`, `scipy.optimize` (L-BFGS-B, least_squares) | `sklearn`, `torch`, `tensorflow`, `keras` |
| core `jax`, `jax.numpy`, `jax.scipy`, `jax.random`, `jax.lax` | `jax.nn`, `jax.experimental.stax`, `flax`, `equinox` |
| `numpyro` (Bayesian inference over the physics model) | `transformers` |
| explicit `jnp` sigmoids/softmax, `softmax_closure` | `jax.nn.sigmoid` / `jax.nn.softmax` |

NumPyro is allowed because it performs *Bayesian inference over the physics forward model* — it is not
a learned regressor; the physics is the model.

## See also

- [Two-backend pipeline](two-backend-pipeline.md) — TID251 covers `jitpipe` automatically (D7).
- [Change taxonomy](change-taxonomy.md) — physics-only hardening items are SAFE-NOW.
- [Module map](module-map.md) — `cflibs/evolution/` in the package inventory.
- `docs/Evolution_Framework.md` — the full physics-only specification.
