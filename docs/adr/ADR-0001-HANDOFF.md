# ADR-0001 Handoff Prompt

> Paste the block below verbatim into the first message of the next session.
> Self-contained — references only persistent artifacts (bd memories, files
> on `dev`), not any prior conversation.

---

```text
You are picking up CF-LIBS-improved after ADR-0001 just landed on dev at commit
64d5e77 (2026-05-13). Working dir: /home/brian/code/CF-LIBS-improved.

# Context recovery (do this first)

1. Run `bd prime` and `bd memories adr-0001` — there are two persistent memory
   entries you need to read: `adr-0001-landed` (final state) and
   `adr-0001-integration-branch` (waves/sequencing reference).
2. Read `docs/adr/ADR-0001-radis-jaxrts-pattern-survey.md` (the survey) and
   `docs/adr/ADR-0001-RUNBOOK.md` (the workflow protocol). Per-bead specs in
   `docs/adr/specs/T1-{1..6}-*.md` describe what was actually built.
3. Baseline files in `docs/adr/`: `adr-0001-baseline.txt`, `adr-0001-baseline-bench.json`
   — captured on origin/dev@2f06dd9 before any T1 work landed. Use these to
   anchor any regression check. (Moved from repo root in the 2026-06 cleanliness
   sweep; the 608 KB `.adr-0001-baseline-cov.txt` coverage dump was deleted.)

# What landed (six Tier-1 beads, all closed)

- 5oar T1-1: shared jit_if_available + vmap_if_available decorators in
  cflibs/core/jax_runtime, JaxMemoryPolicy frozen dataclass + jax_policy()
  factory, AtomicSnapshot frozen dataclass + AtomicDatabase.snapshot()
  builder, minimal pytree registration on SingleZoneLTEPlasma +
  InstrumentModel. Absorbed T2-2 + T2-3 entirely.
- 14p6 T1-3: jax.lax.while_loop in IterativeCFLIBSSolver.solve(); feature
  flag CFLIBS_USE_LAX_WHILE_LOOP defaults off; rtol=1e-5 across six closure
  modes; IterativeCFLIBSSolverJax is now a DeprecationWarning alias.
- e5o8 T1-4: cflibs/radiation/ldm.py — LDM/DIT Gaussian broadening
  (van den Bekerom & Pannier 2021), BroadeningMode.LDM_GAUSSIAN, 19× speedup
  at N_lines=1500, default OFF pending real-data validation.
- swgm T1-2: cflibs/radiation/kernels.py::forward_model — unified kernel
  serving SpectrumModel + manifold + Bayesian (via bridge); machine-precision
  parity across LEGACY/NIST_PARITY/PHYSICAL_DOPPLER.
- ke4z T1-5: forward_model_chunked + overlap_and_add in kernels.py;
  auto_nstitch + _split_wavelength_grid in cflibs/radiation/host.py;
  nstitch=1 default keeps bit-identical existing output.
- 0mor T1-6: cflibs/inversion/solve/bayesian.py 3344-LOC monolith decomposed
  into bayesian/{priors,atomic,forward,results,samplers,two_zone,
  diagnostics}.py (each <800 LOC) + new cflibs/inversion/forward_models/
  registry. 20+ legacy import names preserved via __init__.py shim.

# §6 gate results when ADR-0001 merged

- pytest: 1891 pass / 16 fail / 90 skip (baseline 1829/21/90 → +62 pass)
- Physics-only AST scan clean
- Manifold round-trip working via examples/manifold_smoke_config.yaml (27
  spectra in ~1s); full manifold_config_example.yaml needs GPU (400k spectra)
- ruff 34 errors (improved from 43 baseline); mypy 0; black 15 dirty files

# Next-up beads (P2 priority, ranked by impact × inverse-effort)

Run `bd ready` to see the live queue, but the deferred ADR-0001 follow-ups are:

1. T1-5 follow-up: add line_mask param to forward_model, retire
   _forward_model_per_chunk duplicate, hoist Saha+sigma OUT of lax.scan body.
   Reuse + efficiency both flag this; 20-40% chunked wall-time savings.
2. T1-1 follow-up: complete host.py/kernels.py file split per spec §3 for
   cflibs/radiation/, plasma/, instrument/, manifold/. Largely cosmetic
   relocation; infrastructure layer already landed.
3. T1-6 follow-up: kernel migration of BayesianForwardModel._compute_spectrum
   to call forward_model directly (currently uses _atomic_data_arrays_from
   _snapshot bridge).
4. T1-5 follow-up: ChunkPlan frozen dataclass collapsing 13 kwargs on
   forward_model_chunked.
5. T1-5 follow-up: vectorize overlap_and_add fori_loop into vmap+sum.
6. T1-4 follow-up: ManifoldConfig flag for LDM_GAUSSIAN (production CLI
   still uses PHYSICAL_DOPPLER until real-data validation).
7. Hygiene: black --check on 16 remaining files in cflibs/ (do AFTER any
   in-flight feature work to avoid merge-conflict).

# Hard constraints (non-negotiable)

- Physics-only ban: NO sklearn / torch / tensorflow / keras / flax / equinox
  / transformers / jax.nn / jax.experimental.stax in cflibs/. Allowed only
  under cflibs/evolution/. Enforced by ruff TID251 + cflibs.evolution AST
  scanner. See bd memory cf-libs-hard-project-constraint-the-final-algorithm.
- Beads workflow: use `bd create` / `bd close` for all task tracking. NEVER
  TodoWrite or TaskCreate. NEVER `bd edit` (opens vim and blocks).
- Branch policy: feature work lives on feat/* sub-branches that merge to dev
  via --no-ff. Never push directly to main.
- Session-close protocol: git status → git add → git commit → git push. Work
  is not done until pushed.

# Gotchas learned during ADR-0001

- The `bd update --status inreview` value isn't a built-in status; bd's
  built-ins are open/in_progress/blocked/deferred/closed/pinned/hooked.
- Some beads have a [dispatch-guard] auto-block that flags issues missing
  on-disk targets; bypass with `bd update <id> --status open`.
- The .venv is shared across worktrees via symlink per runbook §4.2. NEVER
  pip install inside a worktree.
- YAML 1.1 parses '1e16' (no explicit exponent sign) as a string. Use
  '1.0e+16' in new configs or coerce via float() in loaders.
- pytest-cov is missing from the .venv in some setups; --cov flags may
  fail. Use plain pytest for the gate sweep if needed.
- The manifold CLI on dev was broken at multiple stacked layers before
  ADR-0001 fixed them; use examples/manifold_smoke_config.yaml for any
  smoke verification (production config needs GPU).

# What I recommend you do first

Start with `bd ready`, pick the highest-impact P2 from the list above, file
or claim it, and follow the runbook §3 spawn template if you parallelize
across worktrees. Otherwise just work on dev with normal feat/<bead-id>
branches.
```
