## Summary

<!-- 1-3 bullets on what changed and why. -->

## Linked issues

<!-- Beads IDs (e.g. CF-LIBS-improved-3fy3) and/or GitHub issues. -->

## Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Refactor (no functional change)
- [ ] Documentation update
- [ ] Test or CI infrastructure

## Physics-only constraint

<!-- Required for any change to cflibs/ outside cflibs/evolution/. -->

- [ ] No new imports of `sklearn`, `torch`, `tensorflow`, `keras`, `flax`, `equinox`, `transformers`, `jax.nn`, or `jax.experimental.stax` in shipped code.
- [ ] If `cflibs/evolution/` is touched, candidates produced by it still pass `cflibs.evolution.evaluator.assert_physics_only`.
- [ ] If new ML imports were unavoidable, this PR also adds the corresponding `[tool.ruff.lint.per-file-ignores]` entry with a tracking bead-ID comment.

## Test plan

<!-- Concrete commands run, results observed. Be specific about what was tested AND what was not. -->

- [ ] `ruff check cflibs/ tests/`
- [ ] `black --check cflibs/`
- [ ] Touched-area pytest slice
- [ ] Full pytest suite (note pre-existing failures separately if any)

## Documentation

<!-- Did this change require updates to CLAUDE.md / AGENTS.md / docs/? -->

- [ ] CLAUDE.md / AGENTS.md updated if architecture or conventions changed
- [ ] `docs/` updated if user-facing APIs or workflows changed
- [ ] Inline docstrings updated where signatures changed

## Notes for reviewer

<!-- Anything else: design tradeoffs, open questions, parts intentionally left out. -->
