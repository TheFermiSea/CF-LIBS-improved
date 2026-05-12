# JAX NNLS Port — Cloud Consultation Synthesis

Date: 2026-05-12
Branch: `feat/jax-nnls-model-selection`
Issue: `CF-LIBS-improved-wstf`

## Question

Port `scipy.optimize.nnls` (Lawson–Hanson active-set NNLS) to JAX for
GPU batching. Targets:

- `cflibs/inversion/identify/spectral_nnls.py` — single call per identify()
- `cflibs/inversion/identify/model_selection.py` — backward elimination
  inside `bic_prune_elements` does ~`n_elements` re-solves on row-masked
  subsets of the basis matrix.

Matrix sizes: `A: (n_pixels=1000-10000, n_components=30-100)`.

`jaxopt` is **not** installed in the project venv, so the port is
hand-rolled.

## Codex (gpt-5.3-codex) recommendations

1. **Algorithm: projected gradient (PGD)** in JAX is strictly better than
   active-set (Lawson–Hanson). Active-set has irregular branching that
   defeats `jit`/`vmap`; first-order methods are jit-friendly and dominate
   for `n_components ≤ 100`.
2. **Use Gram form**: precompute `G = A^T A`, `c = A^T b`, run PGD on
   the (n_components, n_components) system. Cheap when `n_components` is
   small; matmul `(n_pix, n_comp) @ (n_comp, n_pix)` is the only
   dependence on n_pix.
3. **Step size**: `alpha = 1/L`, `L = max(jnp.sum(jnp.abs(G), axis=-1))`
   (cheap row-sum upper bound) OR `jnp.linalg.norm(G, 2)` (exact spectral
   norm, ~microseconds at this size).
4. **Batched subsets** (BIC backward elim): pre-build a `(B, m)` row mask
   `W`, then
   ```python
   G = jnp.einsum('bm,mi,mj->bij', W, A, A)   # (B, n, n)
   c = jnp.einsum('bm,mi,m->bi',  W, A, b)    # (B, n)
   ```
   and `vmap(nnls_pgd_gram)` over the leading batch axis. Static shapes
   throughout, no dynamic_slice.
5. **Convergence**: match on KKT/residual norm, **not** the solution
   vector — Lawson–Hanson and PGD can converge to different points on
   rank-deficient problems (multiple minimizers).
   - Primal residual: `||Ax-b||/||b|| ~ 1e-6 .. 1e-4`
   - Projected gradient: `||min(x, grad)||_inf ~ 1e-6 .. 1e-5` (float64)
6. **Add tiny ridge** `G + λI`, `λ ~ 1e-10..1e-6` for stability on
   ill-conditioned basis matrices.
7. **Warm-start** is very effective for the BIC inner loop.

## Opus 4.6 recommendations

1. **Mask pattern for BIC** (validated):
   ```python
   def nnls_masked(basis, spectrum, mask):
       A = (basis * mask[:, None]).T  # zero out masked rows
       G = A.T @ A
       c = A.T @ spectrum
       return nnls_solver(G, c)
   ```
   Each variant builds its own small `G` — cheap. `vmap` cleanly.
2. **Precision**: form `G` and `c` in float64 even if FISTA runs in
   float32. `cond(G) = cond(A)^2` so float32 loses most digits on
   correlated bases (spectroscopy commonly hits `cond(A) ~ 10^3-10^4`).
   At our sizes the matmul is not memory-bound — just use float64
   throughout.
3. **Accuracy expectation**: at ~300 PGD/FISTA iterations on well-
   conditioned problems, `||x_fista - x_exact|| / ||x_exact|| ~ 1e-4
   .. 1e-5`. Float64. For BIC selection — coefficient changes by ≤1e-4
   are dwarfed by `k * ln(n)` penalty term, so this precision is
   sufficient.
4. **Step size**: skip power iteration. `jnp.linalg.norm(G)` (operator
   norm via SVD) is exact and microseconds at n=100.
5. **KKT quality metric**:
   ```python
   grad = G @ x - c
   kkt_viol = jnp.max(jnp.where(x > 0, jnp.abs(grad), jnp.maximum(grad, 0.0)))
   ```

## Gemini

Rate-limited (RESOURCE_EXHAUSTED). Skipped — Codex + Opus already
converged on the same design.

## Design decisions

1. **Algorithm**: FISTA on Gram form, 300 default iterations, fixed-iter
   `lax.scan` (no early exit — preserves jit-ability).
2. **Step size**: `1 / jnp.linalg.norm(G)` (spectral norm). Cheap, exact,
   jit-friendly.
3. **Precision**: float64 throughout (Opus rationale: small matrices,
   matmul is not the bottleneck).
4. **Single-spectrum API**: `nnls_jax(A, b, max_iter=300) -> (x, info)`.
   `info` carries final KKT violation + residual norm for diagnostics.
5. **Batched API**: `nnls_jax_batch(A, b, masks, max_iter=300) -> X`
   where `masks: (B, n_components)` are boolean row-masks. Builds
   per-batch `G` and `c` via `einsum` and vmaps the inner solver.
6. **Numerical-agreement target**: rtol=1e-4 on coefficients *and*
   rtol=1e-5 on `||Ax-b||` residual norm. Per Codex, coefficient match
   can fail on rank-deficient problems; residual match is the real
   contract.
7. **Opt-in flags** (mirroring PR #118):
   - `SpectralNNLSIdentifier.__init__(..., use_jax_nnls: bool = False)`
   - `bic_prune_elements(..., use_jax_nnls: bool = False)` kwarg.
8. **Tiny ridge**: `λ = 1e-12` for the FISTA iteration to handle
   degenerate Gram matrices without altering well-conditioned solves.

## Out of scope

- Adding `jaxopt` as a hard dependency — file a follow-up bd issue if
  we want it later for the (more accurate) bounded-LBFGS NNLS.
- True early-exit via dynamic iter count — defeats jit; revisit if
  profile shows FISTA is the bottleneck.
- Active-set Lawson–Hanson implementation — branching kills `vmap`;
  see Codex (1).
