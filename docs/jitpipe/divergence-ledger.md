# jitpipe divergence ledger (ADR-0004 R5; J8 spec §2)

Records every *intentional* numerical or behavioural divergence of the jittable
pipeline (`cflibs.jitpipe`) from the frozen reference
(`cflibs.inversion.pipeline.run_pipeline`), with the adjudication. A divergence
that is NOT listed here is a bug. Seeded with the M1 (J8) adjudications.

Each entry: **ID · stage · what diverges · why · parity status**.

---

## M1 (J8 — end-to-end composition) adjudications

### D-J8-1 · solve · production SB-graph path vs the lax fixed point · *parity-scoped*

The production geological preset routes the reference solve through the
**SB-graph Python path**: `IterativeCFLIBSSolver.solve` forces `_solve_python`
whenever `saha_boltzmann_graph=True` *or* a Stark diagnostic is supplied
(`iterative.py:1603`), and the geological preset sets both
(`saha_boltzmann_graph=True`, `stark_ne=True`).

The J8 jit solve spine is `cflibs.jitpipe.solve.scan_solve`, the fixed-K
`lax.scan` mirror of `_run_lax_while_loop` (the `_solve_lax` body). That body
implements the **standard common-slope** plane fit, NOT the pooled global-lstsq
SB-graph intercept extraction (which `_solve_lax` does not implement — it forces
the Python loop). `scan_solve` is parity-tested bit-for-bit against
`_run_lax_while_loop` (`tests/jitpipe/test_parity_j7.py`), and `_solve_lax` is
parity-tested against `_solve_python` for the **standard-closure, SB-graph-OFF**
path (`tests/inversion/test_iterative_lax.py`).

**Adjudication:** for M1, end-to-end COMPOSITION parity is asserted on the
shared solve-math path — the reference pinned to `saha_boltzmann_graph=False`,
`stark_ne=False`, `closure_mode="standard"`
(`tests/jitpipe/test_parity_pipeline.py`). On that path `run_one` reproduces
`run_pipeline` **bit-for-bit** (ΔT = 0, Δn_e = 0, Δconcentration = 0 on both a
synthetic forward-model spectrum and the real ChemCam BHVO-2 loc1 fixture).
The full production SB-graph + oxide + Stark-pinned-n_e estimator
(`joint_wls_solve`, ADR §6.1) is the J8.5+ deliverable; it is the production
estimator, not the M1 composition gate. **Parity status: exact on the scoped
path; production-path parity tracked for J8.5.**

### D-J8-2 · front-end · detect/identify/calibrate driven through the reference path · *delegated*

`run_one`'s front-end (response → calibrate → detect → identify → select) is
driven through `cflibs.jitpipe.host.run_front_end`, which wraps the reference
`detect_and_select_lines`. The J2 (`find_peaks_fixed`), J3/J4
(`score_comb_grid`/`build_observations`/gate stack), and J3 calibrate
(`calibrate_axis_kernel`) jit kernels are each parity-tested in isolation
(`tests/jitpipe/test_parity_j2/j3/j4`), but the host wrapper that reassembles
them into a byte-faithful reproduction of the *full* `detect_line_observations`
gate stack (gA-Boltzmann comb ranking, shift-coherence veto, line-residual gate,
retroactive min-kept-bars drop) is not yet on-device.

**Adjudication:** for M1 the observation set fed to the solve spine is therefore
**byte-identical** to the reference (it IS the reference front-end), so
composition parity is exact. Lifting the FrontEndSnapshot assembly + the full
gate stack fully on-device (so the front-end is itself jit/vmap-clean) is the
remaining work — it is a *re-host*, not a numerical divergence. **Parity status:
exact (front-end is the reference); on-device port is remaining_todo.**

### D-J8-3 · solve · closure native-JAX vs reference `pure_callback` · *numerically identical*

`scan_solve` applies closure (`standard`/`oxide`/`matrix`/ILR) through the
native-JAX `_apply_closure` instead of the reference host `pure_callback`
(`_make_closure_callback`). Inherited from J7; numerically identical for
standard/oxide/matrix, and ILR collapses to standard at the simplex.
**Parity status: exact (J7 parity test).**

### D-J8-4 · failure policy · ValueError → all-FN CFLIBSResult · *intentional, parity-preserving*

At zero usable observations the reference raises
`ValueError("No usable spectral lines detected for inversion.")`
(`pipeline.py:872`), which the scoreboard scores as all-false-negative
(`scoreboard.py:159`). `run_one` instead returns the **same all-FN record**
(`host.all_fn_result`: all-zero concentrations with element keys preserved,
`converged=False`, finite NaN-free T/n_e) — no crash, no NaN. This is the
required masked equivalent (J8 spec §1, AC4). **Parity status: scored-record
equivalent (asserted in `test_failure_policy_parity`).**

### D-J8-5 · batch · host loop vs jit(vmap) · *deferred to J9*

`run_batch` is a host loop over `run_one` for M1 (the front-end gather is impure
— SQLite + scipy — so it stays host-side). The jit-graphable solve core
(`scan_solve`) is already `vmap`-clean (J7 `test_vmap_batched_scan`); lifting the
whole solve spine under `jit(vmap(...))` once the front-end is on-device is J9.
**Parity status: per-spectrum identical to `run_one`; batched vmap is J9.**
