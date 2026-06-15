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

---

## J9 / J12 (on-device front-end + scoreboard) adjudications

### D-J9-1 · front-end · detect/identify/kdet/LineSelector/segmented-calibration moved on-device · *re-host, parity exact*

Supersedes the "remaining_todo" of D-J8-2. `run_one(ondevice_front_end=True)` (the
default) now runs the J1 detect, J3 comb scan / shift-select / veto / observation
build, the kdet coherence pre-filter, the post-detection `LineSelector`, **and** the
segmented wavelength calibration as parity-tested JIT kernels
(`run_front_end_ondevice`). The catalog SQL + gA-Boltzmann comb ranking + scipy peak
detection stay host-side (ADR-0001; dynamic-shaped pre-trace). **Adjudication:** the
on-device gate stack reproduces the reference `detect_and_select_lines` observation
set bit-for-bit (obs Jaccard 1.0 on synthetic + real BHVO-2, raw + geological), and
the segmented axis matches the reference to max|Δλ| 0.00025 nm on multi-segment
broadband. **Parity status: exact (re-host, not a numerical divergence).**

### D-J9-2 · front-end · single-segment calibration delegated to the reference core (aa9e) · *parity-restored*

The on-device segmented calibrator's robust core is a **deterministic stratified
RANSAC**, vs the reference's **600-draw random RANSAC**. On seam-free **single-segment**
spectra (narrow-band synthetic + the real aalto minerals) the two resolve a sparse /
multimodal anchor set to different registrations (model-class flip on sparse synthetic;
~1.9 nm shift-mode flip on muscoviteE35), shifting the corrected axis and flipping
marginal lines / the solve outcome (board run2: synthetic ΔF1 −0.291, aalto fail 1 vs 0).
**Adjudication (bead aa9e, commit a2f6009):** keyed on the *structural* signal
(`detect_ccd_seams(...).size == 0`), single-segment spectra delegate to the reference
`_ld_calibrate`, restoring byte parity (board run3: aalto/synthetic ΔF1 = 0, failures
= reference); the multi-segment path is untouched (D-J9-1). **Known residual:** on
muscoviteE35 parity is restored by inheriting the reference's physically-dubious
~1.9 nm shift mode — a pre-existing reference multimodal-RANSAC issue, filed as a
separate bug (an ambiguity guard for *both* pipelines), not a jit divergence.
**Parity status: exact for single-segment (= reference); regression test
`test_ondevice_calib_single_segment_parity.py`.**

### D-J9-3 · front-end · Stark n_e diagnostic reference-delegated · *delegated (6apc blocked on kernel-hardening)*

`run_front_end_ondevice` calls the reference `measure_stark_ne` for the n_e
diagnostic (`host.py:2512`); n_e pins the production solve to the ≤10 % M1 tolerance
with **no downstream slack**. **6apc attempt (2026-06-15, parked on branch
`6apc-stark-build`):** the host candidate-selection orchestrator (`_ondevice_stark_ne`,
mirroring the reference two-pass control flow incl. the DB 0.1 nm nearest-match,
the multiplet-blend gate, and the break-after-max_lines-*successes* cap-before-trim)
was fully built and the break-after-successes regression test passes. But the
already-"parity-tested" device LM Voigt kernel (`measure_stark_ne_jit` /
`fit_lorentz_fwhm_lm`) **cannot fit real ChemCam windows** — it returns `None` (no
usable line) on BHVO-2 loc1, geological *and* raw. Three kernel-internal root causes:
(1) the 5×5 normal-equations solve goes NaN at ChemCam intensity scale (~1e12);
(2) `extract_windows` centers the window but `_init_params` assumes left-packing, so on
the coarse real axis (~0.049 nm/sample) the init baseline is NaN — synthetic parity
held only because its dense axis fills the whole window; (3) the strict 20-iter
`converged` gate rejects otherwise-good real-data fits. The kernel was only ever
exercised on synthetic dense-axis data. **Adjudication:** Stark stays **delegated** —
the n_e fed to the solve IS the reference's, so composition parity through n_e is exact.
6apc is **blocked on hardening the LM Voigt kernel for real-data** (per-window intensity
normalization + left-packed windows + a scipy-equivalent acceptance gate), tracked as a
new bead; the host orchestrator is ready on `6apc-stark-build` for when the kernel lands.
**Parity status: exact (delegated); on-device port blocked on kernel-hardening.**

### D-J12-1 · scoreboard · jit pipeline vs reference on the goal board · *parity-scoped, capped*

`cflibs scoreboard --pipeline jit` runs `run_one(ondevice_front_end=True)` per spectrum
vs `--pipeline reference` (`run_pipeline`). On the capped board (aalto +
synthetic_fixedforward, max-spectra 8, post-aa9e run3) the jit pipeline reproduces the
reference **micro-F1 exactly (ΔF1 = 0)** and the **failure count exactly**. **Known
residual:** synthetic_fixedforward solve RMSE 48.7 vs 44.4 wt % over the 5 solved
spectra — a small on-device-solve delta within the M1 concentration tolerance, on a
synthetic corpus whose truth is itself suspect (bead 5yo1); not observed on real data.
**Adjudication:** scoped-board parity holds; the full all-spectra + holdout board is
the M3 measurement (bead stdl), gated on D-J9-3 (6apc) for an end-to-end "fully
on-device" claim. **Parity status: exact on the capped board; full-board M3 run
pending.**
