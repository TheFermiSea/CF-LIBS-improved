# J3 Implementation Spec — Jittable Line Matching + Gates (`detect_line_observations`)

**Bead:** J3 · **ADR:** [ADR-0004](../ADR-0004-jittable-inversion-pipeline.md) §4 row 4, D5 · **Track:** A · **Depends:** J0, J1 (soft J2 — develops on reference calibration) · **Estimated effort:** 12–17 pd (snapshot infra 2; kdet + comb scan + shift selection 3–4; veto + residual gate + ownership scan + min-kept bars 4–6 — the subtlest logic in the whole front-end; intensity extraction 1–2; parity + scoreboard gating 2–3)

## 1. Goals

- Port `detect_line_observations` (`cflibs/inversion/identify/line_detection.py:1099`; file is 2,475 lines; pipeline call at `cflibs/inversion/pipeline.py:519`) into `jitpipe/identify.py` with **exact reference semantics** — ADR-0004 D5: this stage is pure discrete logic + identical arithmetic; *nothing here is hot enough to justify semantic drift* (unlike J2's hypothesis scoring). It is the tightest contract in the program.
- Generalize the three existing seed kernels in the same file (`_jax_peaks_within_tolerance` `:76`, `_jax_kdet_candidate_counts` `:118`, `_jax_local_maxima_mask` `:156`) rather than reinventing them.

## 2. Current algorithm and jit-breakers (file:line; all `identify/line_detection.py` unless noted)

Flow: early-exit empties (`:1271-1332`) → adaptive R-aware tolerances (`:592-616`, applied `:1283-1301`) → `_load_transitions` (`:1476`; **SQL per element** `:1502`; gA-Boltzmann top-K=60 ranking `:1550-1579`, weight `g·A·exp(−E_k/1 eV)` `:1524-1537`) → `_find_peaks` (`:1315→2424`) → element-keyed transition **dict** (`:1334-1336`) → **kdet pre-filter** (`:1338→2148`; Rust path `:2185`; NumPy path = per-element max-over-shift-grid in-tolerance counts `:2034-2059`; coherence-mode count-only rule `:2083`) → comb top-30/element (`:1357-1360`) → shift grid (`:1592-1609`) → **comb shift-scan** (`:1851`; Python loop over shifts `:1881`; per (shift, element) score `:1612` calls the **greedy matcher** `_match_transitions_to_peaks` `:1919` — per transition, nearest in-tolerance peak not in the mutable `used` set `:1941-1986`; P/R/F1 + 4 pass thresholds `:1644-1656`; lexicographic best/fallback with `np.isclose` F1 tie-break `:1813-1848`) → accepted-element selection + fallback ladder (`:619-661`) → **shift-coherence veto** (`:1389→2287`; per-element nearest-peak signed residuals `:2227-2252`; pooled median consensus `:2335`; band = tol/3 `:2336`; keep rules `:2338-2366` incl. zero-match pass `:2343-2346`) → **observation build** `_collect_observations` (`:786`): elements in (f1, matched)-desc order (`:830-836`); per element: greedy comb matching with **cross-element peak ownership** (`used_peaks`, `:843-860`) and per-line residual gate (band = tol/3 around pooled consensus `:2255-2284`; **gated candidates recorded without consuming the peak** `:1958-1975`); per-peak nearest-transition pass (`:872-895,2386-2413`) honoring ownership + band; line-key dedupe via `seen_keys` (`:771-774`); **min-kept bars** `_drop_element_if_mostly_gated` (`:931`: kept-fraction <0.5 or kept <3 ⇒ **retroactive `del observations[n_obs_before:]`** `:971` — the only place the pipeline un-does prior appends; used peaks stay claimed `:961-963`) → **intensity extraction** `_build_observation` (`:459`: trapezoid over ±`half_width_px` `:493-498`; Gaussian-equivalent fallback `h·FWHM·√(π/4ln2)` with walk-outward FWHM `:306-365`; Poisson σ `√(Σcounts)·Δλ` `:518-519`); Voigt deconvolution dispatch (`:722-751`; default-off, pipeline does not enable it) → result + warning strings (`:1066-1096`).

Breakers: SQL + `sorted()` ranking; element dicts throughout; mutable `used` sets (catalog-order-dependent); `np.isclose` tie semantics (`:1841-1848`); Python `sort(key=…)` (`:836`) and `sorted(...)[:k]` fallback (`:653-659`); `seen_keys`/`resonance_lines`/`matched_peak_indices` sets; retroactive `del`; `gated_out` append-only bookkeeping (`:1965-1973`); walk-outward FWHM loops (`:349-361`); warning-string lists; `LineObservation` dataclass list output.

## 3. Redesign

- **FrontEndSnapshot** (host, per dataset × element set; mirrors `_AtomicSnapshot`, `solve/iterative.py:382`): `(E_max=32, K_lines=64)` padded arrays — wl, gA-weight, E_k, g_k, A_ki, stage, E_i, is_resonance, mask — plus the comb subset `(E_max, K_comb=32)` from the same deterministic host-side ranking (tie-break at `:1577`).
- **kdet:** generalize `_jax_kdet_candidate_counts` — vmap over `(E_max, S_shift=32)` of searchsorted counts; density/rarity weights (`:2087-2100`) as vectorized reductions; coherence rule (`:2083`) → `(E_max,)` keep mask.
- **Comb scan with exact greedy parity:** precompute per (element, comb line) the `K_match=8` nearest peak slots (banded searchsorted; measured fan-out ≤5 at tolerance). Per (shift, element): `lax.scan` over the 32 comb-line slots carrying a `(P_max,)` used-peak bitmask — each step picks the nearest *available* in-tolerance peak (masked argmin over 8 slots). Sequential depth 32; vmap over (S_shift × E_max) gives the whole scan as one kernel → `(S_shift, E_max)` P/R/F1/pass arrays.
- **Best/fallback shift selection:** encode the lexicographic rules (`:1813-1848`) as composite sort keys with F1 **quantized to reproduce `np.isclose` ties**; masked argmax. Fallback ladder (`:644-661`) = masked top-k by matched count.
- **Veto + consensus:** per-(element, comb-line) nearest-peak signed residual `(E_max, 32)` masked; pooled masked median (sort + count-indexed gather); per-element coherent fraction/count → keep mask replicating `:2338-2366` including the zero-match keep rule.
- **Observation build:** element order = `argsort` by (f1, matched) desc over E_max slots; `lax.scan` over ranked element slots carrying the global used-peak mask. In the scan body: (a) comb greedy sub-scan with the residual band; **gated candidates set a gated flag without consuming the peak** (exact port of `:1958-1975`); (b) per-peak pass = vectorized masked argmin over each free peak's K_match in-band transitions, respecting ownership + band; lines matched in (a) masked out — reproduces `seen_keys` dedupe exactly (line keys unique within the snapshot); (c) `_drop_element_if_mostly_gated` = counts → drop flag → zero the element's observation-validity mask; **used peaks stay claimed** (deliberate semantics at `:961-963`; the mask is never rolled back).
- **Intensity extraction:** gather fixed windows `(OBS_max, 2·half_width_px+1)` (`half_width_px` static per bucket, `:1323`) → trapezoid + Poisson σ vectorized; Gaussian-area fallback's FWHM = vectorized first-crossing-from-apex via boolean cummax + linear interpolation within the window (exact port — the reference also stops at window edges, `:349-361`). Voigt deconvolution stays host-side scipy in phase 1 (default-off, unused); phase 2 = batched fixed-iteration Gauss–Newton Voigt reusing `cflibs/radiation/kernels.py:659` — also the bridge to J10's forward-fitting scorer.
- **Output:** padded `ObservationBatch` pytree (OBS_max=512: wl, intensity, σ, element id, stage, E_k, g_k, A_ki, resonance flag, validity) + counts + int warning bitmask; host wrapper rebuilds `LineDetectionResult` with today's strings.

## 4. Tolerance contract (the tightest in the program)

- MUST agree: (a) `applied_shift_nm` **exactly equal** (same grid `:1592-1609`, replicated tie-breaks); (b) accepted/kept element sets **identical** per corpus spectrum (any diff is a bug or a documented `np.isclose`-boundary case); (c) observation line-key sets identical (target; floor Jaccard ≥ 0.98 with every diff triaged at the residual-band boundary); (d) intensities + σ rtol 1e-10 (trapezoid path: same windows, same arithmetic); (e) `residual_gate` n_gated counts and dropped-element maps equal (order-insensitive).
- MAY differ: observation ordering, diagnostic float formatting, warning ordering.
- Board: expected ΔF1 = 0; hard gate |ΔF1| ≤ 0.005 — per project memory every identifier-scoring change is benchmark-gated (PR #229 paper-faithful-ALIAS −0.041 precedent).

## 5. Acceptance criteria

1. Contract (a)–(e) green on fixtures + the 32-spectrum corpus; **BHVO-2 Sn/Th confounder fixtures as mandatory canaries** (the residual gate and min-kept bars exist for exactly those cases, `:944-958`).
2. jit + vmap (B=16) + padding invariance + no-SQLite-in-kernel.
3. Exact-port unit tests for each of the four named hazards: greedy order-dependence; `np.isclose` ties; gated-without-consuming-peak; retroactive element drop with peaks-stay-claimed.
4. Host wrapper rebuilds `LineDetectionResult`; legacy consumers unaffected.
5. Scoreboard shadow on available datasets within the gate.

## 6. Test plan

`tests/jitpipe/test_parity_identify.py` (corpus + fixtures, assertion style of `test_iterative_lax.py:180`); targeted micro-fixtures per hazard (two elements competing for one peak; F1 within `np.isclose` of each other; gated line whose peak is later claimed by a worse line — must NOT be, per reference semantics; element with 2 kept / 3 gated triggering the drop).

## 7. Risks

*Numerical:* minimal (fp64 comparisons of identical arithmetic). *Memory:* minimal (tensors ≤ few MB). *Behavioral:* **high concentration** — each hazard is one-line-to-get-wrong; mitigation is the exact-scan ports + micro-fixtures; no parallel approximations permitted in this stage.

## 8. Dependencies / files

Depends J0, J1; soft J2. Enables J8; J4 consumes `ObservationBatch`. Files: `cflibs/jitpipe/identify.py`, tests. Reference untouched. The pure-JAX ALIAS/comb/correlation/NNLS scoring ports (`docs/jax-port/{alias,comb-correlation,nnls}-consultation.md`) integrate here for J10's presence scoring but are not required for parity of this stage.
