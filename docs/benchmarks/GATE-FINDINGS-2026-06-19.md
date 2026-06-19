# Benchmark-gate findings — 2026-06-19

The Phase-3 discipline: implement a cited SOTA method as **opt-in**, then **benchmark-gate**
it on the scoreboard (flag-off vs flag-on) and flip the default **only on a measured win**.
This file records what the gate found. The point of the gate is to *prevent* adopting changes
that don't actually help — the repo has regressed 3× on ungated algorithm changes.

## B1 — errors-in-variables (ODR) Boltzmann fit → **REJECT (keep opt-in)**

**Result:** on `synthetic_fixedforward` (the self-contained dataset available off-cluster),
`use_odr=true` produced ΔF1 = +0.000 and ΔRMSE = +0.000 — *identical to 3 decimals* — both at
the default `odr_x_uncertainty=0.0` and at a generous `odr_x_uncertainty=0.1` eV.

**Two verified reasons it is inert, not just weak:**

1. **Wrong code path for the production preset.** The geological preset runs with
   `saha_boltzmann_graph=True`, so temperature is determined by the multi-element common-slope /
   Saha-Boltzmann-graph fit (`iterative.py:_common_slope_kernel` / `_fit_saha_boltzmann_graph`),
   **not** the per-element `BoltzmannPlotFitter` that B1's `use_odr` patches. `boltzmann_fitter.fit`
   is never called on the production solve path under the graph preset — so the flag cannot move the
   result. To make ODR matter it would have to be implemented inside the common-slope kernel.
2. **E_k is precise.** Even on a per-element Boltzmann plot, errors-in-variables only helps when the
   x-axis (upper-level energy E_k) carries real uncertainty. NIST upper-level energies are known to
   ≪ 0.01 eV, so the EIV correction is negligible by construction — `use_odr` with `odr_x_uncertainty≈0`
   *mathematically degenerates to OLS* (asserted by B1's own unit test).

**Disposition:** `use_odr` stays as a correct, tested, default-off opt-in (harmless; may matter for a
future non-graph preset), but is **not** adopted as a default. The gate did its job: it stopped a
"high-leverage" literature method from being adopted as a no-op.

## What a real accuracy win requires (next phase)

The gate is proven to work end-to-end (plumbing → scoreboard → verdict). To demonstrate an actual
improvement over the literature baseline, a candidate must be (a) wired into the **active** solve path
(the common-slope/SB-graph kernel, the `self_absorption_mode` handler, or the Saha balance) — not just
a standalone physics class — and (b) gated on the **composition-truth datasets** (chemcam_calib,
csa_planetary, supercam_*), whose data lives only on the cluster's staged repo. The most promising
candidates on physical grounds (literature-backed, act on the dominant error terms): **CD-SB / C-sigma
self-absorption** (turns the worst lines into signal) and the **multiplet-aware common-slope fit**
(more usable lines, less blend bias). These are tracked for the next benchmark-gated batch.
