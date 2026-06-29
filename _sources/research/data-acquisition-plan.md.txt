# Data Acquisition & Integration Plan — LIBS (Atomic, P0) + Plasma-OES Molecular Thermometry (P1)

**Status:** Authoritative · **Owner:** Synthesis lead · **Scope:** One-time exhaustive data/access acquisition for a physics-based CF-LIBS composition pipeline (now) and a future PLD/CVD non-equilibrium T_vib/T_rot plasma-OES module (later).

> **How to use this document:** Section 6 is the single copy-pasteable **REQUEST-ONCE CHECKLIST**. Sections 1–5 are the rationale; Section 7 lists the human-gated decisions. Every claim traces to a specialist dossier (cited inline as `[D1]`–`[D5]` for the five dossiers, plus repo paths and primary refs).

---

## 1. Strategy & Sequencing

### 1.1 Priority ordering

| Tier | Goal | What | Why now/later |
|------|------|------|---------------|
| **P0 — NOW** | CF-LIBS elemental composition | Atomic lines, levels, partition functions (mostly already in DB), **STARK-B Stark widths**, VALD3/Kurucz cross-checks | The shipped pipeline depends on this; the one outstanding *new* atomic ask (STARK-B for n_e) is the highest-value single item. `[D1][D5]` |
| **P1 — LATER** | PLD/CVD vibrational/rotational temperatures of CN, C2, N2, N2+, OH, NH, CH (+CH4 IR) | Molecular **electronic-emission** line lists (ExoMol/Kurucz/MoLLIST) + **OES thermometry tools** (massiveOES/MOOSE, PGOPHER, LIFBASE, SPECAIR) + **IR** lists (HITEMP/ExoMol) | Greenfield — `cflibs/` has **zero** molecular-emission code today `[D2][D3][D4]`. But the user requests access **once**, so we bundle the molecular ask in the same sitting. |

### 1.2 Two physics distinctions that govern *what to request and which engine can use it*

These two splits are stated up front because they determine the entire molecular half of the plan. Getting them wrong means requesting data that no available engine can consume.

**Distinction A — Electronic-emission bands vs IR rovibrational absorption.** `[D1][D2][D3]`
- **Atomic emission (LIBS):** LTE single-temperature; lives in VAMDC atomic nodes (VALD3, NIST-ASD, Kurucz, CHIANTI, TOPbase) + **STARK-B** for Stark broadening → n_e.
- **Molecular, problem (a) — ELECTRONIC EMISSION band systems** used for T_vib/T_rot in plasmas: CN violet B²Σ–X²Σ ~388 nm, C2 Swan d³Π–a³Π ~516 nm, N2 Second-Positive C³Π–B³Π ~337 nm, N2+ First-Negative B²Σ–X²Σ ~391 nm, OH A²Σ–X²Π ~309 nm, NH A³Π–X³Σ ~336 nm, CH A²Δ–X²Π ~431 nm. **These are NOT in HITRAN.** They live in PGOPHER/SPECAIR/LIFBASE/massiveOES and parts of ExoMol-electronic + Kurucz diatomic lists. `[D2][D3]`
- **Molecular, problem (b) — IR ROVIBRATIONAL absorption** (CH4, OH ground-state, CO): HITRAN/HITEMP/ExoMol-IR. **N2 is homonuclear → no electric-dipole IR → its ONLY usable diagnostic is the electronic bands.** C2 is likewise homonuclear (no IR). `[D2][D3]`

**Distinction B — LTE single-T vs non-equilibrium two-temperature.** `[D2][D3][D4]`
- PLD/CVD plasmas are **non-equilibrium**: T_vib ≠ T_rot ≠ T_exc.
- **ExoJAX is LTE single-temperature only** (one Tgas via Q(T)) → **disqualified** for the molecular goal. It remains the right JAX-native engine for the *atomic* LTE forward model and for *future IR-LTE* molecules. `[D3][D5]`
- **RADIS HAS a true two-temperature mode** (`non_eq_spectrum(Tvib=, Trot=, Ttrans=, vib_distribution='boltzmann'|'treanor', ...)`) — **but only for ground-electronic-state IR rovibrational bands** (CO, CO2, OH-IR, CH4). RADIS does **not** synthesize UV/Vis electronic band systems. `[D3][D5]`
- **Therefore:** for the electronic-band T_vib/T_rot goal, the usable engines are the dedicated OES tools — **massiveOES/MOOSE** (open, two-T + state-by-state), **PGOPHER** (build-any-band from constants), **LIFBASE** (OH/CH/CN/N2+), **SPECAIR** (commercial, air-plasma multi-T). For the IR half, **RADIS**. `[D2][D3][D4]`

### 1.3 One-line consequence

> Request **atomic P0 + STARK-B** to finish CF-LIBS; bundle **molecular electronic line lists + OES tools** (massiveOES/MOOSE, PGOPHER, LIFBASE, optionally SPECAIR) for the non-equilibrium goal and **HITEMP/ExoMol-IR + RADIS** for IR. **Do not** request N2/C2 from HITRAN/HITEMP (not there, would not help). `[D2][D3][D5]`

---

## 2. Atomic Sources (P0) — VAMDC nodes + STARK-B

### 2.1 What is already ingested (do NOT re-request as primary)

The production DB `ASD_da/libs_production.db` (verified present, 6.1 MB, 2026-06-09) already holds: `[D1][D5]`
- **28,727 lines** for 84–86 elements from **NIST-ASD** via `datagen_v2.py` (wavelength_nm, aki, ei/ek_ev, gi/gk, rel_int, aki_uncertainty, **accuracy_grade A..E 100% populated**).
- **9,448 energy levels** (NIST-scraped) and **175 species_physics** rows (IP + atomic mass).
- **146 partition-function** rows from `NIST_ASD_fit` + **Barklem & Collet 2016** (`table8_vNov2022.dat`). B&C is DONE — do not re-request. `[D1]`

→ **NIST-ASD lines/levels and B&C partition functions are DONE.** The only NIST follow-up is a **complete-levels re-ingest** to fix the 5–40% partition-function deficits for Rydberg-rich neutrals (Ca I, Na I, K I) flagged in audit F3 — use the official **Levels CSV** download, not the regex scrape. `[D1]` (`docs/audit/2026-06-09-overhaul/01-forward-physics.md` F3)

### 2.2 The #1 gap — STARK-B (electron-density lever)

The DB has `stark_w` populated for **98.6%** of lines, but `stark_w_source` shows only **244 lines (0.85%)** carry real STARK-B/Konjević literature values; **22,951 (80%)** are `konjevic_lambda_sq_scaled` heuristics, 4,574 interpolated, 562 hydrogenic. `[D1]` Audit `01-forward-physics.md` F5 explicitly recommends extending STARK-B coverage for basalt/LIBS species. Stark widths → **n_e diagnostic** (core to the Saha loop) and Voigt wings. **This is the single highest-value new request.** `[D1][D2][D5]`

### 2.3 Atomic source table

| Node | Coverage for our species | What to request | Access | Format | Already have | Gap / priority |
|------|--------------------------|-----------------|--------|--------|--------------|----------------|
| **NIST-ASD** `[D1][D5]` | All target elements, neutral + ions, with **per-line accuracy grades AAA..E** | (DONE for lines.) **Complete energy-Levels CSV** per species (Ca I, Na I, K I + all) for partition sums; optionally widen weak-line/`rel_int=0` filter | Web forms (no account); also VAMDC node. `lines_form.html` / `levels_form.html`, Tab-delimited output | Tab/CSV; XSAMS via VAMDC | 28,727 lines, 9,448 levels, accuracy grades 100% | **Levels re-ingest only.** **NOT a VAMDC node for the canonical bulk path — fetch native.** Record ASD version + DOI `10.18434/T4W30F`. **P0** |
| **STARK-B** `[D1][D5]` | Calculated electron-/ion-impact Stark **W (FWHM) + shift d**, SCP method, 150+ papers. Confirmed: Ca II, Fe II, Mg II, Si II, Ti II + many neutrals. n_e 1e12–1e19 cm⁻³ (LIBS 1e16–1e18 inside range); T from few-thousand K up | W & d tables for **every (element, ion)** in target set at T=5,000–40,000 K, n_e=1e16–1e18 cm⁻³; capture fit coeffs (log w = a0+a1·logT+a2·(logT)², d/w = b0+b1·logT+b2·(logT)²) and **NV validity flags** | **Manual web export** (element→ion→line→T/n_e grid→CSV/TSV); OR VAMDC-TAP VSS2→XSAMS. **No bulk download exists** | CSV/TSV (**W,d in Ångström: 1 Å = 0.1 nm**); XSAMS | Only 244 lines have real values | **THE top gap.** Repo ingest expects hand-downloaded CSV/TSV at `/cluster/shared/cf-libs-data/stark_b/raw/` cols `(transition, T_e, n_e, gamma_W, gamma_d[, alpha])` (`scripts/archive/migrations/ingest_stark_b.py`, verified present). **P0** |
| **VALD3** `[D1][D5]` | Atomic + some diatomic, broad periodic-table coverage, neutral + ions. **Only source giving log gf + levels + all three damping constants (γ_rad, γ_stark, γ_waals) with per-line source refs** | **Extract All**, long format, **1800–11000 Å** (covers 200–1000 nm + margin; widen the older 1980–9710 Å draft), all target elements + ions I–III, **including damping columns + refs** | **Free registration required**; web Extract All, VALD-EMS email, FTP for large, or VAMDC-TAP (TAP lacks some extraction features — prefer native Extract) | VALD long-format ASCII; XSAMS | — (fills `gamma_vdw_log`, `gamma_self_log`; independent γ_stark cross-check) | Best single **supplement** to NIST. Mirrors: Uppsala/Montpellier/Moscow. **P0** |
| **Kurucz** `[D1][D5]` | Deepest Fe-group inventory. `gfall` = lab-based (use for quantification); `gfpred` = predicted (completeness/blends only). Per-line natural + quadratic-Stark + vdW-by-H damping | `gfall` (merged + per-element) for target species; damping constants. **Avoid mass-ingesting `gfpred` for quantification.** **Audit which Kurucz files are already in the DB before re-pulling (overlap risk)** | **Plain HTTP** `kurucz.harvard.edu` (HTTPS cert broken: `ERR_TLS_CERT_ALTNAME_INVALID`); `wget --no-check-certificate` / `curl -k` | Fixed-width 160-col ASCII | Partially ingested ("NIST-ASD + Kurucz ingested" per CLAUDE.md) — **verify** | Memory note: **Kurucz > NIST on SuperCam** real data → completeness has value. **P0/P1** (audit first) |
| **CHIANTI** `[D1][D5]` | Critically-evaluated levels/A-values/collision strengths, **optimized for hot/coronal (highly-ionized) plasmas** — sparse for neutral + singly-ionized at ~1 eV | A-value/level **cross-check** for specific ionized species if NIST discrepancy found; **collision data** valuable for any future non-LTE/CR validation (audit SOTA gap) | Open tarball `chiantidatabase.org/chianti_download.html` (v11.0.2, no account) + ChiantiPy; also VAMDC node | CHIANTI ASCII (`.elvlc/.wgfa/.scups`) | — | **Mismatched T regime** → not a primary LIBS line source. **P1/P2** (grab tarball opportunistically; it's free + self-contained) |
| **TOPbase** (Opacity Project) `[D1][D5]` | Theoretical R-matrix LS-coupling term energies, gf, photoionization. **Z=1–26 only → MISSES Ni, Cu, Zn, Sr, Ba (Z>26)** | gf / level **gap-filler** for low-Z species only where NIST/Kurucz are thin; specific (ion, transition) — **not a bulk pull** | Web `cds.unistra.fr/topbase`; VAMDC node | ASCII; XSAMS | — | Theoretical (no lab grade) → cross-check only. Distinct from TIPbase/IRON Project. **P2** |
| **Spectr-W3** `[D1]` | Plasma-spectroscopy DB: levels, gf, radiative + autoionization rates, some Stark/electron-impact broadening | Optional secondary Stark/gf cross-check on species STARK-B covers poorly | Web (was slow/timed out); VAMDC node | Web tables; XSAMS | — | Secondary Stark cross-check; STARK-B is canonical. **P2** |

### 2.4 Per-line accuracy grades — which sources deliver them `[D1]`
NIST-ASD = **yes** (AAA ≤0.3% .. E ≥50%, already in DB). STARK-B = **semi** (NV validity flags; widths carry 15–50% uncertainty per Konjević 2002 / Gigosos 2014). VALD3 = **per-line source reference** (traceable provenance). TOPbase / Kurucz-`gfpred` = theoretical, **no experimental grade** (treat as B/C-equivalent at best). CHIANTI = critically-evaluated but no per-line LIBS-style grade.

### 2.5 VAMDC access layer `[D1][D5]`
VAMDC federates ~30 nodes; uniform TAP endpoint `http://<node>/tap/sync/?LANG=VSS2&FORMAT=XSAMS&QUERY=<VSS2>` → XSAMS XML, discoverable via the registry. Python: `astroquery.vamdc` + `vamdclib` (`pip install git+https://github.com/keflavich/vamdclib-1.git`). **Critical access split (verified across D1/D5):** **NIST-ASD and ExoMol are NOT VAMDC nodes** — fetch them natively. VALD3, STARK-B, CHIANTI, TOPbase, TIPbase, Spectr-W3 **are** nodes. Use VAMDC for the *secondary/federated* atomic pull in one reproducible pass; for VALD3 prefer the native Extract tools (cleaner damping columns); for STARK-B the native CSV export matches the repo ingest format directly.

---

## 3. Molecular Sources (P1) — Coverage Matrix

**Legend:** **E** = electronic-emission band usable for T_vib/T_rot · **IR** = ground-state rovibrational/IR · **—** = absent/not usable · **(build)** = simulate from constants. `[D2][D3]`

| Molecule | Diagnostic band (λ) | ExoMol | HITRAN | HITEMP | Kurucz | MoLLIST | PGOPHER | SPECAIR | LIFBASE | massiveOES/MOOSE | **Recommended source** |
|----------|--------------------|--------|--------|--------|--------|---------|---------|---------|---------|------------------|------------------------|
| **CN** | violet B²Σ–X²Σ ~388 nm | **E+IR** (KTPSYT / Trihybrid) | — | — | E (`cnbx`/`cnax`) | E+IR | E(build) | E (violet+red) | E (B-X violet) | E (CN) | **ExoMol Trihybrid/KTPSYT** (.states/.trans/.pf); cross-check Kurucz `CNBX.ASC` |
| **C2** | Swan d³Π–a³Π ~516 nm | **E+IR** (8states incl Swan) | — | — | E (Swan) | E | E | E (Swan) | — | E (Swan) | **ExoMol 8states** (Yurchenko 2018, Swan-bearing); homonuclear → no IR |
| **N2** | 2nd-Positive C³Π–B³Π ~337 nm | **E** (WCCRMT incl C-B 2nd-pos + B-A 1st-pos) | IR (quadrupole only, unusable) | — | partial | — | E(build) | E (1st+2nd pos) | — | E (C-B/SPS) | **ExoMol WCCRMT** *(correction: N2 2nd-pos IS downloadable, not only SPECAIR/PGOPHER)*; homonuclear → no usable IR |
| **N2+** | 1st-Negative B²Σ–X²Σ ~391 nm | **—** | — | — | — | — | E(build) | E (1st-neg + Meinel) | E (B-X 1st-neg) | E (B-X/FNS) | **SPECAIR / LIFBASE / massiveOES / PGOPHER** — *highest-risk omission*; absent from all line-list DBs |
| **OH** | A²Σ–X²Π ~309 nm | **E+IR** (MYTHOS rovibronic, 0–80000 cm⁻¹, Tmax 8000K; +HITEMP) | IR | IR | E (`oh.asc`) | E+IR | E(build) | E (A-X + X-X) | E (A-X 309) | E (A-X) | **ExoMol MYTHOS** (covers BOTH UV A-X 309 + IR in one list) |
| **NH** | A³Π–X³Σ ~336 nm | **E+IR** (2kNigHt, 5 states) | — | — | E (`nh.asc`) | E+IR | E(build) | E (A-X) | — | E (NH) | **ExoMol 2kNigHt**; cross-check Kurucz `nh.asc` |
| **CH** | A²Δ–X²Π ~431 nm ("G-band") | **E+IR** (MoLLIST/SBYT: A-X+B-X+C-X) | — | — | E (`ch.asc`) | E+IR | E(build) | — | E (A-X/B-X/C-X) | **likely absent** → PGOPHER | **ExoMol MoLLIST** (all 3 electronic systems); **CH is the gap MOOSE lacks → generate with PGOPHER** |
| **CH4** | *(no electronic band)* | **IR** (34to10/MM, very high T) | IR (473k) | **IR (31.9M lines, best)** | — | — | — | — | — | — | **HITEMP-CH4** via RADIS for CVD/combustion T; ExoMol if T>2000K; **IR absorption only, not a T_vib/T_rot emission band** |

**Per-molecule recommendation summary:** `[D2][D3]`
- **CN, C2, N2, OH, NH, CH** → primary line lists from **ExoMol** (electronic-state quantum labels enable external two-T population models); **Kurucz diatomic `.ASC`** as free same-sitting cross-check. **MoLLIST** (Bernath) is the empirical backbone — pull it *via ExoMol* (`MoLLIST-<mol>` datasets) for uniform `.states/.trans` format.
- **N2+** → **NOT in any line-list DB**; must come from **SPECAIR + LIFBASE + massiveOES + PGOPHER**. This is the single highest-risk omission for the collaborator's air/PLD plasma goal — request all four covering tools.
- **CH4** → **HITEMP** (hot bands; the correct source, *not* room-T HITRAN) loaded by **RADIS** non-eq mode; IR absorption diagnostic only.

---

## 4. OES Thermometry Tooling (P1)

### 4.1 What each tool gives, and when to use it `[D2][D3][D4]`

| Tool | License | Platform | Bands covered | Non-equilibrium model | Role |
|------|---------|----------|---------------|----------------------|------|
| **MOOSE** (`moose-spectra`) | **MIT, open** | pip, Python ≥3.10 | OH(A-X), N2(C-B/SPS), N2+(B-X/FNS), NH(A-X), NO(B-X), C2(Swan), CN(B-X) | Separate T_rot, T_vib (Boltzmann) | **PRIMARY open fitter.** Reuses massiveOES line lists. Hits the whole target list **except CH and CH4.** Synthetic-spectrum least-squares → handles overlapping bands |
| **massiveOES** | open (Bitbucket) | Python | Same systems as MOOSE | Boltzmann two-T **AND state-by-state** (temperature-independent) | Use when two-T Boltzmann fails and you need **state-by-state** rotational distributions (common in PLD/CVD). Source of the line-list DBs MOOSE reuses |
| **PGOPHER** | **BSD/open** (frozen 2022) | Win/Mac/Linux | **Any** diatomic band from spectroscopic constants (incl. **CH A-X**, **N2+ 1st-neg**) | Custom (non-Boltzmann) populations → arbitrary T_rot/T_vib | **Gap-filler + ground-truth generator.** Build bands no line-list DB / Python fitter covers. Engine behind ExoMol N2 WCCRMT. Reads HITRAN/ExoMol |
| **LIFBASE** | freeware ("no support") | **32-bit Windows only** | OH(A-X), OD, CH(A-X/B-X/C-X), CN(B-X), N2+(B-X 1st-neg), NO | Two-T population control | **Canonical transition-probability source** many other tools derive from. Covers the **N2+ gap.** **Not embeddable in Linux/JAX** → consume data indirectly via MOOSE/massiveOES or export tables once |
| **SPECAIR** (SpectralFit S.A.S.) | **COMMERCIAL** | Windows GUI | N2 1st+2nd-pos, N2+ 1st-neg+Meinel, CN violet+red, C2 Swan, OH A-X, NH A-X, NO, O2, CO 4th-pos + N/O/C atomic lines | **Full multi-T** (T_tr, T_rot, T_vib, T_elec, T_e) + built-in fitting | **Validated cross-check / oracle.** De-facto standard for air-plasma OES; covers N2+. Closed binary → request the program, not a line list. **Only paid item** |

### 4.2 The non-equilibrium T_vib/T_rot story `[D2][D3][D4]`
PLD/CVD VDFs are non-Boltzmann; **two-temperature (Treanor) is an approximation** that may fail. The robust method is **synthetic-spectrum forward-fit** (simulate band contour at trial T_vib/T_rot/instrument-width, least-squares match) — what SPECAIR/massiveOES/MOOSE/PGOPHER do; it handles blended rotational lines. The older **Boltzmann-plot** (ln(I/gA) vs E from resolved lines) **breaks** on dense overlapping plasma OES bands → **do not use it** for CN/C2/N2/OH band fitting.

### 4.3 Recommended adoption `[D3][D4]`
1. **PRIMARY fitter: MOOSE** (open, MIT, pip, two-T, covers most of the target list).
2. **Line-list generator / gap-filler: PGOPHER** (generate CH A-X and N2+ 1st-neg from constants; independent cross-check).
3. **Data source for OH/CH/CN/NO/N2+: LIFBASE** (export tables once; consume via MOOSE).
4. **State-by-state escalation: massiveOES** (when Boltzmann two-T fails).
5. **Validated cross-check: SPECAIR** (request quote — see §7) for N2/N2+/CN/NH/OH air-plasma bands with absolute intensities.
6. **CH4 (IR only): RADIS + HITEMP** — not an OES tool.

---

## 5. Engine Gap — Ingestion / Compute Capability

### 5.1 What each engine can ingest/compute `[D3][D5]`

| Engine | Ingests | Computes | Non-equilibrium? | Verdict for our goals |
|--------|---------|----------|------------------|------------------------|
| **ExoJAX** | `MdbExomol` (.states/.trans/.pf), `MdbHitran`/`MdbHitemp`; atomic `AdbVald`/`AdbSepVald`, `AdbKurucz`/`AdbSepKurucz`; pressure broadening via `gamma_vald3`/`gamma_uns`/`gamma_KA3/KA4` | LTE line-by-line opacity, differentiable (JAX + NumPyro) | **No — LTE single-T (one Tgas) only** | **Right** for atomic LTE forward model + future IR-LTE molecules. **Disqualified** for molecular electronic / two-T. *(Verify `AdbKurucz` class name against installed version.)* |
| **RADIS** | HITRAN/HITEMP/GEISA/ExoMol/CDSD/Kurucz-atomic; auto-downloads | Line-by-line emission/absorption; **`non_eq_spectrum(Tvib, Trot, Ttrans, vib_distribution=boltzmann\|treanor, ...)`** | **Yes — true two-T** | The non-eq engine **but only for ground-state IR rovibrational** (CO, CO2, OH-IR, CH4). **Does NOT model UV/Vis electronic bands.** Built-in non-LTE constants mature mainly for **CO/CO2** — other molecules need external `.states` energies (ExoMol) |
| **Custom (to build)** | ExoMol/massiveOES electronic `.states` (electronic+v+J labels) | Hönl-London + Franck-Condon + separate T_vib/T_rot/T_exc populations | Yes (by design) | Net-new physics code for the electronic-band non-equilibrium goal if not delegating wholly to massiveOES/MOOSE |

### 5.2 What cflibs has today (audit result) `[D3][D5]`
- **Atomic:** `cflibs/datagen_v2.py` ingests NIST-ASD via `ASDCache` (lines 190–950 nm Mechelle, levels with high-Rydberg fix, IPs, partition functions with B&C-2022 parse). `cflibs/radiation/emissivity.py` models **only atomic** line emissivity (ε = hc/4πλ · A_ki · n_k, `Transition` keyed by element+ionization_stage; n_k from Saha-Boltzmann). The `lines` table already has `stark_w`/`stark_alpha`/`stark_shift`/`stark_w_source` columns + provenance field anticipating STARK-B. **Atomic backbone is built.**
- **Molecular: ZERO support** (confirmed via ripgrep/Serena). The only `exojax`/`radis` hits are **architectural-mirroring comments** (ADR-0001 kernels.py overlap-and-add; jax_runtime.py MemoryPolicy/MDBSnapshot; bayesian samplers note petitRADTRANS+exojax) — **not** molecular-DB ingestion. **No diatomic / Hönl-London / Franck-Condon / T_vib / T_rot code exists.**

### 5.3 What must be built for molecular non-equilibrium thermometry `[D3][D5]`
A **new `cflibs/molecular/` band-system emitter** (parallel to `emissivity.py`), fed by a new rovibronic `.states/.trans` reader, with **Hönl-London + Franck-Condon factors and separate T_vib/T_rot/T_exc populations** — the SahaBoltzmann LTE single-T solver cannot be extended in place. **Architecture decision (beyond data acquisition):** adopt **massiveOES/MOOSE** as the embeddable engine (open, plasma-specific, two-T + state-by-state) rather than building from scratch; use **RADIS** for the IR half; use **SPECAIR/PGOPHER** as validation oracles. A deeper symbol-level audit is warranted before architecting the module.

---

## 6. THE "REQUEST-ONCE" CHECKLIST

> Do all of this in **one sitting**. Order: **P0 atomic first**, then **P1 molecular**. Record every version string and license/acknowledgment at download time. `[D5]`

### Account matrix (do these first) `[D5]`
- **No account:** NIST-ASD, Kurucz, CHIANTI, STARK-B (web), ExoMol, RADIS/ExoJAX/PGOPHER/massiveOES/LIFBASE downloads, VAMDC portal queries.
- **Free account REQUIRED:** **VALD3** (registration) · **HITRAN + HITEMP** (single HITRANonline login covers both).
- **Commercial / paid:** **SPECAIR** only (decision-gated — see §7).

---

### ─── P0 ATOMIC (do now) ───

**☐ 1. VALD3 — register + one Extract-All** `[D1][D5]`
- Register email at a mirror: Uppsala `http://vald.astro.uu.se/~vald/php/vald.php` (or Montpellier/Moscow).
- Mode **Extract All**; window **1800–11000 Å**; format **LONG** (incl. γ_rad, γ_stark, γ_waals + per-line refs); gf as **log gf**; energy unit eV; **wavelength medium = pick ONE (air or vacuum) and RECORD it** to match pipeline convention; delivery **via FTP** (Extract-All is large).
- **Cite/ack:** *"This work has made use of the VALD database, operated at Uppsala University, the Institute of Astronomy RAS in Moscow, and the University of Vienna."* + Ryabchikova et al. 2015 (Phys. Scr. 90, 054005); Piskunov et al. 1995; Kupka et al. 1999/2000; Pakhomov et al. 2019 (hyperfine). Free academic.

**☐ 2. NIST-ASD — complete Levels CSV (+ optional weak-line lines refresh)** `[D1][D5]`
- Lines (if refreshing): `https://physics.nist.gov/PhysRefData/ASD/lines_form.html`, full window **180–1100 nm**, all cols (Aki, Acc, Ei, Ek, gi, gk, Type), Tab-delimited.
- **Levels (the real action):** `https://physics.nist.gov/PhysRefData/ASD/levels_form.html` — **complete energy-level CSV** for **every target species, prioritizing Ca I, Na I, K I** (fixes partition-function deficits, audit F3).
- Use canonical `physics.nist.gov` host (ignore any `*-test.nist.gov` redirect). **Not a VAMDC node — fetch native.**
- **Cite:** Kramida, Ralchenko, Reader & NIST ASD Team, NIST ASD (ver. X.Y), `https://physics.nist.gov/asd`, **DOI 10.18434/T4W30F**. **RECORD THE VERSION.** Public domain.

**☐ 3. STARK-B — manual export of W & d tables (THE top item)** `[D1][D5]`
- Web interface `http://stark-b.obspm.fr` → walk the element picker; for **every (element, ion)** in the target set (verify each — coverage is paper-by-paper): **Fe I/II, Ca I/II, Mg I/II, Si I/II, Ti I/II, Al I/II, Na I, K I, Cr, Mn**, plus **H I Balmer α/β** for n_e.
- Export **W and d** at **T = 5,000–40,000 K** and **n_e = 1e16–1e18 cm⁻³**; capture fit coeffs (a0,a1,a2 / b0,b1,b2) and **NV validity flags**.
- Save as CSV/TSV (**W,d in Ångström**) to `/cluster/shared/cf-libs-data/stark_b/raw/` with cols `(transition, T_e, n_e, gamma_W, gamma_d[, alpha])` for `scripts/archive/migrations/ingest_stark_b.py` (verified present).
- **If H Balmer coverage is thin in STARK-B, supplement with standard Gigosos-Cardeñoso / Stehlé H-β Stark tables (outside STARK-B).**
- **Cite:** Sahal-Bréchot, Dimitrijević & Ben Nessib 2014 (Atoms 2, 225) + 2011 (Baltic Astron. 20, 523); Konjević et al. 2002 (JPCRD 31, 819); Gigosos 2014 (J. Phys. D 47, 343001). Free academic. **Restrict the n_e diagnostic to `stark_w_source IN ('stark_b','interpolated')`.**

**☐ 4. Kurucz — `gfall` + diatomic `.ASC` (dual-use; audit overlap first)** `[D1][D5]`
- **Audit which Kurucz files are already in the DB before pulling** (CLAUDE.md says ingested).
- Over **plain HTTP** (HTTPS cert broken): `http://kurucz.harvard.edu/linelists.html` and `/linelists/` — atomic `gfall.dat` + per-element `gf*.dat` for target species (lab lines for quantification; skip mass `gfpred`).
- **Same sitting, diatomic (serves P1):** `CNAX.ASC`, `CNBX.ASC` (CN red A-X + **violet B-X 388 nm**), `C2*.ASC` (incl. Swan), `ch.asc`, `oh.asc`, `nh.asc`. Use `wget --no-check-certificate` / `curl -k`.
- **Cite:** Kurucz line-list releases (Kurucz 2017, Can. J. Phys.; CD-ROM 23, 1995); acknowledge R. L. Kurucz, Harvard-Smithsonian CfA. Public domain.

**☐ 5. (Opportunistic) CHIANTI tarball** `[D1][D5]`
- Open download `http://www.chiantidatabase.org/chianti_download.html` (v11.0.2, no account) + `pip install ChiantiPy`. Free, self-contained; **value = collision data for future non-LTE/CR validation** (audit SOTA gap), not LTE line data.
- **Cite:** Dere et al. 1997 (A&AS 125, 149) + version paper (Del Zanna et al.); CHIANTI consortium acknowledgment text.

**☐ 6. (Optional, only if a specific ion shows gaps) TOPbase / Spectr-W3** via VAMDC portal `https://portal.vamdc.eu` — Z≤26 gf gap-fill / secondary Stark cross-check. P2; skip unless needed. `[D1][D5]`

---

### ─── P1 MOLECULAR (bundle in the same sitting) ───

**☐ 7. HITRANonline — create ONE free account, pull HITEMP + HITRAN** `[D5]`
- Register `https://hitran.org`. **HITEMP** (`https://hitran.org/hitemp/`, the correct hot-band source): **OH, CO, NO, CH4, CO2, H2O** — `.par.bz2` bulk (plan disk: CO2/H2O are multi-GB; extract range with `hitemp_bz2_extract.py`). For PLD/CVD prioritize **OH, CO, CH4**. **HITRAN** (room-T) only for species HITEMP lacks (**HCN, C2H2**). Programmatic via **HAPI** (`pip`).
- **DO NOT request N2/C2** (homonuclear, no usable IR).
- **Cite:** HITEMP — Rothman et al. 2010 (JQSRT 111, 2139) + per-molecule updates: CH4 Hargreaves 2020 (10.3847/1538-4365/ab7a1a); CO2 Hargreaves 2024 (10.1016/j.jqsrt.2024.109324); NO/NO2/N2O Hargreaves 2019; CO Li 2015. HITRAN2020 — Gordon et al. 2022 (JQSRT 277, 107949). **Record the version.**

**☐ 8. ExoMol — electronic line lists (.states + .trans + .pf + .def, all in one pass)** `[D2][D3][D5]`
- Native HTTP, no account (`https://www.exomol.com/data/molecules/<MOL>/`). **Download `.pf` alongside `.states/.trans`** (needed for population/temperature retrieval). Pull `.def.json` per dataset for authoritative line counts/Tmax.
  - **CN:** `12C-14N__KTPSYT.*` (Kozlov 2024) and/or `__Trihybrid` (Syme & McKemmish 2021) — violet B-X 388 + red A-X.
  - **C2:** `12C2__8states.*` (Yurchenko 2018) — Swan 516.
  - **N2:** `14N2__WCCRMT.states/.trans` + `14N2__WCCRMT__ERJ.trans` (the C-B 2nd-positive transitions; Western 2018 + Jans 2024).
  - **OH:** `16O-1H__MYTHOS.*` (Mitev 2025) — A-X 309 + IR in one list.
  - **NH:** `14N-1H__2kNigHt.*` (Perri & McKemmish 2023) — A-X 336.
  - **CH:** `12C-1H__MoLLIST.*` (Masseron 2014 / Bernath 2020) — A-X 431 + B-X + C-X.
  - **CH4 (if T>2000 K beyond HITEMP):** `12C-1H4` (34to10/MM).
- **N2+ is NOT in ExoMol** → see ☐ 11–12.
- **License:** CC-BY (some pages note CC-BY-SA-4.0 share-alike — check `https://www.exomol.com/data/licence/`, affects redistribution of derived DBs). **Cite** the ExoMol release paper (Tennyson/Yurchenko) + per-molecule paper (CN: Syme & McKemmish 2021 MNRAS 505, 4383; C2: Yurchenko 2018 MNRAS 480, 3397; OH: Mitev 2025 MNRAS 536, 3401; etc.).

**☐ 9. RADIS — install (the IR non-equilibrium engine)** `[D3][D5]`
- `pip install radis` (or conda-forge). Auto-downloads HITRAN/HITEMP/ExoMol on demand. Use `non_eq_spectrum(Tvib=, Trot=)` for **IR** species (CH4, CO, OH-IR). **Not** for UV/Vis electronic bands. Built-in non-LTE constants mature mainly for CO/CO2 — supply ExoMol `.states` for others. **Cite:** Pannier & Laux 2019 (JQSRT 222–223, 12). LGPL-3.

**☐ 10. ExoJAX — install (LTE atomic + future IR-LTE only)** `[D5]`
- `pip install exojax`. Reads VALD3/Kurucz atomic into a differentiable JAX model (aligns with the codebase). **Flag explicitly: cannot do two-T → not for the non-equilibrium molecular goal.** **Cite:** Kawahara et al. 2022 (ApJS 258, 31).

**☐ 11. massiveOES + MOOSE — clone/install (PRIMARY open electronic-band fitter)** `[D3][D4][D5]`
- `pip install moose-spectra` + clone `https://github.com/AntoineTUE/Moose`; clone `https://bitbucket.org/OES_muni/massiveoes` for bundled simulation DBs. Covers OH/N2/N2+/NH/NO/C2/CN with two-T + state-by-state. **Covers the N2+ gap.** **Cite:** Voráč et al. 2017 (PSST 26, 025010 + J. Phys. D 50, 294002); MOOSE repo (A. Salden, TU/e, MIT).

**☐ 12. PGOPHER — download + collect constant files (gap-filler / oracle)** `[D2][D3][D4]`
- Free BSD download `https://pgopher.chm.bris.ac.uk` (Win/Mac/Linux; frozen 2022, stable). Gather `.pgo` constant files for **CH A-X** (MOOSE lacks it) and **N2+ 1st-negative**, plus CN violet, C2 Swan, N2 C-B, OH A-X, NH A-X. **Cite:** Western 2017 (JQSRT 186, 221).

**☐ 13. LIFBASE — download (N2+/OH/CH transition-probability source)** `[D2][D3][D4]`
- Freeware (32-bit Windows, no support) `https://www.sri.com/platform/lifbase-spectroscopy-tool/`. Export OH(A-X), CH(A-X), CN, NO, **N2+(B-X 1st-neg)** tables once (covers the N2+ gap). **Not embeddable on Linux** → consume data via MOOSE/massiveOES or one-time export; may need a Windows VM/compat layer. **Cite:** Luque & Crosley 1999 (SRI Report MP 99-009).

**☐ 14. (Decision-gated) SPECAIR — request license/quote** `[D2][D3][D4]`
- Commercial; order/manual at `http://www.specair-radiation.net` (site intermittently `ECONNREFUSED` — retry or email SpectralFit). Decide academic vs commercial license up front (see §7). Best-validated air-plasma N2/N2+/NO bands w/ absolute intensities + full multi-T fitting. **Cite:** Laux et al. 2003 (PSST 12, 125).

**☐ 15. (Optional, P2) CDSD CO2 / CDMS / JPL** via VAMDC — CDSD-1000 redundant with HITEMP CO2; CDMS/JPL are **rotational/radio, OUT OF SCOPE** for optical electronic thermometry (documented to prevent a wasted request). `[D5]`

---

## 7. Open Decisions & Blockers

### 7.1 Needs the user's choice
1. **SPECAIR procurement (human-gated, $).** Only paid item. Decide **academic vs commercial license** and whether to buy at all — the free **massiveOES/MOOSE + PGOPHER + LIFBASE** stack may suffice (it covers N2+). Recommend: proceed open-stack first, request SPECAIR only as a validation oracle if air-plasma absolute-intensity calibration is needed. `[D3][D4][D5]`
2. **Wavelength-medium convention (air vs vacuum)** — must be chosen **once** and applied consistently to the NIST and VALD3 extractions (and recorded). Project convention is vacuum Å below ~200 nm, configurable above. `[D5]`
3. **Molecular engine architecture** — adopt **massiveOES/MOOSE** as the embeddable electronic-band engine (recommended) vs build a custom `cflibs/molecular/` emitter. This is an ADR-level decision beyond data acquisition; a deeper symbol-level audit should precede it. `[D3][D5]`
4. **CH4 / IR scope** — confirm whether the collaborator actually does IR absorption diagnostics. If not, **skip HITEMP CH4/CO2/H2O multi-GB downloads** and RADIS-IR work entirely (saves significant disk + setup). `[D2][D5]`

### 7.2 Human-gated / external access
5. **VALD3 registration** — free but requires an email account + approval; do this first so the Extract-All FTP link arrives during the session. `[D5]`
6. **HITRANonline registration** — free single login unlocks both HITRAN + HITEMP; required before any IR pull. `[D5]`
7. **LIFBASE on Linux** — 32-bit Windows-only, "no support," unclear redistribution license → needs a VM/compat layer; plan to **export tables once** rather than embed. `[D3][D4]`

### 7.3 Coverage blockers / verify-at-request-time
8. **N2+ First-Negative (391 nm) — highest-risk omission.** Absent from ExoMol/HITRAN/HITEMP/Kurucz-standard. **Mitigation built into checklist:** request SPECAIR + LIFBASE + massiveOES/MOOSE + PGOPHER (all four cover it). `[D2][D3]`
9. **STARK-B per-(element,stage) coverage is paper-by-paper** — no complete machine-readable species list; the picker is image-based. **Verify each target species at the picker**, especially neutral **Fe I, Ni, Cu, Zn, Sr, Ba**. No bulk download exists. `[D1]`
10. **Kurucz overlap risk** — already partially ingested; **audit the DB before re-pulling** to avoid duplicate lines. `[D1]`
11. **Kurucz HTTPS broken** (`ERR_TLS_CERT_ALTNAME_INVALID`) — use plain HTTP / `--no-check-certificate`. Not a dead site. `[D1][D5]`
12. **SPECAIR / ExoMol-page / massiveOES-Bitbucket fetch fragility** — several sites rendered thin or refused connection during research; **verify exact `.trans/.states` filenames, install steps, and current license text on the live pages at request time.** `[D2][D3][D5]`
13. **RADIS non-LTE fidelity** — built-in spectroscopic constants mature mainly for CO/CO2; for NO/OH/CH4 supply external ExoMol `.states` energies or accept reduced non-eq fidelity. Verify per molecule before relying on T_vib≠T_rot retrieval. `[D5]`
14. **ExoJAX `AdbKurucz` class name** — confirmed `AdbVald`/`AdbSepVald`; verify the Kurucz reader class against the installed ExoJAX version. `[D3]`
15. **VAMDC reliability for the two anchors** — **NIST-ASD and ExoMol are NOT VAMDC nodes**; do not route those through the portal — native fetch is mandatory. `[D5]`

### 7.4 Greenfield-build blocker (post-acquisition)
16. **No molecular code exists in `cflibs/`** — T_vib/T_rot is net-new physics (Hönl-London + Franck-Condon + multi-population solver), not a config change. Data acquisition unblocks but does not complete the molecular goal; file a follow-up bead for the `cflibs/molecular/` module + engine-adoption ADR. `[D3][D5]`

---

**Provenance key:** `[D1]` = VAMDC atomic-acquisition dossier · `[D2]` = molecular line-lists dossier · `[D3]` = plasma-OES thermometry tools dossier · `[D4]` = OES tooling/engine dossier · `[D5]` = acquisition-mechanics (request-once) dossier. Repo claims verified: `ASD_da/libs_production.db` (present, 6.1 MB), `scripts/archive/migrations/ingest_stark_b.py` (present), `docs/audit/2026-06-09-overhaul/01-forward-physics.md` (F3/F5).
