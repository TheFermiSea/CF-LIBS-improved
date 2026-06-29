# Citation backfill manifest — 2026-05-19

Companion artifact to bead `CF-LIBS-improved-s5ba`. Produced by the
codebase-audit citation-gap pass on dev `e9602d2`+. Lists every paper
cited in CF-LIBS code/docs that did **not** previously have a
`paper/refs.bib` entry, with the resolved canonical metadata + DOI.

This file is laid out so each verified paper can be pasted directly
into NotebookLM as an "Add source → URL" entry; the URLs link straight
to the publisher landing page (preferred) or an open-access mirror.

---

## NotebookLM ingestion — STATUS: COMPLETE

The 14 verified citations below are now **live in NotebookLM**:

- **Notebook:** *CF-LIBS Repository Citations*
- **Notebook ID:** `6307a6a1-a4e9-4c43-ba9f-24ebb2f2fb9f`
- **Source count:** 14
- **Ingested:** 2026-05-19 (via `nlm source add --text` from `briansquires@brians-macbook-pro-2310`)

Why a new notebook rather than the existing 71-source "CF-LIBS:
Calibration-Free Laser-Induced Breakdown Spectroscopy" master notebook:
that one is at NotebookLM's per-notebook source cap (URL/text adds to
it return error code `[9]` "generic" — empirically the cap is around
70–75 sources). The new notebook is a clean, repo-focused index that
mirrors `paper/refs.bib`; sources are titled with the cite-key so a
search for `\cite{LabutinEtAl2013}` in code maps directly to the
NotebookLM source.

URL-based ingestion was attempted first but failed for the publisher
landing pages (Wiley / ScienceDirect / AIP all returned the same `[9]`
error — paywalled abstract pages aren't crawlable by NotebookLM). The
final ingestion used `--text` with a structured citation block (DOI,
authors, journal, URL, CF-LIBS code-site cross-reference) per source.

**Cap-overflow guidance:** NotebookLM enforces an empirically ~70-source
per-notebook ceiling (the master "CF-LIBS: Calibration-Free LIBS"
notebook hit this at 71 sources — every URL/text add to it now returns
`[9] "generic"`). When the *CF-LIBS Repository Citations* notebook
approaches ~60 sources, the same trap is on the horizon. Plan: create
a successor notebook ("CF-LIBS Repository Citations — Continued") and
update this manifest's notebook-ID field to point at the new one.
Sources already added need not be migrated.

For each row in §"Unverified — flagged for follow-up" below:
- Either find the canonical paper via a focused Asta search, OR
- Add a Note in NotebookLM: "Cited in CF-LIBS code/docs at
  {file:line} but unable to locate authoritative version; consult
  author or remove the reference."

---

## Verified — added to `paper/refs.bib`

| Cite-key | Title | Source URL |
|---|---|---|
| `KepesEtAl2020` | Benchmark Classification Dataset for LIBS (Sci Data 7:53) | https://www.nature.com/articles/s41597-020-0396-8 |
| `NoelEtAl2025` | Automated Line Identification for Atomic Spectroscopy (ALIAS) | https://www.sciencedirect.com/science/article/abs/pii/S0584854725001405 |
| `JochumEtAl2005` | GeoReM: A New Geochemical Database for Reference Materials | https://onlinelibrary.wiley.com/doi/10.1111/j.1751-908X.2005.tb00904.x |
| `JochumEtAl2016` | Reference Values Following ISO Guidelines for Rock RMs | https://onlinelibrary.wiley.com/doi/10.1111/j.1751-908X.2015.00392.x |
| `LabutinEtAl2013` | Automatic Identification of Emission Lines in LIP by Correlation | https://pubs.acs.org/doi/10.1021/ac303270q |
| `CristoforettiEtAl2010` | LTE in LIBS: Beyond the McWhirter Criterion | https://www.sciencedirect.com/science/article/abs/pii/S0584854709003541 |
| `VandenbekeromPannier2021` | A Discrete Integral Transform for Rapid Spectral Synthesis | https://www.sciencedirect.com/science/article/abs/pii/S0022407320308700 |
| `PannierLaux2018` | RADIS: Nonequilibrium Line-by-Line Radiative Code | https://www.sciencedirect.com/science/article/abs/pii/S0022407318306411 |
| `VolkerGornushkin2023` | Extension of the Boltzmann Plot Method for Multiplets | https://www.sciencedirect.com/science/article/abs/pii/S0022407323002595 |
| `GajarskaEtAl2024` | Automated Detection of Element-specific Features in LIBS | https://pubs.rsc.org/en/content/articlelanding/2024/ja/d4ja00247d |
| `KramidaEtAl2024NIST` | NIST Atomic Spectra Database (Version 5.12) | https://physics.nist.gov/asd |
| `Salzmann1998` | Atomic Physics in Hot Plasmas (Oxford Univ. Press) | https://global.oup.com/academic/product/atomic-physics-in-hot-plasmas-9780195109306 |
| `ColombantTonon1973` | X-ray Emission in Laser-Produced Plasmas (J. Appl. Phys. 44:3524) | https://pubs.aip.org/aip/jap/article-abstract/44/8/3524/8472 |
| `Griem1974` | Spectral Line Broadening by Plasmas (Academic Press) | https://archive.org/details/spectrallinebroa0000grie |

### Code-side correction notes

- **`Wakil 2023`** in `boltzmann.py:146` is a misattribution — the
  multiplet-aggregation paper is **Völker & Gornushkin 2023**. Update
  the comment to use cite-key `VolkerGornushkin2023`.
- **`Vrabel et al. 2020`** in `loaders.py:835` correctly identifies the
  dataset, but the first author is Képeš. Both attributions are
  defensible (the dataset is named after Vrabel; the paper is
  authored-led by Képeš). Cite-key `KepesEtAl2020`.
- **`Ciucci et al. (2009)`** in `README.md:282` is a year typo — the
  CF-LIBS paper is **Ciucci et al. (1999)**, already in the bib as
  `Ciucci1999`.
- **`Colombant & Tonon 1973`** in `README.md:289` is cited as *Physics
  of Fluids* — canonical venue is *J. Appl. Phys.* 44:3524.

---

## Unverified — flagged for follow-up

These four citations appear in code/docs but were not findable on
Semantic Scholar / Scopus / arXiv / publisher landing pages with the
metadata as written. Likely scenarios: (a) misattribution (year or
author wrong), (b) preprint not indexed, (c) misremembered.

| Citation as written | File:line | Status |
|---|---|---|
| Black et al. 2024 (sparse NNLS overfitting) | `alias.py:1502` | Cannot locate. Likely candidates: Black 2024 reviews of NNLS regularisation in remote-sensing literature. **Action:** trace the original reviewer/author and update or delete the reference. |
| El Sherbini 2020 (curve-of-growth SA) | `boltzmann.py:12` | Only El Sherbini's 2005 work findable. Likely the comment author meant the 2005 paper or a later El Sherbini follow-up that wasn't surfaced. **Action:** confirm with the author or substitute `ElSherbiniEtAl2005`. |
| Landi & Degl'Innocenti 1999 (Stark broadening of H) | `README.md:294` | Could not pinpoint a 1999 paper. Landi Degl'Innocenti's polarization work spans 1994–2004. **Action:** consult the README author for the specific reference. |
| Jochum 2022 (USGS reference update) | `usgs.py:333` | Could not locate a 2022 update from Jochum. The 2016 update (`JochumEtAl2016`) is likely what was meant. **Action:** confirm and correct year. |

When NotebookLM is reconnected, run Asta over each entry above and either
verify (move into the bib) or drop the code reference.

---

## What this manifest replaces / supersedes

- The "Citation gap list" section of
  `docs/architecture/codebase-audit-2026-05-19.md` is now resolved for
  the 14 verified entries. The audit doc's gap table can be marked
  superseded by this manifest.
- Bead `CF-LIBS-improved-s5ba` (docs-2) should advance to `inreview`
  status once the four unverified citations are either confirmed or
  dropped from code/docs.
