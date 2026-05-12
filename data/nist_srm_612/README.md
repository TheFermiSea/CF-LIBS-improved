# NIST SRM 612 LIBS Spectra (TRACE-ELEMENT GLASS) — SPECTRA GAP

> **STATUS: NO SPECTRA INGESTED.**
> This directory is a placeholder documenting a gap in publicly available data.
> See the gap analysis below and the beads issue CF-LIBS-improved-9jvd.

## PROJECT DIRECTIVE (2026-05-12)

Per direct project-owner instruction: **synthesized spectra alone are not an
acceptable dataset.** The CF-LIBS benchmark requires validated, downloadable
LIBS spectra from a published source. Synthesis is acceptable only as a
*supplement* to real measured data, with a clear `spectrum_kind: "synthetic"`
flag, and it cannot be the basis of any accuracy claim.

The loader at `cflibs/benchmark/loaders.py:_load_nist_srm_612` therefore
returns `None` (graceful no-op) when this directory contains no spectra.
The certified composition (`cflibs/benchmark/reference_compositions.py`,
merged via PR #115) is retained for future use the moment real spectra
become available.

Until then, `nist_srm_612` is **not a usable benchmark dataset** — the BHVO-2
dataset (sibling directory `data/bhvo2_usgs/`, 12 real spectra from 4
validated public sources) is the shipped deliverable from this work.

## What is NIST SRM 612?

NIST SRM 612 is a borosilicate glass doped with ~38 ppm of 61 trace elements in a
SiO₂-Al₂O₃-CaO-Na₂O matrix. It is one of the most widely used calibration standards
in laser ablation and LIBS geochemistry (Pearce et al. 1997; Jochum et al. 2011).

Certified major-element composition (for CF-LIBS benchmark purposes):

| Element | Mass fraction |
|---------|---------------|
| Si | 0.3366 |
| Al | 0.0106 |
| Ca | 0.0857 |
| Na | 0.1039 |

Trace dopants (~38 ppm each = 0.0000038 mass fraction) are below the LOQ for
CF-LIBS and are not tracked in the benchmark composition schema.

## Gap analysis: sources searched (2026-05-12)

The following ranked sources were checked exhaustively; none yielded downloadable
experimental LIBS spectra of SRM 612 as a bulk material:

1. **NASA PDS Mars 2020 SuperCam calibration archive**
   - URL: https://pds-geosciences.wustl.edu/m2020/urn-nasa-pds-mars2020_supercam/calibration_supercam/
   - Result: Steel SRMs (SRM 688, 97a, 98a, 88b) present; SRM 612 **absent**.

2. **NASA PDS MSL ChemCam calibration archive**
   - URL: https://pds-geosciences.wustl.edu/missions/msl/chemcam.htm
   - Result: Geologic standards (BHVO-2, BCR-2, etc.) present; SRM 612 **absent**.

3. **Bertherat 2020 MIT thesis search (MIT DSpace)**
   - URL: https://dspace.mit.edu/handle/1721.1/
   - Result: No LIBS + USGS reference materials thesis found.

4. **USGS GeoREM**
   - URL: https://georem.mpch-mainz.gwdg.de/
   - Result: Geochemical composition database only; no spectral data of any kind.

5. **NIST PML LIBS database**
   - URL: https://physics.nist.gov/PhysRefData/ASD/LIBS/libs-form.html
   - Result: Atomic emission line positions and intensities from pure elements.
     Does NOT contain measured LIBS spectra of SRM 612 as a glass material.

6. **Zenodo API** (`q=BHVO LIBS`, `q=SRM 612 LIBS`)
   - Result: No matching datasets in top 10 results; broader search returned
     steel-LIBS and meteorite-LIBS datasets only.

7. **Figshare / Google Scholar**
   - A dataset of NIST SRM 610 and 612 *analysis results* was found on Figshare
     (doi:10.6084/m9.figshare.12804490) but this contains LA-ICP-MS compositional
     data, not LIBS spectra.

## Why SRM 612 is hard to find as open LIBS data

SRM 612 is primarily used as a **trace-element** calibration standard in LA-ICP-MS,
where single-pass ablation of a homogeneous glass gives sub-ppb detection limits.
In LIBS, the major-element matrix (Si-Al-Ca-Na glass) is straightforward, but the
trace-element signal (38 ppm) is buried in noise at typical LIBS laser energies.
Most published SRM 612 LIBS work focuses on the matrix elements and is not released
as open spectral data — only as calibration curves in paywalled journal articles.

## Recommended follow-up actions

1. **Contact IRAP SuperCam team** (Roger Wiens, Sam Clegg, Sylvestre Maurice):
   SuperCam's 2019 pre-flight calibration likely included SRM 612 but the spectra
   have not yet been deposited in PDS. Email: supercam-pds@irap.omp.eu

2. **Check ORDaR platform** (https://ordar.obspm.fr/) — LIBS mineral databases
   from French labs, some of which have used SRM glass standards.

3. **LIBS-ENFL group, University of Malaga** — published SRM 610/612 LIBS work
   (Laserna group); data may be available on request.

4. **Synthesis fallback** (requires Lead approval): Simulate SRM 612 LIBS spectra
   using the NIST atomic line database + Boltzmann distribution at assumed plasma
   temperature (~8000–12000 K, LTE approximation). These would be flagged
   `spectrum_kind: "synthetic"` and excluded from CF-LIBS accuracy gates.

## Beads issue

CF-LIBS-improved-9jvd: Ingest BHVO-2 USGS basalt SRM + NIST SRM 612 reference datasets
