# BHVO-2 LIBS Spectra (USGS Hawaiian Basalt)

BHVO-2 is the USGS Geochemical Reference Standard for Hawaiian basalt, one of the
most widely cited calibration standards in geochemical LIBS literature (used by
ChemCam, SuperCam, and laboratory geochemistry groups worldwide).

## Contents

12 spectra from 4 independent publicly-available sources:

| File | Source | n_shots | wvl_range (nm) | Intensity units |
|------|--------|---------|----------------|-----------------|
| `chemcam_bhvo2_loc{1-4}_spectrum.csv` | ChemCam MSL pre-flight (2013) | ~30 | 240.8–905.6 | photon/shot/mm²/sr/nm |
| `csa_bhvo2_1000pulse_spectrum.csv` | CSA Planetary LIBS (2017) | 1000 | 198.1–970.1 | ADU (dark-subtracted) |
| `csa_bhvo2_200pulse_spectrum.csv` | CSA Planetary LIBS (2017) | 200 | 198.1–970.1 | ADU (dark-subtracted) |
| `supercam_1545mm_loc{1-3}_spectrum.csv` | SuperCam pre-flight @ 1545 mm (2019) | 15 | 243.8–852.8 | DN (raw, dark-subtracted) |
| `supercam_4250mm_loc{1-3}_spectrum.csv` | SuperCam pre-flight @ 4250 mm (2019) | 30 | 243.8–852.8 | DN (raw, dark-subtracted) |

Each file is a two-column CSV: `wavelength` (nm), `intensity`. Spectra are
mean-averages over all shots acquired at a single raster position on the BHVO-2 pellet.

## CSV format

```
wavelength,intensity
240.811,1.61e+11
240.864,9.93e+10
...
```

Readable by pandas: `pd.read_csv(path)` → columns `wavelength`, `intensity`.

## Sources

### 1. ChemCam MSL pre-flight calibration (NASA open data)

**Files:** `chemcam_bhvo2_loc{1-4}_spectrum.csv`

Four raster locations on a pressed BHVO-2 pellet, measured in a cleanroom at LANL
using the ChemCam flight-model instrument. Mars-analog conditions: 7 Torr CO2, 3 m
standoff, ~10 mJ/pulse at 1064 nm.

- **Instrument:** ChemCam LIBS (Wiens et al. 2012, Space Sci. Rev. 170:167-227)
- **Spectrometer:** Echelle, 240–905 nm (6144 channels)
- **Intensity calibration:** Calibrated radiance in photon/shot/mm²/sr/nm
  (Clegg et al. 2017, Spectrochim. Acta B 129:64-85)
- **Data source:** `data/chemcam_calib/msl_ccam_libs_calib.csv`
  (PDS archive: https://pds-geosciences.wustl.edu/missions/msl/chemcam.htm)
- **License:** NASA open data — no copyright restriction

**Citation:**
> Clegg, S.M. et al. (2017). Recalibration of the Mars Science Laboratory ChemCam
> instrument with an expanded geochemical database. *Spectrochim. Acta B* 129:64-85.
> https://doi.org/10.1016/j.sab.2016.12.003

### 2. CSA Planetary LIBS Open Dataset (Government of Canada Open Data)

**Files:** `csa_bhvo2_{1000,200}pulse_spectrum.csv`

Two averaged spectra at 1000 and 200 pulse averages respectively, from ambient-air
laboratory LIBS measurements in a dataset assembled by the Canadian Space Agency for
planetary analog studies. Dated 2017-01-19 per the description document.

- **Instrument:** CSA LIBS; Nd:YAG 1064 nm; ambient air; 198–970 nm (13490 channels)
- **Intensity:** Dark-subtracted ADU counts; some near-zero background channels
  may be slightly negative
- **Data source:** `data/csa_planetary_libs/LIBSOpenDatacsv.7z`
  (Canadian Space Agency Open Data: https://donnees-data.asc-csa.gc.ca/dataset/137bdc17-2f46-4d70-98e7-acc46f602e9f)
- **License:** Government of Canada Open Government Licence v2.0

**Citation:**
> Canadian Space Agency (2017). Laser-Induced Breakdown Spectroscopy (LIBS) dataset
> for materials for Planetary Exploration. Government of Canada Open Data.
> UUID: 137bdc17-2f46-4d70-98e7-acc46f602e9f
> https://donnees-data.asc-csa.gc.ca/dataset/137bdc17-2f46-4d70-98e7-acc46f602e9f

### 3 & 4. SuperCam pre-flight FM calibration (NASA open data)

**Files:** `supercam_{1545,4250}mm_loc{1-3}_spectrum.csv`

Three raster locations at each of two standoff distances (1545 mm and 4250 mm),
measured on April 14, 2019 using the SuperCam flight-model LIBS during pre-launch
calibration at IRAP (Toulouse) and LANL. Data are raw digital number (DN) counts,
dark-subtracted but not radiometrically calibrated.

- **Instrument:** SuperCam LIBS (Wiens et al. 2021, Space Sci. Rev. 217:4)
- **Laser:** Nd:YAG 532 nm, ~10 mJ/pulse
- **Spectrometer:** 7933 channels, 243.8–852.8 nm
- **n_shots:** 15 (1545mm) and 30 (4250mm) shots per raster location
- **Data source:** NASA PDS Mars 2020 SuperCam calibration archive
  - 1545 mm: https://pds-geosciences.wustl.edu/m2020/urn-nasa-pds-mars2020_supercam/calibration_supercam/1545mm/LBHVO20401/
  - 4250 mm: https://pds-geosciences.wustl.edu/m2020/urn-nasa-pds-mars2020_supercam/calibration_supercam/4250mm/LBHVO20401/
- **License:** NASA open data — no copyright restriction

**Citation:**
> Wiens, R.C. et al. (2021). The SuperCam Instrument Suite on the NASA Mars 2020
> Rover: Body Unit and Combined System Tests. *Space Sci. Rev.* 217:4.
> https://doi.org/10.1007/s11214-020-00777-5

## Certified Composition

True composition is in `cflibs.benchmark.reference_compositions.BHVO2_BASALT_USGS`,
sourced from Jochum et al. 2005 (GeoReM) + USGS Reference Materials Program:

| Element | Mass fraction |
|---------|---------------|
| Si | 0.2333 |
| Ti | 0.0164 |
| Al | 0.0714 |
| Fe | 0.0861 |
| Mn | 0.0013 |
| Mg | 0.0436 |
| Ca | 0.0815 |
| Na | 0.0165 |
| K  | 0.0043 |
| P  | 0.0012 |

**Reference:** Jochum, K.P. et al. (2005). GeoReM: A New Geochemical Database.
*Geostandards and Geoanalytical Research* 29(3):333-338.

## Notes on cross-source comparability

These 12 spectra come from **three different instruments** at different wavelength
ranges, intensity scales, and atmospheric conditions (CO2 vs. air, different pressures).
They should **not** be merged into a single calibration matrix without per-source
normalization. The benchmark loader (`cflibs/benchmark/loaders.py`) handles
multi-source datasets by storing per-spectrum provenance in `SampleMetadata`.

Negative intensity values in CSA and SuperCam spectra are dark-subtraction
artifacts in low-signal background channels — treat as zero for
peak-integration purposes.

## Beads issue

CF-LIBS-improved-9jvd: Ingest BHVO-2 USGS basalt SRM + NIST SRM 612 reference datasets
