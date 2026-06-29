# Initial Concept

## Project Goal

Enable quantitative elemental analysis without calibration standards (Calibration-Free LIBS) using rigorous plasma physics.

## Target Users

- **Analytical Chemists & Spectroscopists**: For quantitative elemental analysis and material characterization.
- **Plasma Physicists**: For modeling plasma parameters and validating theoretical models.
- **Instrument Developers**: For optimizing spectrometer designs and calibration routines.

## Key Features

- **Forward Modeling**: Generate synthetic spectra from plasma parameters (Temperature, Density, Composition).
- **Physics Engine**: Models Saha-Boltzmann equilibrium, line emission, and instrument effects.
- **Performance**: Vectorized with NumPy; optional GPU acceleration via JAX.
- **Calibration-Free Analysis**: Determine elemental concentrations directly from spectral measurements.
- **Atomic Database**: SQLite interface with NIST data.

## Project Status

- **Phase 1 (Physics Engine)**: Complete.
- **Phase 2 (Inversion/CF-LIBS)**: In progress.
