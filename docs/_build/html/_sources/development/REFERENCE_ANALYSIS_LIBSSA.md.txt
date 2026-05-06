# Analysis of libssa (v2.5.1)

**Source:** [https://github.com/kstenio/libssa](https://github.com/kstenio/libssa)
**Date:** 2026-01-07
**Context:** Reference analysis for CF-LIBS architecture and feature planning.

## 1. Overview
`libssa` is a GUI-focused Python application for the analysis of LIBS (Laser-Induced Breakdown Spectroscopy) data. It provides a complete workflow from data import to plasma parameter determination (Saha-Boltzmann) and multivariate analysis (PCA/PLS).

## 2. Architecture
*   **Pattern:** Monolithic State Object (`Spectra` class) coupled with a Qt-based GUI.
*   **State Management:** The `Spectra` class acts as a "God Object," holding all application state (raw data, processed data, configuration, models) in mutable dictionaries and NumPy arrays.
*   **Coupling:** Core physics logic (`env/equations.py`) is relatively decoupled (pure functions), but the data orchestration is tightly bound to the `Spectra` class structure.
*   **Concurrency:** Uses `QThreadPool` for background tasks, aiming to keep the GUI responsive during heavy fitting operations.

## 3. Performance & Scalability
*   **Compute Engine:** Relies on standard NumPy/SciPy.
    *   **Vectorization:** Physics functions (e.g., `lorentz`, `voigt`) are vectorized over the wavelength axis.
    *   **Bottlenecks:** Peak fitting involves Python-level loops over the number of peaks (`for i, p in enumerate(params)`). For complex multi-peak fitting on massive datasets, this will be a performance limiter compared to fully vectorized or JIT-compiled approaches.
*   **Memory Model:** In-memory processing. The `Spectra` class loads all data into RAM. This limits scalability to the available system memory and makes analyzing massive hyper-spectral imaging datasets difficult.
*   **Storage:** Uses custom Pickle-based format (`.lb2e`) and standard CSV/Excel. No evidence of high-performance binary storage (HDF5/Zarr) for large-scale data.

## 4. Key Features to Note
*   **Outlier Removal:** Implements SAM (Spectral Angle Mapper) and MAD (Median Absolute Deviation). **Action:** Consider implementing these in `CF-LIBS`.
*   **Line Profiles:** Supports Lorentzian, Gaussian, and Voigt profiles.
*   **Saha-Boltzmann:** Implements the standard graphical method (Boltzmann plot) and electron density determination via Stark broadening approximations.
*   **Multivariate Analysis:** Integrated PCA and PLS using `scikit-learn`.

## 5. Code Quality & Testing
*   **Style:** Uses type hints and NumPy-style docstrings (good practice).
*   **Testing:** Test suite (`src/tests`) is minimal, focusing on high-level integration (loading, startup) rather than rigorous unit testing of physics kernels.
*   **Dependencies:** Conservative version pinning (`numpy<=1.25.1`), likely for stability.

## 6. Recommendations for CF-LIBS
To achieve the goal of being "more advanced, robust, and performant," CF-LIBS should diverge from `libssa` in the following ways:

| Feature | libssa Approach | Recommended CF-LIBS Approach |
| :--- | :--- | :--- |
| **Compute** | NumPy (CPU) | **JAX (CPU/GPU)** for auto-diff and JIT compilation. |
| **Architecture** | Monolithic Class | **Functional/Pipeline** architecture. Separate data state from compute kernels. |
| **Storage** | Pickle/CSV | **HDF5 or Zarr** for lazy loading and massive datasets. |
| **Fitting** | `scipy.optimize` (loop-heavy) | **Gradient-Descent** (via JAX/Optax) or **Levenberg-Marquardt** (vectorized). |
| **Testing** | Sparse Integration | **Rigorous Unit Tests** for every physics kernel against NIST benchmarks. |
| **Typing** | Standard Hints | **Strict Static Analysis** (mypy strict) + Array Shape Typing (jaxtyping). |

## 7. Extensibility
`libssa` is extensible via code modification but lacks a plugin system. Adding a new line profile requires modifying `equations.py` and the GUI logic. CF-LIBS should aim for a registry-based pattern where new physics models can be registered without modifying core code.
