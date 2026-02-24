# High-Performance Computational Framework for Ultrafast CF-LIBS
**Architecture Design Document v2.0**

## 1. Executive Summary

This document details the architecture for an **Industrial-Grade, Real-Time Compositional Analyzer** for Laser Powder Bed Fusion (LPBF) and Directed Energy Deposition (DED) processes.

Unlike traditional "Calibration-Free" (CF-LIBS) approaches that rely on slow, iterative non-linear solvers (e.g., Levenberg-Marquardt) during run-time, this framework utilizes a **Spectral Search Engine** paradigm. This approach decouples the heavy physics calculations from the real-time inference loop, leveraging High-Performance Computing (HPC) resources to achieve **kHz throughput** with **<5 ms latency**.

### Core Philosophy: "Physics at Compile Time"
Instead of asking *"What physics parameters fit this spectrum?"* (Inverse Modeling), we ask *"Which pre-calculated physical reality matches this observation?"* (Forward Modeling). This preserves 100% of the physical rigor of the Saha-Boltzmann equations while treating the inference problem as a high-speed vector search.

---

## 2. Physics Model: The "White Box" Manifold

The system is grounded in the standard model of plasma emission, solved explicitly for a dense, ultrafast laser-induced plasma.

### 2.1. The Forward Model
We generate a **Manifold** $\mathcal{M}$—a high-dimensional lookup table of synthetic spectra generated from first principles.

For a given set of plasma parameters $\theta = \{T_{max}, n_{e,max}, \mathbf{C}_{species}\}$, the synthetic spectral radiance $I_{\text{synth}}(\lambda)$ is computed as the time-integrated emission of a cooling plasma:

$$
I_{\text{synth}}(\lambda; \theta) = \int_{t_{gate}}^{t_{end}} \sum_{s \in \text{Species}} \sum_{k \in \text{Lines}} \epsilon_{s,k}(\lambda, T_e(t), n_e(t)) \cdot \text{optics}(\lambda) \, dt
$$

Where the instantaneous emissivity $\epsilon_{s,k}$ is governed by the **Saha-Boltzmann** equation:

$$
\epsilon_{s,k} = \frac{h c}{4 \pi \lambda_k} A_{ki} \frac{g_k}{U_s(T_e)} n_s \exp\left(-\frac{E_k}{k_B T_e}\right) \times \text{Voigt}(\lambda, \lambda_k, \Gamma_L(n_e), \Gamma_G(T_e))
$$

**Critical Physics Features:**
* **Ionization Balance:** Solved via the Saha-Eggert equation iteratively to ensure charge neutrality ($n_e = \sum Z_i n_i$).
* **Ultrafast Cooling:** The temperature $T_e(t)$ follows a power-law decay characteristic of fs-plasmas, integrated over the ICCD gate width ($5 \mu s$).
* **Opacity:** Self-absorption is modeled via the radiative transfer equation for the "hot track" geometry.

### 2.2. Traceability
This is **not Machine Learning**.
* Every point in the Manifold is traceable to specific atomic transition probabilities ($A_{ki}$) from NIST.
* When the system identifies a match, it returns the exact physical parameters ($T_e, n_e$) used to generate that spectrum.

---

## 3. HPC Architecture Implementation

The system is split into two distinct phases to maximize hardware utilization.

### Phase 1: Offline Manifold Generation (The "Simulator")
**Hardware Target:** 3 Nodes $\times$ Tesla V100 GPU
**Software:** Python + JAX (Just-In-Time compiled XLA)

The goal is to compute $10^7 - 10^8$ synthetic spectra covering the entire operational envelope of the LPBF/DED process.

* **Grid Search Space:**
    * **Temperature ($T_{max}$):** 0.5 eV – 2.0 eV (Step: 0.05 eV)
    * **Electron Density ($n_e$):** $10^{16}$ – $10^{19}$ cm$^{-3}$ (Log steps)
    * **Composition ($\mathbf{C}$):** $0\% - 100\%$ for primary alloys (e.g., Ti, Al, V, Fe)
* **JAX Vectorization:**
    Instead of looping, we broadcast the Voigt profile calculation across the GPU. A single V100 can compute ~10,000 high-resolution spectra per second.
* **Output:** Hierarchical Data Format (HDF5) or Apache Parquet file (~500 GB - 2 TB).

### Phase 2: Online Inference Engine (The "Sensor")
**Hardware Target:** 1 Node (Xeon Gold CPUs, >256 GB RAM)
**Software:** Rust + ZeroMQ + AVX-512

This service runs on the "Edge" node connected to the ICCD. It loads the generated Manifold into RAM and performs nearest-neighbor search and fine-tuning.

#### The Rust Actor Model
The system uses an asynchronous Actor model (Tokio) to handle high-speed I/O without blocking the compute threads.

1.  **Ingest Actor:** Subscribes to ZeroMQ `PUB` socket from the Spectrometer driver. Buffers incoming spectra.
2.  **Compute Actor (The Worker Pool):**
    * Pulls a spectrum.
    * **Step A: Cleaning (AirPLS)** – Removes the Blackbody background ($>2000$ K melt pool emission) using sparse matrix solvers.
    * **Step B: Deconvolution (NNLS)** – Solves the Non-Negative Least Squares problem to separate overlapping peaks.
    * **Step C: Inference** – Matches the cleaned feature vector against the in-memory Manifold.
3.  **Control Actor:** Publishes the result (`{"Al": 6.04, "V": 3.98}`) to the machine PLC via TCP/IP or Modbus.

---

## 4. Algorithmic Details

### 4.1. Robust Baseline: AirPLS
To handle the intense thermal background from the trailing melt pool, we use **Adaptive Iteratively Reweighted Penalized Least Squares**.

**Objective Function:** Minimize the roughness of the baseline $z$ while keeping it faithful to the data $x$:
$$
S = \sum_{i} w_i (x_i - z_i)^2 + \lambda \sum_{i} (\Delta^2 z_i)^2
$$
* **Iterative Weighting ($w_i$):** Weights are adjusted such that $w_i \to 0$ in peak regions (signal) and $w_i \to 1$ in baseline regions.
* **Rust Implementation:** Uses Cholesky decomposition on a pentadiagonal matrix (extremely fast on CPU).

### 4.2. High-Throughput Deconvolution: NNLS
Instead of "finding peaks," we assume *every* line in the database exists and solve for their amplitudes.

**The Matrix Equation:**
$$
\min_{\mathbf{x} \ge 0} \| \mathbf{A}\mathbf{x} - \mathbf{b} \|^2
$$
* $\mathbf{b}$: The measured, background-subtracted spectrum ($2048 \times 1$).
* $\mathbf{A}$: The **Design Matrix** ($2048 \times N_{lines}$). Column $j$ contains the unit-area Pseudo-Voigt profile for atomic line $j$ at the spectrometer's resolution.
* $\mathbf{x}$: The vector of true line intensities.

**Why NNLS?**
1.  **Sparsity:** It mathematically forces non-existent elements to exactly zero amplitude.
2.  **Super-Resolution:** It can accurately quantify two lines that are closer than the instrument's resolution limit (deconvolution).
3.  **Speed:** The design matrix $\mathbf{A}$ is pre-computed. Solving this takes $<1$ ms using AVX-512 optimizations.

---

## 5. Hardware Utilization Strategy

| Component | Hardware | Implementation Detail | Justification |
| :--- | :--- | :--- | :--- |
| **Manifold Generator** | **Tesla V100s** | JAX `pmap` across 3 nodes | Calculating $10^8$ Voigt profiles is a massive FP64 workload ideal for Tensor Cores. |
| **Spectral Database** | **RAM (380GB)** | `lazy_static` HashMaps in Rust | The entire NIST database and the Coarse Manifold fit in RAM, eliminating disk I/O latency. |
| **Matrix Solver** | **Xeon Gold (AVX-512)** | `nalgebra` / Intel MKL | Dense linear algebra for $2048 \times N$ matrices is faster on CPU than paying the PCIe penalty to send single frames to the GPU. |
| **Inter-Process Comms** | **ZeroMQ** | TCP Loopback | Decouples the hardware driver (C++/Python) from the Analysis Engine (Rust), preventing GC pauses from crashing the sensor. |

## 6. Development Roadmap

1.  **Week 1:** Port `saha-eggert.py` to **JAX**. Run on 1 GPU to verify "Time-Integrated" physics.
2.  **Week 2:** Run the **Manifold Generation** job on the full cluster. Generate the HDF5 artifact.
3.  **Week 3:** Deploy the **Rust Microservice**. Implement the AirPLS and NNLS modules.
4.  **Week 4:** Integration testing with the ICCD. Calibrate the "Design Matrix" ($\mathbf{A}$) using a standard lamp to fix the exact pixel-to-wavelength mapping.

---
*© 2025 TheFermiSea. All Rights Reserved.*
