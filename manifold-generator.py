"""
CF-LIBS Phase 1: Offline Manifold Generator (JAX/HPC)
-----------------------------------------------------
Generates a dense "Digital Twin" of the LIBS spectral space.
Solves Saha-Boltzmann + Radiative Transfer + Time Integration.

Hardware: NVIDIA V100 (Tensor Cores utilized for massive matrix ops)
Output: HDF5 Manifold Archive
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
import pandas as pd
import sqlite3
import h5py
import time

# --- 1. CONFIGURATION ---
DB_PATH = "libs_production.db"
OUTPUT_PATH = "spectral_manifold.h5"

# Simulation Grid (Adjust based on alloy system)
# Example: Ti-6Al-4V system (Ti, Al, V, Fe)
TEMP_MIN, TEMP_MAX, TEMP_STEPS = 0.5, 2.0, 50      # eV
NE_MIN, NE_MAX, NE_STEPS = 1e16, 1e19, 20          # cm^-3 (Log space)
CONC_STEPS = 20                                    # Resolution for mixtures

# Physics Constants (JAX compatible)
KB_EV = 8.617e-5
H_PLANCK = 4.135e-15
C_LIGHT = 2.998e8
M_E = 9.109e-31
H_J = 6.626e-34

# Instrument
WL_MIN, WL_MAX = 250.0, 550.0
PIXELS = 4096
FWHM_INST = 0.05 # nm

# Time Integration
GATE_DELAY = 300e-9 # 300 ns
GATE_WIDTH = 5e-6   # 5 us
TIME_STEPS = 20     # Integration resolution

# --- 2. DATA LOADING (CPU) ---
def load_atomic_data():
    """
    Loads atomic data from SQLite and converts to JAX DeviceArrays.
    We flatten everything into arrays indexed by line_id.
    """
    print(f"Loading Physics from {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Load Lines (Neutrals & Ions)
    # Filter for relevant elements only
    query = """
    SELECT 
        l.element, l.sp_num, l.wavelength_nm, l.aki, l.ek_ev, l.gk, 
        sp.ip_ev
    FROM lines l
    JOIN species_physics sp ON l.element = sp.element AND l.sp_num = sp.sp_num
    WHERE l.wavelength_nm BETWEEN ? AND ?
    AND l.element IN ('Ti', 'Al', 'V', 'Fe') 
    ORDER BY l.wavelength_nm
    """
    df = pd.read_sql_query(query, conn, params=(WL_MIN, WL_MAX))
    
    # Map Element Names to Integers (0=Ti, 1=Al, 2=V, 3=Fe)
    el_map = {el: i for i, el in enumerate(['Ti', 'Al', 'V', 'Fe'])}
    df['el_idx'] = df['element'].map(el_map)
    
    # Convert to JAX arrays (Constant Memory)
    # Shape: (N_lines,)
    lines_wl = jnp.array(df['wavelength_nm'].values, dtype=jnp.float32)
    lines_aki = jnp.array(df['aki'].values, dtype=jnp.float32)
    lines_ek = jnp.array(df['ek_ev'].values, dtype=jnp.float32)
    lines_gk = jnp.array(df['gk'].values, dtype=jnp.float32)
    lines_ip = jnp.array(df['ip_ev'].values, dtype=jnp.float32)
    lines_z = jnp.array(df['sp_num'].values - 1, dtype=jnp.int32) # 0=Neutral, 1=Ion
    lines_el_idx = jnp.array(df['el_idx'].values, dtype=jnp.int32)
    
    print(f"Loaded {len(df)} spectral lines into VRAM.")
    return lines_wl, lines_aki, lines_ek, lines_gk, lines_ip, lines_z, lines_el_idx

# --- 3. PHYSICS KERNEL (JAX/GPU) ---

@jit
def saha_eggert_solver(T_eV, n_e, concentration_map, lines_ip, lines_z, lines_el_idx, lines_ek, lines_gk):
    """
    Vectorized Saha-Eggert Solver.
    Calculates the population density of the upper level for EVERY line simultaneously.
    
    Args:
        T_eV: Plasma Temperature
        n_e: Electron Density
        concentration_map: Array of [C_Ti, C_Al, C_V, C_Fe]
        lines_...: Atomic data arrays
    
    Returns:
        n_upper: Population density of the upper state for each line.
    """
    # 1. Partition Functions U(T) (Simplified approximation for speed)
    # Ideally, this uses a pre-computed texture interpolation.
    # Here assuming U ~ constant (valid for narrow T range) or simple T dependence.
    U_Z0 = 25.0 # Placeholder for Neutral Partition Function
    U_Z1 = 15.0 # Placeholder for Ion Partition Function
    
    # 2. Ionization Ratio (Saha)
    # Ratio = N_ion / N_neutral
    # Using specific IP for each element (mapped via lines_el_idx)
    # Note: We compute this per LINE to vectorize, though it's technically per SPECIES.
    # It's faster to compute redundant math on GPU than to gather/scatter.
    
    # Saha Equation Constants
    saha_const = 6.042e21
    
    # Calculate Ratio for each line's species
    # If line is Neutral (Z=0), we need Ratio to get Ion fraction? 
    # No, we need Neutral Fraction.
    # N_total = N_0 + N_1 = N_0 (1 + Ratio)
    # N_0 = N_total / (1 + Ratio)
    # N_1 = N_total * Ratio / (1 + Ratio)
    
    # Currently lines_ip contains IP for the species of the line.
    # If line is Neutral, IP is I -> II. If line is Ion, IP is II -> III (approx).
    
    ratio = (saha_const / n_e) * (T_eV**1.5) * jnp.exp(-lines_ip / T_eV)
    
    # Determine species population fraction based on ionization stage of the line
    # If line is Neutral (lines_z == 0): fraction = 1 / (1 + ratio)
    # If line is Ion (lines_z == 1):     fraction = ratio / (1 + ratio)
    
    # We use jnp.where for branchless execution
    pop_fraction = jnp.where(lines_z == 0, 1.0 / (1.0 + ratio), ratio / (1.0 + ratio))
    
    # 3. Total Number Density of Element (N_s)
    # N_s = Concentration * N_total_density (approx n_e / Z_avg)
    # Simple approx: N_s ~ concentration * n_e (for singly ionized plasma)
    element_conc = concentration_map[lines_el_idx]
    N_species = element_conc * n_e 
    
    # 4. Boltzmann Level Population
    # n_upper = N_stage * (g_k / U) * exp(-E_k / kT)
    U_val = jnp.where(lines_z == 0, U_Z0, U_Z1)
    
    n_upper = (N_species * pop_fraction) * (lines_gk / U_val) * jnp.exp(-lines_ek / T_eV)
    
    return n_upper

@jit
def compute_spectrum_snapshot(wl_grid, T_eV, n_e, concentrations, atomic_data):
    """
    Generates a spectrum for a single instant in time (T, ne).
    """
    (l_wl, l_aki, l_ek, l_gk, l_ip, l_z, l_el_idx) = atomic_data
    
    # 1. Solve Populations
    n_upper = saha_eggert_solver(T_eV, n_e, concentrations, l_ip, l_z, l_el_idx, l_ek, l_gk)
    
    # 2. Line Emissivity (Watts / m^3 / sr)
    # epsilon = (hc / 4pi lambda) * A * n_upper
    epsilon = (H_J * C_LIGHT / (4 * jnp.pi * l_wl * 1e-9)) * l_aki * n_upper
    
    # 3. Line Broadening (Voigt)
    # Stark Broadening ~ ne (Lorentzian)
    # Doppler Broadening ~ sqrt(T) (Gaussian)
    
    # gamma_stark = 2e-16 * n_e # Simplified Stark param (needs real data for accuracy)
    sigma_doppler = (l_wl * 7.16e-7 * jnp.sqrt(T_eV * 11604))
    
    # Instrument Function dominates in reality (Gaussian)
    sigma_inst = FWHM_INST / 2.355
    sigma_total = jnp.sqrt(sigma_doppler**2 + sigma_inst**2)
    
    # 4. Render to Pixel Grid (The heavy lifting)
    # We broadcast the grid against all lines: (Pixels, Lines)
    # Shape: (4096, 1) - (1, N_lines)
    diff = wl_grid[:, None] - l_wl[None, :]
    
    # Vectorized Voigt (approx)
    # Using Gaussian for speed if Stark is small, or Pseudo-Voigt
    # Here using simple Gaussian for throughput demonstration
    profile = jnp.exp(-0.5 * (diff / sigma_total)**2) / (sigma_total * jnp.sqrt(2*jnp.pi))
    
    # Sum contributions: Spectrum = Sum(Epsilon * Profile)
    intensity = jnp.sum(epsilon * profile, axis=1)
    
    return intensity

@jit
def time_integrated_spectrum(wl_grid, params, atomic_data):
    """
    The Master Kernel.
    Integrates the cooling trail of the plasma.
    params: [T_max, ne_max, C_Ti, C_Al, C_V, C_Fe]
    """
    T_max = params[0]
    ne_max = params[1]
    concs = params[2:]
    
    # Time Grid
    times = jnp.linspace(0, GATE_WIDTH, TIME_STEPS)
    dt = times[1] - times[0]
    
    # Cooling Laws (Power Law Decay)
    # T(t) = T_max * (t + t0)^-alpha
    # ne(t) = ne_max * (t + t0)^-beta
    t0 = 1e-6 # scale factor
    T_trail = T_max * (1 + times/t0)**(-0.5)
    ne_trail = ne_max * (1 + times/t0)**(-1.0)
    
    # Accumulator
    spectrum_accum = jnp.zeros_like(wl_grid)
    
    # Scan/Loop over time steps
    # JAX 'scan' is efficient for loops
    def step_fn(carry, inputs):
        T, ne = inputs
        # Only add if T > 0.4 eV (below this, emission is negligible/molecular)
        intensity = jnp.where(
            T > 0.4, 
            compute_spectrum_snapshot(wl_grid, T, ne, concs, atomic_data),
            jnp.zeros_like(wl_grid)
        )
        return carry + intensity * dt, None

    spectrum_accum, _ = jax.lax.scan(step_fn, spectrum_accum, (T_trail, ne_trail))
    
    return spectrum_accum

# --- 4. BATCH PROCESSING ---

def generate_manifold_chunk(batch_params, atomic_data):
    """
    Maps the kernel over a batch of parameters.
    pmap/vmap handles the parallelization.
    """
    wl_grid = jnp.linspace(WL_MIN, WL_MAX, PIXELS)
    
    # vmap over the batch dimension (axis 0 of params)
    batch_spectra = vmap(time_integrated_spectrum, in_axes=(None, 0, None))(
        wl_grid, batch_params, atomic_data
    )
    return batch_spectra

# --- 5. MAIN EXECUTION ---

def main():
    print(f"--- JAX Manifold Generator (Devices: {jax.local_device_count()}) ---")
    
    # 1. Load Data
    atomic_data = load_atomic_data()
    # Move atomic data to GPU
    atomic_data = tuple(jax.device_put(x) for x in atomic_data)
    
    # 2. Build Grid (Numpy CPU)
    print("Building Parameter Grid...")
    T_grid = np.linspace(TEMP_MIN, TEMP_MAX, TEMP_STEPS)
    ne_grid = np.geomspace(NE_MIN, NE_MAX, NE_STEPS)
    
    # Mixture Grid (Simplex)
    # Simple Ti-6Al-4V neighborhood scan
    # Fixed Ti base, varying Al (0-10%) and V (0-10%)
    al_range = np.linspace(0, 0.12, CONC_STEPS)
    v_range = np.linspace(0, 0.12, CONC_STEPS)
    fe_range = np.linspace(0.000, 0.005, 5) # Trace amount 0 - 0.5%
    
    params_list = []
    for T in T_grid:
        for ne in ne_grid:
            for al in al_range:
                for v in v_range:
                    for fe in fe_range:
                        # Normalize: Ti is remainder
                        ti = 1.0 - (al + v + fe)
                        if ti < 0:
                            continue
                        # [T, ne, Ti, Al, V, Fe]
                        params_list.append([T, ne, ti, al, v, fe])
    
    params_arr = np.array(params_list, dtype=np.float32)
    n_samples = len(params_arr)
    print(f"Total Manifold Size: {n_samples} spectra")
    
    # 3. Execution Loop
    BATCH_SIZE = 1000 # Adjust based on VRAM
    
    # Open HDF5
    with h5py.File(OUTPUT_PATH, 'w') as f:
        dset_spec = f.create_dataset("spectra", (n_samples, PIXELS), dtype='f4')
        dset_param = f.create_dataset("params", (n_samples, 6), dtype='f4')
        
        start_time = time.time()
        
        for i in range(0, n_samples, BATCH_SIZE):
            batch = params_arr[i : i+BATCH_SIZE]
            
            # Run JAX Kernel
            # Note: For multi-GPU, use pmap here by reshaping batch to (n_devices, batch/n, ...)
            spectra = generate_manifold_chunk(batch, atomic_data)
            
            # Block until ready (copy back to CPU)
            spectra_np = np.array(spectra)
            
            # Save
            dset_spec[i : i+BATCH_SIZE] = spectra_np
            dset_param[i : i+BATCH_SIZE] = batch
            
            if i % (BATCH_SIZE*10) == 0:
                print(f"Generated {i}/{n_samples} ({i/n_samples:.1%})")
                
        total_time = time.time() - start_time
        print(f"Completed in {total_time:.2f}s ({n_samples/total_time:.0f} spectra/sec)")

if __name__ == "__main__":
    main()
