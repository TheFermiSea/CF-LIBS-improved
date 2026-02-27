import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import nnls
from scipy.special import wofz

class LineIdentifier:
    """
    Production-Grade Line Identifier based on High-Throughput Deconvolution.
    
    Implements the workflow:
    1. Non-linear Background Subtraction (AirPLS/BEADS equivalent)
    2. Matrix-Based Deconvolution (NNLS) to identify and quantify lines
    
    Reference: 'High-Throughput Calibration-Free Laser-Induced Breakdown Spectroscopy'
    """
    
    def __init__(self, db_conn, instrument_resolution_nm=0.05):
        self.conn = db_conn
        self.resolution = instrument_resolution_nm

    def airPLS(self, x, lambda_=100, porder=1, itermax=15):
        """
        Adaptive Iteratively Reweighted Penalized Least Squares.
        Removes broad Blackbody/Bremsstrahlung background while preserving peak areas.
        
        Parameters:
            lambda_: Smoothness parameter (higher = stiffer baseline)
        """
        m = x.shape[0]
        w = np.ones(m)
        for i in range(itermax):
            d = np.diff(np.eye(m), 2)
            D = csc_matrix(np.diff(d, axis=0))
            W = diags(w, 0, shape=(m, m))
            Z = W + lambda_ * D.dot(D.transpose())
            z = spsolve(Z, w * x)
            d = x - z
            dssn = np.abs(d[d < 0].sum())
            if dssn < 0.001 * (abs(x)).sum() or i == itermax - 1:
                return z
            w[d >= 0] = 0
            w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
            w[0] = np.exp(i * (d[d < 0]).max() / dssn) 
            w[-1] = w[0]
        return z

    def _pseudo_voigt(self, x, center, amplitude, sigma, fraction=0.5):
        """
        Vectorized Pseudo-Voigt profile (Sum of Gaussian and Lorentzian).
        Used to construct the design matrix for deconvolution.
        """
        z = (x - center) / (sigma * np.sqrt(2))
        # Approximation using Faddeeva (wofz) is accurate but slow.
        # For High-Throughput, we use the linear combo approximation:
        # PV(x) = eta * L(x) + (1-eta) * G(x)
        
        # Gaussian Component
        G = np.exp(-0.5 * ((x - center) / sigma)**2)
        G /= (sigma * np.sqrt(2 * np.pi))
        
        # Lorentzian Component
        gamma = sigma # Approximation for similar FWHM
        L = (1 / np.pi) * (gamma / ((x - center)**2 + gamma**2))
        
        return amplitude * (fraction * L + (1 - fraction) * G)

    def process_spectrum(self, wavelengths, intensities, candidate_elements):
        """
        The Main Pipeline:
        1. Remove Background
        2. Fetch Candidate Lines
        3. Deconvolve (NNLS) to find true intensities
        """
        # 1. Background Subtraction
        baseline = self.airPLS(intensities)
        signal = intensities - baseline
        # Clip negative noise to 0 for NNLS stability
        signal = np.maximum(signal, 0) 
        
        results = []
        
        # 2. Windowed Deconvolution (High Throughput)
        # We break the spectrum into chunks to keep matrix size manageable
        chunk_size_nm = 10.0
        start_wl = wavelengths.min()
        end_wl = wavelengths.max()
        
        current_start = start_wl
        while current_start < end_wl:
            current_end = current_start + chunk_size_nm
            
            # Slice Data
            mask = (wavelengths >= current_start) & (wavelengths < current_end)
            if np.sum(mask) < 10: 
                current_start += chunk_size_nm
                continue
                
            wl_chunk = wavelengths[mask]
            sig_chunk = signal[mask]
            
            # Fetch Candidates in this window
            # Only fetch lines that "could" exist (Ultrafast filter)
            placeholders = ','.join(['?']*len(candidate_elements))
            query = f"""
                SELECT element, wavelength_nm, aki, gk, ek_ev, sp_num
                FROM lines 
                WHERE wavelength_nm BETWEEN ? AND ?
                AND element IN ({placeholders})
                AND sp_num <= 2
            """
            params = [current_start - 0.5, current_end + 0.5] + candidate_elements
            db_lines = pd.read_sql_query(query, self.conn, params=params)
            
            if db_lines.empty:
                current_start += chunk_size_nm
                continue
                
            # 3. Construct Design Matrix (A)
            # A has shape (n_pixels, n_lines). Each column is a Unit Profile.
            # Equation: Signal = A * Intensities
            
            n_pixels = len(wl_chunk)
            n_lines = len(db_lines)
            A = np.zeros((n_pixels, n_lines))
            
            # Width Estimation: Use instrument resolution / 2.355 for sigma
            sigma = self.resolution / 2.355 
            
            for i, (_, line) in enumerate(db_lines.iterrows()):
                profile = self._pseudo_voigt(wl_chunk, line['wavelength_nm'], 1.0, sigma)
                A[:, i] = profile
                
            # 4. Solve NNLS (Non-Negative Least Squares)
            # This finds the BEST combination of lines to explain the spectrum
            # It naturally sets non-existent lines to 0.0
            fitted_intensities, residual = nnls(A, sig_chunk)
            
            # 5. Store Results
            for i, intensity in enumerate(fitted_intensities):
                if intensity > 0:
                    line_data = db_lines.iloc[i].to_dict()
                    line_data['fitted_intensity'] = intensity
                    line_data['snr'] = intensity / (np.std(sig_chunk - A @ fitted_intensities) + 1e-9)
                    results.append(line_data)
            
            current_start += chunk_size_nm
            
        return pd.DataFrame(results)

    def identify_composition(self, wavelengths, intensities, target_elements=['Fe', 'Ti', 'Al', 'V']):
        """
        High-level wrapper to identify elements and return a clean DataFrame
        ready for the Boltzmann Solver.
        """
        print(f"--- Running NNLS Line Identification for {target_elements} ---")
        
        df_results = self.process_spectrum(wavelengths, intensities, target_elements)
        
        if df_results.empty:
            print("No lines identified.")
            return pd.DataFrame()
        
        # Filter weak hits (SNR check)
        # In NNLS, noise often gets fitted as tiny amplitudes.
        # We apply a dynamic threshold.
        df_clean = df_results[df_results['snr'] > 3.0].copy()
        
        print(f"Identified {len(df_clean)} valid spectral lines.")
        print(df_clean.groupby('element')['fitted_intensity'].sum().sort_values(ascending=False))
        
        return df_clean

# Usage Example
if __name__ == "__main__":
    import sqlite3
    import matplotlib.pyplot as plt
    
    # Mock DB connection (Requires your libs_production.db)
    if not os.path.exists("libs_production.db"):
        print("Please run datagen_v2.py first.")
    else:
        conn = sqlite3.connect("libs_production.db")
        
        # Create Identifier
        identifier = LineIdentifier(conn, instrument_resolution_nm=0.05)
        
        # Create Mock Data (Ti-6Al-4V style)
        wl = np.linspace(300, 400, 2048)
        # True Signal
        true_signal = (
            identifier._pseudo_voigt(wl, 334.9, 1000, 0.02) + # Ti
            identifier._pseudo_voigt(wl, 396.1, 500, 0.02) +  # Al
            identifier._pseudo_voigt(wl, 310.2, 300, 0.02)    # V
        )
        # Add Background + Noise
        bg = 200 * np.exp(-(wl-300)/100)
        noise = np.random.normal(0, 10, len(wl))
        spectrum = true_signal + bg + noise
        
        # Run Identification
        identified_lines = identifier.identify_composition(wl, spectrum, ['Ti', 'Al', 'V', 'Fe'])
        
        print("\nTop Lines Found:")
        print(identified_lines.sort_values('fitted_intensity', ascending=False).head(5))
