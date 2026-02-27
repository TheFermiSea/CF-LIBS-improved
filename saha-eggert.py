# saha-eggert.py
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import wofz
import os

# --- CONSTANTS ---
KB_EV = 8.617e-5         # Boltzmann constant (eV/K)
H_PLANCK = 4.135e-15     # Planck constant (eV s)

class LibsEngine:
    def __init__(self, db_path):
        if not os.path.exists(db_path):
            raise FileNotFoundError("Database not found. Run datagen_v2.py first.")
        self.conn = sqlite3.connect(db_path)
        
    def get_ionization_potential(self, element, sp_num):
        """Get IP from the species_physics table"""
        query = "SELECT ip_ev FROM species_physics WHERE element=? AND sp_num=?"
        cur = self.conn.cursor()
        cur.execute(query, (element, sp_num))
        res = cur.fetchone()
        return res[0] if res else None

    def get_lines(self, element, sp_num, min_wl, max_wl):
        query = """
            SELECT wavelength_nm, aki, ek_ev, gk 
            FROM lines 
            WHERE element = ? AND sp_num = ? 
            AND wavelength_nm BETWEEN ? AND ?
        """
        return pd.read_sql_query(query, self.conn, params=(element, sp_num, min_wl, max_wl))

    def saha_eggert_ratio(self, element, T_eV, Ne_cm3):
        """
        Calculates N_II / N_I ratio using Saha-Eggert Equation.
        """
        ip_ev = self.get_ionization_potential(element, 1)
        if not ip_ev: return 0.0
        
        # Saha-Eggert Ratio (Simplified for speed)
        # Ratio = N(z+1)/N(z)
        saha_factor = (6.04e21 / Ne_cm3) * (T_eV**1.5) * np.exp(-ip_ev / T_eV)
        return saha_factor

    def generate_snapshot(self, element, T_eV, Ne_cm3, min_wl, max_wl, x_grid):
        """
        Generates a SINGLE instant in time (T, Ne).
        Returns y array.
        """
        # 1. Calculate Balance
        ion_neutral_ratio = self.saha_eggert_ratio(element, T_eV, Ne_cm3)
        frac_I = 1.0 / (1.0 + ion_neutral_ratio)
        frac_II = 1.0 - frac_I
        
        # 2. Get Lines
        df_I = self.get_lines(element, 1, min_wl, max_wl)
        df_II = self.get_lines(element, 2, min_wl, max_wl)
        
        y_snapshot = np.zeros_like(x_grid)
        
        # 3. Compute Intensities
        if not df_I.empty:
            df_I['I'] = frac_I * (df_I['gk'] * df_I['aki']) * np.exp(-df_I['ek_ev'] / T_eV)
            y_snapshot += self._render_lines(df_I, x_grid)
            
        if not df_II.empty:
            df_II['I'] = frac_II * (df_II['gk'] * df_II['aki']) * np.exp(-df_II['ek_ev'] / T_eV)
            y_snapshot += self._render_lines(df_II, x_grid)
            
        return y_snapshot

    def _render_lines(self, df, x_grid):
        """Render lines onto grid using Voigt profiles"""
        y = np.zeros_like(x_grid)
        # Filter weak lines for speed
        max_I = df['I'].max()
        cutoff = max_I * 0.005 # 0.5% cutoff
        
        # Mechelle Resolution approx 0.05nm
        sigma = 0.05 
        gamma = 0.02 # Lorentzian component
        
        # Pre-calculate constants
        sqrt2 = np.sqrt(2)
        
        for _, line in df[df['I'] > cutoff].iterrows():
            z = (x_grid - line['wavelength_nm'] + 1j*gamma) / (sigma * sqrt2)
            profile = line['I'] * np.real(wofz(z))
            y += profile
        return y

    def generate_integrated_spectrum(self, element, T_max, Ne_max, min_wl, max_wl, steps=10):
        """
        INDUSTRIAL MODE: Simulates the ICCD 'Delayed Integration'.
        Integrates the spectrum as the plasma cools from T_max -> T_min.
        This is what your Manifold Generator should run.
        """
        x_grid = np.linspace(min_wl, max_wl, 5000)
        y_integrated = np.zeros_like(x_grid)
        
        # Cooling Model: Simple linear decay for now (can upgrade to t^-b)
        # T drops to 40% of max, Ne drops to 10% of max
        temps = np.linspace(T_max, T_max * 0.4, steps)
        densities = np.linspace(Ne_max, Ne_max * 0.1, steps)
        
        print(f"--- Integrating Cooling Trail for {element} ---")
        print(f"Gate Open: {T_max} eV, {Ne_max:.1e} cm-3")
        
        for T, Ne in zip(temps, densities):
            y_snap = self.generate_snapshot(element, T, Ne, min_wl, max_wl, x_grid)
            y_integrated += y_snap
            
        # Normalize
        if y_integrated.max() > 0:
            y_integrated /= y_integrated.max()
            
        return x_grid, y_integrated

# --- MAIN ---
if __name__ == "__main__":
    if os.path.exists("libs_production.db"):
        engine = LibsEngine("libs_production.db")
        
        # Simulate Ti64 Component (Titanium)
        # Ultrafast plasma starts hot (1.2 eV) but we integrate until it's cold
        x, y = engine.generate_integrated_spectrum("Ti", T_max=1.2, Ne_max=1e17, min_wl=300, max_wl=400)
        
        plt.figure(figsize=(10, 5))
        plt.plot(x, y, color='k', label='Simulated Integrated Spectrum (5µs Gate)')
        plt.title("Ti Integrated Spectrum (Simulation of Trailing Sensor)")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Normalized Intensity")
        plt.legend()
        plt.show()
    else:
        print("Please run datagen_v2.py first!")
