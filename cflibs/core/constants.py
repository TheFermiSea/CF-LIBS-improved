"""
Physical constants for CF-LIBS calculations.

All constants are in SI base units unless otherwise specified.
For convenience, some constants are also provided in eV-based units.
"""

import numpy as np

# ============================================================================
# Fundamental Constants
# ============================================================================

# Boltzmann constant
KB = 1.380649e-23  # J/K
KB_EV = 8.617333262e-5  # eV/K

# Planck constant
H_PLANCK = 6.62607015e-34  # J·s
H_PLANCK_EV = 4.135667696e-15  # eV·s

# Reduced Planck constant
HBAR = H_PLANCK / (2.0 * np.pi)  # J·s
HBAR_EV = H_PLANCK_EV / (2.0 * np.pi)  # eV·s

# Speed of light
C_LIGHT = 2.99792458e8  # m/s

# Electron properties
M_E = 9.1093837015e-31  # kg (electron mass)
E_CHARGE = 1.602176634e-19  # C (elementary charge)

# ============================================================================
# Atomic Physics Constants
# ============================================================================

# Proton mass (CODATA 2018)
M_PROTON = 1.67262192369e-27  # kg

# Rydberg constant
RYDBERG = 10973731.568160  # m^-1

# Bohr radius
A_BOHR = 5.29177210903e-11  # m

# Fine structure constant
ALPHA_FS = 7.2973525693e-3

# ============================================================================
# Conversion Factors
# ============================================================================

# Energy conversions
EV_TO_J = E_CHARGE  # 1 eV = E_CHARGE J
J_TO_EV = 1.0 / EV_TO_J

# Wavelength conversions
CM_TO_EV = 1.23984193e-4  # cm^-1 to eV
EV_TO_CM = 1.0 / CM_TO_EV

# Temperature conversions
K_TO_EV = KB_EV  # K to eV (at room temp, ~0.025 eV)
EV_TO_K = 1.0 / K_TO_EV

# ============================================================================
# Plasma Physics Constants
# ============================================================================

# Saha equation constant (pre-factor)
# n_{z+1} * n_e / n_z = (2π m_e k_B T / h^2)^(3/2) * (2 U_{z+1} / U_z) * exp(-χ/kT)
# The constant (2π m_e k_B / h^2)^(3/2) in cm^-3 units.
# NOTE: The factor of 2 for electron spin degeneracy (g_e = 2) is INCLUDED in this
# value.  The partition function ratio U_{z+1}/U_z used with this constant should NOT
# contain an additional factor of 2 for the free electron.
# Derivation: (2π × 9.109e-31 × 1.381e-23 / (6.626e-34)^2)^{3/2} × 2 × 1e-6 ≈ 6.04e21
SAHA_CONST_CM3 = 6.042e21  # cm^-3 (at T=1 K, but scales as T^1.5)

# McWhirter criterion constant for LTE validity
# n_e >= 1.6e12 * T^(1/2) * (ΔE)^3
MCWHIRTER_CONST = 1.6e12  # cm^-3 (for ΔE in eV, T in K)

# ============================================================================
# Standard Conditions
# ============================================================================

# Standard temperature and pressure
STP_TEMP = 273.15  # K
STP_PRESSURE = 101325.0  # Pa (1 atm)

# Loschmidt number (particles per m^3 at STP)
LOSCHMIDT = 2.686780111e25  # m^-3

# ============================================================================
# Numerical Constants
# ============================================================================

# Small number for numerical stability
EPSILON = np.finfo(np.float64).eps

# Large number for bounds
LARGE = 1e10
