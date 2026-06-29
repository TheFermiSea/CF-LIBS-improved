"""ExoJAX-grade REFERENCE forward model for CF-LIBS.

Single-zone LTE plasma EMISSION computed with ExoJAX's atomic-line opacity engine
(Voigt-Hjerting line profiles + line-strength summation, `exojax.opacity.lpf.lpf`)
driven by OUR Saha-Boltzmann level populations. ExoJAX ships NO Saha solver, so we
supply the ionization/excitation balance and feed ExoJAX a per-line "line strength"
Sij = emission weight. The result is an optically-thin emissivity on a fixed
instrument grid (optional escape-factor self-absorption), instrument-convolved.

Physics chain (all in JAX, jit/vmap-clean, differentiable):
  PlasmaState(T, n_e, composition)
    -> partition functions U_stage(T)  = exp(sum_n a_n (ln T)^n)        [from bundle]
    -> Saha ratio  n_II/n_I = (2 U_II/U_I)(2 pi m_e kT/h^2)^1.5 (2/n_e) e^{-IP/kT}
       (done in LOG-SPACE via lax.logistic -> stage fractions f_I, f_II; float32-safe)
    -> Boltzmann level pop weight  (g_k A_ki / U_stage) e^{-E_k/kT}
    -> per-line emission line strength  Sij = C_elem * f_stage * boltz * (hc/lambda)
    -> ExoJAX xsvector(numatrix, sigmaD, gammaL, Sij)  [Voigt-Hjerting opacity sum]
       sigmaD = Doppler sigma (T, mass);  gammaL = Stark Lorentz HWHM (n_e)
    -> optional escape-factor self-absorption (optically-thin <-> thick interpolation)
    -> instrument Gaussian convolution on the fixed grid.

The atomic-line set is a pluggable static bundle (see build_atomic_bundle.py).

theta parameterization for inversion (matches varpro_bench.py 17-param layout but
generalized to N_elem):  theta = [T(K), log10(n_e), raw_0 .. raw_{Nelem-1}]
with composition = softmax(raw) (closure on the simplex, smooth, scale-free).

A `forward_with_basis(T, ne)` factory returns the (N_grid, N_elem) per-species basis
B so that S = B @ composition is LINEAR in composition -- this is what powers the
structured-Jacobian K=1 Gauss-Newton inversion (15+ conc columns with NO autodiff).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

# ExoJAX opacity engine (Voigt-Hjerting line profile + line-strength summation).
from exojax.opacity.lpf.lpf import xsvector as exojax_xsvector
from exojax.opacity.lpf.lpf import vvoigt as exojax_vvoigt

# ---------------------------------------------------------------------------
# Physical constants (CODATA, SI unless noted).
# ---------------------------------------------------------------------------
KB_EV = 8.617333262e-5          # Boltzmann constant [eV/K]
KB_J = 1.380649e-23             # [J/K]
H_J = 6.62607015e-34            # Planck [J s]
ME_KG = 9.1093837015e-31        # electron mass [kg]
C_M = 2.99792458e8              # speed of light [m/s]
C_CM = 2.99792458e10            # [cm/s]
HC_EVNM = 1239.841984           # hc [eV nm]
# Saha prefactor  2 (2 pi m_e k / h^2)^1.5  [m^-3 K^-1.5], times the 2 (stat) handled
# explicitly below. (2 pi m_e k / h^2)^1.5 = 2.4146e21 m^-3 K^-1.5.
SAHA_GEOM = 2.0 * (2.0 * np.pi * ME_KG * KB_J / (H_J * H_J)) ** 1.5  # m^-3 K^-1.5
DOPP_FWHM_COEF = 7.16e-7        # Doppler FWHM = lambda * coef * sqrt(T/M)  (nm, amu, K)


@dataclass(frozen=True)
class AtomicBundle:
    """Static atomic-line bundle (device arrays). Pluggable -- the forward model is
    agnostic to which elements/lines it carries."""

    elements: tuple
    elem_mass: jnp.ndarray       # (E,)
    elem_ip_eV: jnp.ndarray      # (E,)
    U_coeffs: jnp.ndarray        # (E, 2, 5) ln-poly coeffs
    line_wl_nm: jnp.ndarray      # (L,)
    line_gA: jnp.ndarray         # (L,)  g_k * A_ki
    line_Ek_eV: jnp.ndarray      # (L,)
    line_elem: jnp.ndarray       # (L,) int
    line_stage: jnp.ndarray      # (L,) int 0/1
    line_stark_w: jnp.ndarray    # (L,)  Stark HWHM ref [nm] at ne=1e17
    n_elem: int
    n_lines: int

    @staticmethod
    def from_npz(path):
        d = np.load(path, allow_pickle=True)
        elements = tuple(str(x) for x in d["elements"])
        return AtomicBundle(
            elements=elements,
            elem_mass=jnp.asarray(d["elem_mass"], dtype=jnp.float32),
            elem_ip_eV=jnp.asarray(d["elem_ip_eV"], dtype=jnp.float32),
            U_coeffs=jnp.asarray(d["U_coeffs"], dtype=jnp.float32),
            line_wl_nm=jnp.asarray(d["line_wl_nm"], dtype=jnp.float32),
            line_gA=jnp.asarray(d["line_gA"], dtype=jnp.float32),
            line_Ek_eV=jnp.asarray(d["line_Ek_eV"], dtype=jnp.float32),
            line_elem=jnp.asarray(d["line_elem"], dtype=jnp.int32),
            line_stage=jnp.asarray(d["line_stage"], dtype=jnp.int32),
            line_stark_w=jnp.asarray(d["line_stark_w"], dtype=jnp.float32),
            n_elem=len(elements),
            n_lines=int(d["line_wl_nm"].shape[0]),
        )


@dataclass(frozen=True)
class InstrumentConfig:
    wl_grid_nm: np.ndarray       # fixed instrument grid (nm)
    fwhm_nm: float = 0.05        # instrument Gaussian FWHM (nm); used in sigmaD floor
    use_self_absorption: bool = False
    escape_tau_scale: float = 0.0  # if >0, escape-factor self-absorption strength


# ---------------------------------------------------------------------------
# Forward-model factory: binds an AtomicBundle + instrument grid, returns jit-clean
# differentiable functions. ExoJAX opacity works in WAVENUMBER (cm^-1); we convert.
# ---------------------------------------------------------------------------
def make_reference_forward(bundle: AtomicBundle, instr: InstrumentConfig):
    wl_grid = jnp.asarray(instr.wl_grid_nm, dtype=jnp.float32)        # (G,) nm
    nu_grid = 1.0e7 / wl_grid                                          # (G,) cm^-1
    line_nu = 1.0e7 / bundle.line_wl_nm                                # (L,) cm^-1
    # ExoJAX numatrix convention: numatrix[line, grid] = nu_grid - line_nu.
    numatrix = nu_grid[None, :] - line_nu[:, None]                     # (L, G)

    le = bundle.line_elem
    ls = bundle.line_stage
    n_elem = bundle.n_elem
    n_lines = bundle.n_lines
    fwhm_instr = jnp.float32(instr.fwhm_nm)
    # one-hot (L, E) for fast per-element basis assembly (linear in composition)
    elem_onehot = (le[:, None] == jnp.arange(n_elem)[None, :]).astype(jnp.float32)

    def _partition(T):
        """U_stage(T) per (element, stage): exp(sum_n a_n (ln T)^n). -> (E, 2)."""
        lnT = jnp.log(T)
        powers = jnp.stack([lnT ** k for k in range(bundle.U_coeffs.shape[-1])])  # (5,)
        lnU = jnp.einsum("esc,c->es", bundle.U_coeffs, powers)                    # (E,2)
        return jnp.exp(lnU)

    def _line_profile_terms(T, ne):
        """Compute everything except composition: per-line emission weight (unit
        concentration) and the Voigt profile broadening params, all (L,)-shaped.

        Returns (Sij_unit, sigmaD_nu, gammaL_nu):
            Sij_unit : per-line emission line strength at C_elem = 1 [arb. emissivity]
            sigmaD_nu: Doppler sigma in WAVENUMBER (cm^-1) for ExoJAX
            gammaL_nu: Lorentz HWHM in WAVENUMBER (cm^-1) for ExoJAX
        """
        kT = KB_EV * T
        U = _partition(T)                                   # (E, 2)
        U_I = U[:, 0]
        U_II = U[:, 1]
        # --- Saha ionization balance (LOG-SPACE -> stage fractions) ---
        # log(n_II/n_I) = log(SAHA_GEOM) - log(ne) + 1.5 log(T)
        #               + log(U_II/U_I) - IP/kT     (the leading 2 stat already in SAHA_GEOM)
        log_saha = (
            jnp.log(SAHA_GEOM) - jnp.log(ne) + 1.5 * jnp.log(T)
            + jnp.log(U_II / U_I) - bundle.elem_ip_eV / kT
        )                                                   # (E,)
        f_II = jax.lax.logistic(log_saha)                   # n_II / (n_I + n_II)
        f_I = jax.lax.logistic(-log_saha)
        stage_frac = jnp.stack([f_I, f_II], axis=1)         # (E, 2)

        sf_line = stage_frac[le, ls]                        # (L,)
        U_line = U[le, ls]                                  # (L,)
        # Boltzmann population of the upper level x A_ki (emission), per unit n_elem:
        #   emiss_weight = (g_k A_ki / U_stage) e^{-E_k/kT} * (hc/lambda)  [photon->energy]
        boltz = (bundle.line_gA / U_line) * jnp.exp(-bundle.line_Ek_eV / kT)
        photon_E = HC_EVNM / bundle.line_wl_nm              # (L,) eV (per-photon energy)
        Sij_unit = sf_line * boltz * photon_E              # (L,)  emissivity weight

        # --- broadening in wavenumber for ExoJAX ---
        mass_line = bundle.elem_mass[le]                    # (L,) amu
        # Doppler FWHM in nm, convert to wavenumber HWHM->sigma below.
        dopp_fwhm_nm = bundle.line_wl_nm * DOPP_FWHM_COEF * jnp.sqrt(T / mass_line)
        # INSTRUMENT BROADENING folded into the Gaussian core: convolving a Voigt
        # with a Gaussian instrument function yields a Voigt with
        #   sigma_total^2 = sigma_Doppler^2 + sigma_instr^2  (Lorentz part unchanged).
        # This is exact and removes the separate (cuDNN-fragile, slow) convolution.
        sigma_instr_nm = fwhm_instr / 2.35482
        sigmaD_nm = jnp.sqrt((dopp_fwhm_nm / 2.35482) ** 2 + sigma_instr_nm ** 2)
        # Stark Lorentz HWHM (nm) scales linearly with n_e (ref at 1e17 cm^-3)
        gammaL_nm = bundle.line_stark_w * (ne / 1.0e17) + 1.0e-5
        # nm -> wavenumber (cm^-1): |dnu| = 1e7/lambda^2 * dlambda
        nm_to_nu = 1.0e7 / (bundle.line_wl_nm ** 2)        # (L,) cm^-1 per nm
        sigmaD_nu = sigmaD_nm * nm_to_nu + 1.0e-4
        gammaL_nu = gammaL_nm * nm_to_nu + 1.0e-6
        return Sij_unit, sigmaD_nu, gammaL_nu

    def _convolve(spec):
        # Instrument broadening is folded EXACTLY into the Voigt Gaussian core
        # (sigma_instr added in quadrature in _line_profile_terms), so the spectrum
        # is already instrument-resolved here. We deliberately do NOT call
        # jnp.convolve: on this V100S/cuDNN stack it lowers to __cudnn$convForward
        # which fails to autotune (CUDNN_STATUS 5003), and a Gaussian(x)Voigt is a
        # Voigt anyway -- so this is both correct and sub-ms-fast.
        return spec

    # ---------------- public differentiable forward functions ----------------

    def emissivity_lines(T, ne, comp):
        """Optically-thin EMISSION spectrum via ExoJAX Voigt-Hjerting opacity.
        comp: (E,) composition (number fractions / simplex). Returns (G,)."""
        Sij_unit, sigmaD_nu, gammaL_nu = _line_profile_terms(T, ne)
        c_line = comp[le]                                   # (L,) elemental weight
        Sij = c_line * Sij_unit                             # (L,) full line strength
        # ExoJAX cross-section vector = sum_lines Voigt(numatrix) * Sij  -> (G,)
        emis = exojax_xsvector(numatrix, sigmaD_nu, gammaL_nu, Sij)
        return emis

    def _maybe_self_absorb(emis_thin, T, ne, comp):
        if not instr.use_self_absorption or instr.escape_tau_scale <= 0.0:
            return emis_thin
        # Escape-factor self-absorption: tau ~ scale * emis_thin (proxy), the
        # standard optically-thin<->thick interpolation  (1 - e^{-tau}) / tau.
        tau = instr.escape_tau_scale * emis_thin
        esc = jnp.where(tau > 1e-6, (1.0 - jnp.exp(-tau)) / (tau + 1e-30), 1.0)
        return emis_thin * esc

    def forward_comp(T, ne, comp):
        """Full reference forward: (T, ne, composition) -> instrument spectrum (G,)."""
        emis = emissivity_lines(T, ne, comp)
        emis = _maybe_self_absorb(emis, T, ne, comp)
        return _convolve(emis)

    def conc_from_theta(theta):
        raw = theta[2:]
        e = jnp.exp(raw - jnp.max(raw))
        return e / jnp.sum(e)

    def forward(theta):
        """theta = [T, log10 ne, raw...] -> spectrum (G,)."""
        T = theta[0]
        ne = 10.0 ** theta[1]
        comp = conc_from_theta(theta)
        return forward_comp(T, ne, comp)

    def forward_with_basis(T, ne):
        """(G, E) per-species basis B: column s = spectrum of element s at unit
        concentration (after instrument convolution). Full spectrum = B @ comp.
        Linear in composition -> structured Jacobian conc columns need NO autodiff."""
        Sij_unit, sigmaD_nu, gammaL_nu = _line_profile_terms(T, ne)
        # per-line Voigt profiles on the grid: (L, G)
        profiles = exojax_vvoigt(numatrix, sigmaD_nu, gammaL_nu)     # (L, G)
        weighted = profiles * Sij_unit[:, None]                     # (L, G) unit-conc
        # collapse lines -> element basis, then convolve each column
        elem_emis = jnp.einsum("lg,le->eg", weighted, elem_onehot)  # (E, G)
        if instr.use_self_absorption and instr.escape_tau_scale > 0.0:
            # self-absorption is nonlinear; apply per-column as a proxy (kept off by
            # default so the structured-GN linearity assumption holds exactly).
            tau = instr.escape_tau_scale * elem_emis
            esc = jnp.where(tau > 1e-6, (1.0 - jnp.exp(-tau)) / (tau + 1e-30), 1.0)
            elem_emis = elem_emis * esc
        return elem_emis.T                                          # (G, E)

    return dict(
        forward=forward,
        forward_comp=forward_comp,
        forward_with_basis=forward_with_basis,
        conc_from_theta=conc_from_theta,
        emissivity_lines=emissivity_lines,
        wl_grid=wl_grid,
        n_elem=n_elem,
        n_lines=n_lines,
    )
