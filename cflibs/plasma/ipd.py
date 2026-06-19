"""Ionization-potential-depression (IPD) models for the Saha balance.

This module provides *selectable* continuum-lowering models that satisfy the
:class:`cflibs.plasma.saha_boltzmann.IPDModel` protocol
(``calculate_lowering(n_e_cm3, T_K) -> float`` returning the lowering ``Δχ`` in
eV).  They can be passed to :class:`~cflibs.plasma.saha_boltzmann.SahaBoltzmannSolver`
(and its JAX twin) via the ``ipd_model`` constructor argument.

The default solver behaviour is **unchanged**: the solver still defaults to its
Debye-Hückel model when ``ipd_model`` is ``None``.  The Stewart-Pyatt model here
is strictly opt-in — construct it explicitly (or via :func:`make_ipd_model`) and
pass it to the solver.

Stewart-Pyatt model
-------------------

The Stewart & Pyatt (1966) continuum lowering interpolates smoothly between the
weak-coupling **Debye-Hückel** limit (``λ_D ≫ R_0``) and the strong-coupling
**ion-sphere** limit (``λ_D ≪ R_0``):

.. math::

    \\Delta\\chi_{SP} = \\frac{3\\,(z+1)\\,e^2}{2 R_0}
        \\left[\\left(1 + (\\lambda_D / R_0)^3\\right)^{2/3}
               - (\\lambda_D / R_0)^2\\right]

with the ion-sphere radius :math:`R_0 = [3 / (4\\pi n_i)]^{1/3}` and the Debye
length :math:`\\lambda_D = \\sqrt{k_B T / (4\\pi n_e e^2)}` (Gaussian-CGS units,
matching :func:`cflibs.plasma.partition.ionization_potential_depression`).

The leading factor of ``3/2`` (rather than the original ``1/2``) is a fixed,
deliberate normalisation choice so that the **weak-coupling limit reduces to the
exact same Debye-Hückel ``Δχ`` already used elsewhere in this package** (the
partition-function cutoff and the existing default IPD), keeping the
interpolation self-consistent.  With ``x ≡ λ_D / R_0`` the bracket admits the
expansion ``(1 + x^3)^{2/3} - x^2 → (2/3) x^{-1}`` as ``x → ∞``, so

.. math::

    \\Delta\\chi_{SP} \\xrightarrow[\\lambda_D \\gg R_0]{}
        \\frac{(z+1)\\,e^2}{\\lambda_D} \\equiv \\Delta\\chi_{DH},

i.e. the canonical Debye-Hückel form.  In the opposite, strong-coupling limit
the bracket ``→ 1`` and

.. math::

    \\Delta\\chi_{SP} \\xrightarrow[\\lambda_D \\ll R_0]{}
        \\frac{3\\,(z+1)\\,e^2}{2 R_0} \\equiv \\Delta\\chi_{IS},

the ion-sphere result.  Because ``(1 + x^3)^{2/3} - x^2`` is monotonically
decreasing in ``x`` and ``R_0`` and ``λ_D`` both fall with ``n_e``, the lowering
``Δχ_SP`` is monotonically **increasing** in ``n_e``, and it always lies *below*
the bare Debye-Hückel value (it softens the unphysical Debye divergence at high
density), as expected physically.

References
----------
- Stewart, J. C. & Pyatt, K. D. (1966), ApJ 144, 1203
  ("Lowering of Ionization Potentials in Plasmas").
- Crowley, B. J. B. (2014), "Continuum lowering - A new perspective",
  High Energy Density Physics 13, 84 (arXiv:1309.1456) — gives the explicit
  Debye-Hückel and ion-sphere limits of the SP analytical formula.
- Pain, J.-C. (2022), Plasma 5, 4, 387 — restates the SP IPD in terms of
  ``λ_D / R_0`` with ``R_0 = [3 / (4π N_i)]^{1/3}``.
- Mihalas, D. (1978), *Stellar Atmospheres*, Eq. 9-106 (Debye-Hückel limit).
"""

from __future__ import annotations

import numpy as np

from cflibs.core.constants import C_LIGHT, E_CHARGE, J_TO_EV, KB

# Derived Gaussian-CGS helpers, built from the canonical SI constants in
# ``cflibs.core.constants`` exactly as in ``cflibs.plasma.partition`` and
# ``cflibs.plasma.saha_boltzmann`` so every Δχ in the package shares ONE unit
# convention.
_E_ESU = E_CHARGE * C_LIGHT * 10.0  # electron charge [esu = statcoulomb]
_KB_ERG = KB * 1.0e7  # Boltzmann constant [erg/K]
_ERG_TO_EV = J_TO_EV * 1.0e-7  # conversion factor erg -> eV


def debye_length_cm(n_e_cm3: float, T_K: float) -> float:
    """Electron Debye length ``λ_D`` in cm (Gaussian-CGS).

    .. math::

        \\lambda_D = \\sqrt{\\frac{k_B T}{4\\pi n_e e^2}}

    Identical convention to
    :func:`cflibs.plasma.partition.ionization_potential_depression`.

    Parameters
    ----------
    n_e_cm3 : float
        Electron density in cm⁻³ (must be > 0).
    T_K : float
        Temperature in Kelvin (must be > 0).

    Returns
    -------
    float
        Debye length in cm.
    """
    return float(np.sqrt(_KB_ERG * T_K / (4.0 * np.pi * n_e_cm3 * _E_ESU**2)))


def ion_sphere_radius_cm(n_i_cm3: float) -> float:
    """Ion-sphere (Wigner-Seitz) radius ``R_0`` in cm.

    .. math::

        R_0 = \\left[\\frac{3}{4\\pi n_i}\\right]^{1/3}

    Parameters
    ----------
    n_i_cm3 : float
        Ion number density in cm⁻³ (must be > 0).

    Returns
    -------
    float
        Ion-sphere radius in cm.
    """
    return float((3.0 / (4.0 * np.pi * n_i_cm3)) ** (1.0 / 3.0))


def stewart_pyatt_lowering(
    n_e_cm3: float,
    T_K: float,
    *,
    z_net: int = 0,
    n_i_cm3: float | None = None,
) -> float:
    """Stewart-Pyatt (1966) continuum lowering ``Δχ`` in eV.

    Implements the closed-form Stewart-Pyatt interpolation between the
    Debye-Hückel (weak-coupling) and ion-sphere (strong-coupling) limits — see
    the module docstring for the full derivation and limits:

    .. math::

        \\Delta\\chi_{SP} = \\frac{3\\,(z+1)\\,e^2}{2 R_0}
            \\left[(1 + x^3)^{2/3} - x^2\\right],
        \\qquad x \\equiv \\lambda_D / R_0 .

    The ``3/2`` prefactor normalises the weak-coupling limit to the package's
    canonical Debye-Hückel ``Δχ = (z+1) e²/λ_D``.

    Parameters
    ----------
    n_e_cm3 : float
        Electron density in cm⁻³.
    T_K : float
        Temperature in Kelvin.
    z_net : int, optional
        Net charge of the ion *being ionized* (0 for a neutral atom losing its
        first electron, 1 for a singly-ionized ion, ...).  The lowering scales
        with ``z_net + 1`` — the charge of the resulting ion (default 0).
    n_i_cm3 : float, optional
        Ion number density in cm⁻³ for the ion-sphere radius.  When ``None``
        (default) the quasi-neutral singly-ionized approximation ``n_i = n_e``
        is used, mirroring the single-stage assumption already implicit in this
        package's Saha balance.

    Returns
    -------
    float
        IPD ``Δχ`` in eV.  Returns 0 for non-physical inputs
        (``n_e_cm3 <= 0`` or ``T_K <= 0``).

    References
    ----------
    Stewart & Pyatt (1966) ApJ 144, 1203; Crowley (2014) HEDP 13, 84.
    """
    if n_e_cm3 <= 0.0 or T_K <= 0.0:
        return 0.0
    n_i = n_e_cm3 if n_i_cm3 is None else n_i_cm3
    if n_i <= 0.0:
        return 0.0

    lambda_D = debye_length_cm(n_e_cm3, T_K)
    R0 = ion_sphere_radius_cm(n_i)
    x = lambda_D / R0  # λ_D / R_0 : large => weak coupling, small => strong

    bracket = (1.0 + x**3) ** (2.0 / 3.0) - x**2
    # Numerical guard: the bracket is analytically positive for all x >= 0 but
    # catastrophic cancellation at very large x can produce a tiny negative
    # value; Δχ is a non-negative lowering, so floor at 0.
    bracket = max(bracket, 0.0)

    delta_chi_erg = 1.5 * (z_net + 1) * _E_ESU**2 / R0 * bracket
    return float(delta_chi_erg * _ERG_TO_EV)


class StewartPyattIPD:
    """Selectable Stewart-Pyatt IPD model for the Saha balance.

    Satisfies the :class:`cflibs.plasma.saha_boltzmann.IPDModel` protocol so it
    can be passed as the ``ipd_model`` argument of
    :class:`~cflibs.plasma.saha_boltzmann.SahaBoltzmannSolver` (and its JAX
    twin).  **Strictly opt-in:** the solver default remains Debye-Hückel; this
    model is used only when explicitly supplied.

    Parameters
    ----------
    z_net : int, optional
        Net charge of the ion being ionized for the ``(z_net + 1)`` prefactor
        (default 0, i.e. neutral → singly-ionized — the dominant LIBS stage).
    n_i_cm3 : float, optional
        Fixed ion density for the ion-sphere radius.  When ``None`` (default)
        the quasi-neutral ``n_i = n_e`` approximation is used, consistent with
        the solver's single-stage Saha treatment.

    Examples
    --------
    >>> from cflibs.plasma.ipd import StewartPyattIPD
    >>> ipd = StewartPyattIPD()
    >>> dchi = ipd.calculate_lowering(1e17, 12000.0)
    >>> 0.0 < dchi < 0.1  # ~0.06 eV at canonical LIBS conditions
    True
    """

    def __init__(self, z_net: int = 0, n_i_cm3: float | None = None) -> None:
        self.z_net = int(z_net)
        self.n_i_cm3 = n_i_cm3

    def calculate_lowering(self, n_e_cm3: float, T_K: float) -> float:
        """Stewart-Pyatt ``Δχ`` in eV for the solver's IPD hook.

        Parameters
        ----------
        n_e_cm3 : float
            Electron density in cm⁻³.
        T_K : float
            Temperature in Kelvin.

        Returns
        -------
        float
            IPD ``Δχ`` in eV (>= 0).
        """
        return stewart_pyatt_lowering(n_e_cm3, T_K, z_net=self.z_net, n_i_cm3=self.n_i_cm3)


#: Registry of selectable IPD models by name.  ``"debye_huckel"`` resolves to
#: the package default (the existing Debye-Hückel model in ``saha_boltzmann``);
#: ``"stewart_pyatt"`` selects the model in this module.  ``"none"`` / ``None``
#: signal "use the solver default" (still Debye-Hückel — defaults unchanged).
_IPD_MODEL_NAMES = ("none", "debye_huckel", "stewart_pyatt")


def make_ipd_model(name: str | None):
    """Construct a selectable IPD model by name (strictly opt-in factory).

    Parameters
    ----------
    name : str or None
        One of ``"stewart_pyatt"``, ``"debye_huckel"``, ``"none"`` (or
        ``None``).  ``"none"`` / ``None`` returns ``None`` so the caller keeps
        the solver's unchanged default (Debye-Hückel).

    Returns
    -------
    IPDModel or None
        A model instance, or ``None`` to defer to the solver default.

    Raises
    ------
    ValueError
        If ``name`` is not a recognised model.
    """
    if name is None:
        return None
    key = name.strip().lower()
    if key in ("none", ""):
        return None
    if key == "stewart_pyatt":
        return StewartPyattIPD()
    if key == "debye_huckel":
        # Import lazily to avoid a circular import at module load time.
        from cflibs.plasma.saha_boltzmann import DebyeHuckelIPD

        return DebyeHuckelIPD()
    raise ValueError(f"Unknown ipd_model {name!r}. Choose one of {_IPD_MODEL_NAMES}.")
