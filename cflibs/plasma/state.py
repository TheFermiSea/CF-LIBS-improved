"""
Plasma state representations and composition conversion helpers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from cflibs.core.constants import KB_EV
from cflibs.core.jax_runtime import HAS_JAX, jnp  # noqa: F401  -- re-exported for callers
from cflibs.core.logging_config import get_logger

if HAS_JAX:
    import jax.core

    _JAX_TRACER: type = jax.core.Tracer
else:
    _JAX_TRACER = type(None)  # sentinel that will never match


def _is_jax_tracer_or_array(value) -> bool:
    """Return True if ``value`` is a JAX tracer or array (not a concrete scalar).

    Under ``jax.jit`` or ``jax.vmap`` the plasma constructor receives tracers
    or batched arrays; formatting them with ``f"{T_e:.1f}"`` raises
    ``ConcretizationTypeError``. Use this to gate the constructor's
    ``logger.info`` call.
    """
    if HAS_JAX and isinstance(value, _JAX_TRACER):
        return True
    return hasattr(value, "ndim")


logger = get_logger("plasma.state")


def _normalize_positive_fractions(
    fractions: Mapping[str, float],
    quantity_name: str,
) -> dict[str, float]:
    """Normalize positive fractions to sum to one."""
    non_finite = [el for el, v in fractions.items() if not math.isfinite(float(v))]
    if non_finite:
        raise ValueError(
            f"{quantity_name} contains non-finite values: {', '.join(sorted(non_finite))}"
        )
    negative_elements = [element for element, value in fractions.items() if float(value) < 0.0]
    if negative_elements:
        elements_str = ", ".join(sorted(negative_elements))
        raise ValueError(f"{quantity_name} cannot contain negative components: {elements_str}")

    normalized_input = {
        element: float(value) for element, value in fractions.items() if float(value) > 0.0
    }
    total = sum(normalized_input.values())
    if total <= 0.0:
        raise ValueError(f"{quantity_name} must contain at least one positive component")
    return {element: value / total for element, value in normalized_input.items()}


def species_densities_to_number_fractions(species: Mapping[str, float]) -> dict[str, float]:
    """
    Convert element number densities to normalized number fractions.

    Parameters
    ----------
    species : Mapping[str, float]
        Element number densities in cm^-3.

    Returns
    -------
    dict[str, float]
        Element number fractions summing to 1.0.
    """
    return _normalize_positive_fractions(species, "species")


def number_fractions_to_species_densities(
    number_fractions: Mapping[str, float],
    total_number_density_cm3: float,
) -> dict[str, float]:
    """
    Convert number fractions to element number densities.

    Parameters
    ----------
    number_fractions : Mapping[str, float]
        Element number fractions.
    total_number_density_cm3 : float
        Total heavy-particle number density in cm^-3.

    Returns
    -------
    dict[str, float]
        Element number densities in cm^-3.
    """
    if not math.isfinite(total_number_density_cm3) or total_number_density_cm3 <= 0.0:
        raise ValueError("total_number_density_cm3 must be finite and positive")

    normalized = _normalize_positive_fractions(number_fractions, "number_fractions")
    return {
        element: total_number_density_cm3 * fraction for element, fraction in normalized.items()
    }


def mass_fractions_to_number_fractions(
    mass_fractions: Mapping[str, float],
    atomic_masses_amu: Mapping[str, float],
) -> dict[str, float]:
    """
    Convert mass fractions to number fractions.

    Parameters
    ----------
    mass_fractions : Mapping[str, float]
        Element mass fractions.
    atomic_masses_amu : Mapping[str, float]
        Atomic masses in amu keyed by element symbol.

    Returns
    -------
    dict[str, float]
        Element number fractions summing to 1.0.
    """
    weighted: dict[str, float] = {}
    for element, raw_mass_fraction in mass_fractions.items():
        mass_fraction = float(raw_mass_fraction)
        if not math.isfinite(mass_fraction):
            raise ValueError(f"Mass fraction for element {element!r} must be finite")
        if mass_fraction < 0.0:
            raise ValueError(f"mass_fractions cannot contain negative components: {element}")
        if mass_fraction == 0.0:
            continue
        if element not in atomic_masses_amu:
            raise KeyError(f"Missing atomic mass for element {element!r}")
        atomic_mass = float(atomic_masses_amu[element])
        if not math.isfinite(atomic_mass):
            raise ValueError(f"Atomic mass for element {element!r} must be finite")
        if atomic_mass <= 0.0:
            raise ValueError(f"Atomic mass for element {element!r} must be positive")
        weighted[element] = mass_fraction / atomic_mass
    return _normalize_positive_fractions(weighted, "mass_fractions")


def mass_fractions_to_species_densities(
    mass_fractions: Mapping[str, float],
    total_number_density_cm3: float,
    atomic_masses_amu: Mapping[str, float],
) -> dict[str, float]:
    """
    Convert mass fractions directly to element number densities.

    Parameters
    ----------
    mass_fractions : Mapping[str, float]
        Element mass fractions.
    total_number_density_cm3 : float
        Total heavy-particle number density in cm^-3.
    atomic_masses_amu : Mapping[str, float]
        Atomic masses in amu keyed by element symbol.

    Returns
    -------
    dict[str, float]
        Element number densities in cm^-3.
    """
    number_fractions = mass_fractions_to_number_fractions(mass_fractions, atomic_masses_amu)
    return number_fractions_to_species_densities(number_fractions, total_number_density_cm3)


@dataclass
class PlasmaState:
    """
    Base plasma state representation.

    Attributes
    ----------
    T_e : float
        Electron temperature in K
    T_g : float
        Gas/ion temperature in K (optional, defaults to T_e)
    n_e : float
        Electron density in cm^-3
    species : dict[str, float]
        Element number densities in cm^-3 (key: element symbol). These are
        heavy-particle densities and are distinct from mass fractions, number
        fractions, and electron density.
    pressure : float | None
        Pressure in atm (optional)
    """

    T_e: float  # K
    n_e: float  # cm^-3
    species: dict[str, float]  # cm^-3
    T_g: float | None = None
    pressure: float | None = None

    @property
    def T_e_eV(self) -> float:
        """Electron temperature in eV."""
        return self.T_e * KB_EV

    @property
    def T_g_eV(self) -> float:
        """Gas temperature in eV."""
        if self.T_g is None:
            return self.T_e_eV
        return self.T_g * KB_EV

    @property
    def total_species_density_cm3(self) -> float:
        """Total heavy-particle number density in cm^-3."""
        return float(sum(self.species.values()))

    @property
    def species_number_fractions(self) -> dict[str, float]:
        """Element number fractions implied by ``species``."""
        return species_densities_to_number_fractions(self.species)


class SingleZoneLTEPlasma(PlasmaState):
    """
    Single-zone LTE plasma model.

    This is the simplest plasma model: a homogeneous, optically thin
    plasma in local thermodynamic equilibrium.
    """

    def __init__(
        self,
        T_e: float,
        n_e: float,
        species: dict[str, float],
        T_g: float | None = None,
        pressure: float | None = None,
    ):
        """
        Initialize single-zone LTE plasma.

        Parameters
        ----------
        T_e : float
            Electron temperature in K
        n_e : float
            Electron density in cm^-3
        species : dict[str, float]
            Element number densities in cm^-3
        T_g : float, optional
            Gas temperature in K (defaults to T_e)
        pressure : float, optional
            Pressure in atm
        """
        super().__init__(T_e, n_e, species, T_g, pressure)
        # Gate logging on tracer/array detection to avoid ConcretizationTypeError
        # when called inside JAX jit/vmap traces or with batched arrays.
        if not _is_jax_tracer_or_array(T_e):
            logger.info(
                f"Created SingleZoneLTEPlasma: T_e={T_e:.1f} K, n_e={n_e:.2e} cm^-3, "
                f"species={list(species.keys())}"
            )

    @classmethod
    def from_number_fractions(
        cls,
        T_e: float,
        n_e: float,
        number_fractions: Mapping[str, float],
        total_species_density_cm3: float,
        T_g: float | None = None,
        pressure: float | None = None,
    ) -> "SingleZoneLTEPlasma":
        """
        Build a plasma state from element number fractions.

        Parameters
        ----------
        T_e : float
            Electron temperature in K.
        n_e : float
            Electron density in cm^-3.
        number_fractions : Mapping[str, float]
            Element number fractions on a heavy-particle basis.
        total_species_density_cm3 : float
            Total heavy-particle number density in cm^-3.
        T_g : float, optional
            Gas temperature in K. Defaults to ``T_e`` when omitted.
        pressure : float, optional
            Pressure in atm.

        Returns
        -------
        SingleZoneLTEPlasma
            Plasma state with element number densities derived from the
            supplied number fractions.
        """
        species = number_fractions_to_species_densities(
            number_fractions,
            total_number_density_cm3=total_species_density_cm3,
        )
        return cls(T_e=T_e, n_e=n_e, species=species, T_g=T_g, pressure=pressure)

    @classmethod
    def from_mass_fractions(
        cls,
        T_e: float,
        n_e: float,
        mass_fractions: Mapping[str, float],
        total_species_density_cm3: float,
        atomic_masses_amu: Mapping[str, float],
        T_g: float | None = None,
        pressure: float | None = None,
    ) -> "SingleZoneLTEPlasma":
        """
        Build a plasma state from element mass fractions.

        Parameters
        ----------
        T_e : float
            Electron temperature in K.
        n_e : float
            Electron density in cm^-3.
        mass_fractions : Mapping[str, float]
            Element mass fractions.
        total_species_density_cm3 : float
            Total heavy-particle number density in cm^-3.
        atomic_masses_amu : Mapping[str, float]
            Atomic masses in amu keyed by element symbol.
        T_g : float, optional
            Gas temperature in K. Defaults to ``T_e`` when omitted.
        pressure : float, optional
            Pressure in atm.

        Returns
        -------
        SingleZoneLTEPlasma
            Plasma state with element number densities derived from the
            supplied mass fractions.
        """
        species = mass_fractions_to_species_densities(
            mass_fractions,
            total_number_density_cm3=total_species_density_cm3,
            atomic_masses_amu=atomic_masses_amu,
        )
        return cls(T_e=T_e, n_e=n_e, species=species, T_g=T_g, pressure=pressure)

    def validate(self) -> bool:
        """
        Validate plasma state.

        Returns
        -------
        bool
            True if valid

        Raises
        ------
        ValueError
            If plasma state is invalid
        """
        if self.T_e <= 0:
            raise ValueError("Electron temperature must be positive")

        if self.n_e <= 0:
            raise ValueError("Electron density must be positive")

        if not self.species:
            raise ValueError("At least one species must be specified")

        for element, density in self.species.items():
            if density <= 0:
                raise ValueError(f"Species density for {element} must be positive")

        total_species_density = self.total_species_density_cm3
        if total_species_density <= 0.0:
            raise ValueError("Total species density must be positive")

        # This is only a rough plausibility check. ``species`` stores heavy-particle
        # densities rather than charge density, so we only warn on extreme mismatch.
        if abs(self.n_e - total_species_density) / max(self.n_e, total_species_density) > 0.5:
            logger.warning(
                f"Charge neutrality check: n_e={self.n_e:.2e}, "
                f"total_species_density={total_species_density:.2e}"
            )

        return True


class TwoRegionPlasma(SingleZoneLTEPlasma):
    """
    Two-region (core + corona) plasma model.

    Based on Hermann (2017), modeling the plasma as a hot core
    and a cooler corona. This helps reduce bias for high-Z elements
    emitting from outer shells.
    """

    def __init__(
        self,
        T_core: float,
        T_corona: float,
        n_e: float,
        species: dict[str, float],
        T_g: float | None = None,
        pressure: float | None = None,
    ):
        """
        Initialize two-region plasma.

        Parameters
        ----------
        T_core : float
            Core temperature in K
        T_corona : float
            Corona temperature in K
        n_e : float
            Electron density in cm^-3
        species : dict[str, float]
            Element number densities in cm^-3
        T_g : float, optional
            Gas temperature in K
        pressure : float, optional
            Pressure in atm
        """
        super().__init__(T_core, n_e, species, T_g, pressure)
        self.T_core = T_core
        self.T_corona = T_corona
        # Gate logging on tracer detection to avoid ConcretizationTypeError
        # when called inside JAX jit/vmap traces.
        if not isinstance(T_core, _JAX_TRACER):
            logger.info(
                f"Created TwoRegionPlasma: T_core={T_core:.1f} K, T_corona={T_corona:.1f} K, "
                f"n_e={n_e:.2e} cm^-3"
            )

    def validate(self) -> bool:
        """Validate two-region plasma state."""
        super().validate()
        if self.T_corona <= 0:
            raise ValueError("Corona temperature must be positive")
        if self.T_corona > self.T_core:
            logger.warning(
                f"Unphysical temperature gradient: T_corona ({self.T_corona:.1f} K) "
                f"> T_core ({self.T_core:.1f} K)"
            )
        return True


class SingleZoneLTEPlasmaJax(SingleZoneLTEPlasma):
    """JAX-friendly wrapper around :class:`SingleZoneLTEPlasma`.

    This subclass exposes ``T_e``, ``n_e`` and the per-element densities as
    ``jnp.ndarray`` views in addition to the existing Python-float
    attributes. The Python-float surface is preserved verbatim so existing
    consumers (validation, charge-neutrality check, ``solve_*`` methods)
    keep working without modification — that means a ``SingleZoneLTEPlasmaJax``
    is interchangeable with a plain ``SingleZoneLTEPlasma`` everywhere it
    appears in the forward model.

    Use this class when you want to feed the plasma's tensors into a
    JIT-compiled forward kernel (e.g. ``SpectrumModelJax``) without
    re-converting on every call.
    """

    def __init__(
        self,
        T_e: float,
        n_e: float,
        species: dict[str, float],
        T_g: float | None = None,
        pressure: float | None = None,
    ):
        super().__init__(T_e=T_e, n_e=n_e, species=species, T_g=T_g, pressure=pressure)
        if HAS_JAX:
            # Stable iteration order — Python 3.7+ dicts preserve insertion
            # order, which is what we want for the (element -> density) view.
            self._species_keys: tuple[str, ...] = tuple(species.keys())
            self._T_e_jax = jnp.asarray(float(T_e))
            self._n_e_jax = jnp.asarray(float(n_e))
            self._species_densities_jax = jnp.asarray(
                [float(species[k]) for k in self._species_keys]
            )
        else:  # pragma: no cover - JAX should be installed in this repo
            self._species_keys = tuple(species.keys())
            self._T_e_jax = None
            self._n_e_jax = None
            self._species_densities_jax = None

    @property
    def T_e_jax(self) -> "jnp.ndarray":
        """Electron temperature in K as a 0-d ``jnp.ndarray``."""
        if not HAS_JAX:  # pragma: no cover
            raise ImportError("JAX is required for T_e_jax")
        return self._T_e_jax

    @property
    def T_e_eV_jax(self) -> "jnp.ndarray":
        """Electron temperature in eV as a 0-d ``jnp.ndarray``."""
        if not HAS_JAX:  # pragma: no cover
            raise ImportError("JAX is required for T_e_eV_jax")
        return self._T_e_jax * KB_EV

    @property
    def n_e_jax(self) -> "jnp.ndarray":
        """Electron density in cm^-3 as a 0-d ``jnp.ndarray``."""
        if not HAS_JAX:  # pragma: no cover
            raise ImportError("JAX is required for n_e_jax")
        return self._n_e_jax

    @property
    def species_keys(self) -> tuple[str, ...]:
        """Stable element-symbol ordering matching ``species_densities_jax``."""
        return self._species_keys

    @property
    def species_densities_jax(self) -> "jnp.ndarray":
        """1-D ``jnp.ndarray`` of element number densities, ordered by ``species_keys``."""
        if not HAS_JAX:  # pragma: no cover
            raise ImportError("JAX is required for species_densities_jax")
        return self._species_densities_jax

    @classmethod
    def from_plasma(cls, plasma: SingleZoneLTEPlasma) -> "SingleZoneLTEPlasmaJax":
        """Construct a JAX plasma view from an existing NumPy plasma."""
        return cls(
            T_e=plasma.T_e,
            n_e=plasma.n_e,
            species=dict(plasma.species),
            T_g=plasma.T_g,
            pressure=plasma.pressure,
        )


# Wire JAX pytree registration so consumers can `jax.vmap(forward, in_axes=(0, ...))`
# over a batched SingleZoneLTEPlasma without explicit setup. Idempotent.
from cflibs.core.jax_runtime import _ensure_pytrees_registered as _register_pytrees  # noqa: E402

_register_pytrees()
