"""
Factory patterns for creating plasma models, solvers, and instruments.
"""

from typing import Any, Dict, Type, Optional, cast
from pathlib import Path

from cflibs.core.abc import SolverStrategy, PlasmaModel, InstrumentModelInterface, AtomicDataSource
from cflibs.plasma.state import SingleZoneLTEPlasma
from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolver
from cflibs.instrument.model import InstrumentModel
from cflibs.core.logging_config import get_logger

logger = get_logger("core.factory")


class SolverFactory:
    """Factory for creating solver instances."""

    _solvers: Dict[str, Type[SolverStrategy]] = {}

    @classmethod
    def register(cls, name: str, solver_class: Type[SolverStrategy]) -> None:
        """
        Register a solver class.

        Parameters
        ----------
        name : str
            Solver name
        solver_class : Type[SolverStrategy]
            Solver class
        """
        cls._solvers[name] = solver_class
        logger.debug(f"Registered solver: {name}")

    @classmethod
    def create(cls, name: str, atomic_db: AtomicDataSource, **kwargs) -> SolverStrategy:
        """
        Create a solver instance.

        Parameters
        ----------
        name : str
            Solver name
        atomic_db : AtomicDataSource
            Atomic data source
        **kwargs
            Additional arguments for solver constructor

        Returns
        -------
        SolverStrategy
            Solver instance

        Raises
        ------
        ValueError
            If solver name is not registered
        """
        if name not in cls._solvers:
            available = ", ".join(cls._solvers.keys())
            raise ValueError(f"Unknown solver: {name}. Available: {available}")

        # Concrete solvers accept (atomic_db, **kwargs); SolverStrategy ABC
        # itself defines no __init__, so cast through Any to satisfy mypy.
        solver_class = cast(Any, cls._solvers[name])
        return solver_class(atomic_db, **kwargs)

    @classmethod
    def list_solvers(cls) -> list:
        """List available solver names."""
        return list(cls._solvers.keys())


class PlasmaModelFactory:
    """Factory for creating plasma model instances."""

    _models: Dict[str, Type[PlasmaModel]] = {}

    @classmethod
    def register(cls, name: str, model_class: Type[PlasmaModel]) -> None:
        """
        Register a plasma model class.

        Parameters
        ----------
        name : str
            Model name
        model_class : Type[PlasmaModel]
            Model class
        """
        cls._models[name] = model_class
        logger.debug(f"Registered plasma model: {name}")

    @classmethod
    def create(cls, name: str, **kwargs) -> PlasmaModel:
        """
        Create a plasma model instance.

        Parameters
        ----------
        name : str
            Model name
        **kwargs
            Arguments for model constructor

        Returns
        -------
        PlasmaModel
            Model instance

        Raises
        ------
        ValueError
            If model name is not registered
        """
        if name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise ValueError(f"Unknown plasma model: {name}. Available: {available}")

        model_class = cls._models[name]
        return model_class(**kwargs)

    @classmethod
    def list_models(cls) -> list:
        """List available model names."""
        return list(cls._models.keys())


class InstrumentFactory:
    """Factory for creating instrument model instances."""

    _instruments: Dict[str, Type[InstrumentModelInterface]] = {}

    @classmethod
    def register(cls, name: str, instrument_class: Type[InstrumentModelInterface]) -> None:
        """
        Register an instrument class.

        Parameters
        ----------
        name : str
            Instrument name
        instrument_class : Type[InstrumentModelInterface]
            Instrument class
        """
        cls._instruments[name] = instrument_class
        logger.debug(f"Registered instrument: {name}")

    @classmethod
    def create(
        cls, name: str, config_path: Optional[Path] = None, **kwargs
    ) -> InstrumentModelInterface:
        """
        Create an instrument instance.

        Parameters
        ----------
        name : str
            Instrument name
        config_path : Path, optional
            Path to configuration file
        **kwargs
            Additional arguments for instrument constructor

        Returns
        -------
        InstrumentModelInterface
            Instrument instance

        Raises
        ------
        ValueError
            If instrument name is not registered
        """
        if name not in cls._instruments:
            available = ", ".join(cls._instruments.keys())
            raise ValueError(f"Unknown instrument: {name}. Available: {available}")

        # InstrumentModelInterface is a structural Protocol; the concrete
        # InstrumentModel exposes a `from_file` classmethod that the Protocol
        # cannot declare. Cast through Any for the dynamic dispatch.
        instrument_class = cast(Any, cls._instruments[name])

        if config_path:
            return instrument_class.from_file(config_path)
        else:
            return instrument_class(**kwargs)

    @classmethod
    def list_instruments(cls) -> list:
        """List available instrument names."""
        return list(cls._instruments.keys())


# Register default implementations. Mypy does not always recognise that
# concrete classes structurally satisfy the Protocol; the runtime behaviour
# matches the @runtime_checkable contract on PlasmaModel / InstrumentModelProtocol.
SolverFactory.register("saha_boltzmann", SahaBoltzmannSolver)
PlasmaModelFactory.register("single_zone_lte", cast(Type[PlasmaModel], SingleZoneLTEPlasma))
InstrumentFactory.register("standard", cast(Type[InstrumentModelInterface], InstrumentModel))
