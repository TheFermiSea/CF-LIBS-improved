"""
Factory for creating hardware components.
"""

from typing import Any, Dict, Optional, Type, cast
from pathlib import Path

from cflibs.hardware.abc import HardwareComponent
from cflibs.hardware.spectrograph import SpectrographHardware
from cflibs.hardware.laser import LaserHardware
from cflibs.hardware.stages import MotionStageHardware
from cflibs.hardware.flow import FlowRegulatorHardware
from cflibs.core.logging_config import get_logger

logger = get_logger("hardware.factory")


class HardwareFactory:
    """Factory for creating hardware component instances."""

    _components: Dict[str, Dict[str, Type[HardwareComponent]]] = {
        "spectrograph": {},
        "laser": {},
        "motion_stage": {},
        "flow_regulator": {},
    }

    @classmethod
    def register(
        cls, component_type: str, name: str, component_class: Type[HardwareComponent]
    ) -> None:
        """
        Register a hardware component class.

        Parameters
        ----------
        component_type : str
            Component type ('spectrograph', 'laser', 'motion_stage', 'flow_regulator')
        name : str
            Component name/identifier
        component_class : Type[HardwareComponent]
            Component class
        """
        if component_type not in cls._components:
            raise ValueError(
                f"Unknown component type: {component_type}. "
                f"Available: {list(cls._components.keys())}"
            )

        cls._components[component_type][name] = component_class
        logger.debug(f"Registered {component_type}: {name}")

    @classmethod
    def create(
        cls, component_type: str, name: str, config: Optional[Dict[str, Any]] = None, **kwargs
    ) -> HardwareComponent:
        """
        Create a hardware component instance.

        Parameters
        ----------
        component_type : str
            Component type
        name : str
            Component name/identifier
        config : dict, optional
            Configuration dictionary
        **kwargs
            Additional arguments for component constructor

        Returns
        -------
        HardwareComponent
            Component instance

        Raises
        ------
        ValueError
            If component type or name is not registered
        """
        if component_type not in cls._components:
            available = ", ".join(cls._components.keys())
            raise ValueError(f"Unknown component type: {component_type}. Available: {available}")

        components = cls._components[component_type]

        # Use name if registered, otherwise use default placeholder
        if name in components:
            component_class = components[name]
        else:
            # Use default placeholder based on type
            defaults = {
                "spectrograph": SpectrographHardware,
                "laser": LaserHardware,
                "motion_stage": MotionStageHardware,
                "flow_regulator": FlowRegulatorHardware,
            }
            default_class = defaults.get(component_type)
            if default_class is None:
                raise ValueError(f"No default implementation for {component_type}")
            # The fallback classes are concrete subclasses of HardwareComponent;
            # the dict's annotation flattens them to type[HardwareComponent], which
            # mypy then rejects as abstract. Cast through Any for the assignment.
            component_class = cast(Any, default_class)
            logger.warning(
                f"Component '{name}' not registered for type '{component_type}', "
                f"using default placeholder"
            )

        return component_class(name=name, config=config, **kwargs)

    @classmethod
    def create_from_config(cls, config_path: Path) -> Dict[str, HardwareComponent]:
        """
        Create hardware components from configuration file.

        Parameters
        ----------
        config_path : Path
            Path to configuration file (YAML or JSON)

        Returns
        -------
        dict
            Dictionary mapping component names to instances
        """
        from cflibs.core.config import load_config

        config = load_config(config_path)

        if "hardware" not in config:
            raise ValueError("Configuration must contain 'hardware' section")

        hardware_config = config["hardware"]
        components = {}

        for component_type in ["spectrograph", "laser", "motion_stage", "flow_regulator"]:
            if component_type in hardware_config:
                comp_config = hardware_config[component_type]
                if isinstance(comp_config, dict):
                    # Single component
                    name = comp_config.get("name", component_type)
                    comp = cls.create(component_type, name, comp_config)
                    components[name] = comp
                elif isinstance(comp_config, list):
                    # Multiple components of same type
                    for comp_config_item in comp_config:
                        name = comp_config_item.get("name", f"{component_type}_{len(components)}")
                        comp = cls.create(component_type, name, comp_config_item)
                        components[name] = comp

        return components

    @classmethod
    def list_components(cls, component_type: Optional[str] = None) -> Dict[str, list]:
        """
        List registered components.

        Parameters
        ----------
        component_type : str, optional
            Filter by component type

        Returns
        -------
        dict
            Dictionary mapping component types to lists of registered names
        """
        if component_type:
            if component_type not in cls._components:
                return {}
            return {component_type: list(cls._components[component_type].keys())}

        return {
            comp_type: list(components.keys()) for comp_type, components in cls._components.items()
        }


# Register default implementations
HardwareFactory.register("spectrograph", "default", SpectrographHardware)
HardwareFactory.register("laser", "default", LaserHardware)
HardwareFactory.register("motion_stage", "default", MotionStageHardware)
HardwareFactory.register("flow_regulator", "default", FlowRegulatorHardware)
