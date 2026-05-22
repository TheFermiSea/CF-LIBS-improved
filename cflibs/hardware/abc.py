"""
Abstract base classes for hardware interfaces.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, List
from enum import Enum
import numpy as np

__all__ = [
    "FlowRegulatorInterface",
    "HardwareComponent",
    "HardwareStatus",
    "LaserInterface",
    "MotionStageInterface",
    "SpectrographInterface",
]


class HardwareStatus(Enum):
    """Hardware component status."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    READY = "ready"
    ERROR = "error"
    BUSY = "busy"


class HardwareComponent(ABC):
    """
    Abstract base class for all hardware components.

    All hardware components must implement this interface to ensure
    consistent behavior and integration with control systems.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize hardware component.

        Parameters
        ----------
        name : str
            Component name/identifier
        config : dict, optional
            Configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self._status = HardwareStatus.DISCONNECTED
        self._error_message: Optional[str] = None

    @property
    def status(self) -> HardwareStatus:
        """Current hardware status."""
        return self._status

    @property
    def is_connected(self) -> bool:
        """Check if hardware is connected."""
        return self._status in [HardwareStatus.CONNECTED, HardwareStatus.READY, HardwareStatus.BUSY]

    @property
    def is_ready(self) -> bool:
        """Check if hardware is ready for operation."""
        return self._status == HardwareStatus.READY

    @property
    def error_message(self) -> Optional[str]:
        """Get error message if status is ERROR."""
        return self._error_message

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to hardware.

        Returns
        -------
        bool
            True if connection successful
        """
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from hardware.

        Returns
        -------
        bool
            True if disconnection successful
        """
        pass

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize hardware (after connection).

        Returns
        -------
        bool
            True if initialization successful
        """
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get detailed status information.

        Returns
        -------
        dict
            Status dictionary with component-specific information
        """
        pass

    def reset(self) -> bool:
        """
        Reset hardware to default state.

        Returns
        -------
        bool
            True if reset successful
        """
        if not self.is_connected:
            return False
        return self.initialize()

    def __enter__(self):
        """Context manager entry."""
        if self.connect():
            self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


class SpectrographInterface(HardwareComponent):
    """
    Abstract interface for spectrograph/detector hardware.

    This interface handles:
    - Detector control (exposure, gain, binning)
    - Wavelength calibration
    - Data acquisition
    - Triggering
    """

    @abstractmethod
    def set_exposure_time(self, time_ms: float) -> bool:
        """
        Set exposure time.

        Parameters
        ----------
        time_ms : float
            Exposure time in milliseconds

        Returns
        -------
        bool
            True if successful
        """
        pass

    @abstractmethod
    def get_exposure_time(self) -> float:
        """
        Get current exposure time.

        Returns
        -------
        float
            Exposure time in milliseconds
        """
        pass

    @abstractmethod
    def set_gain(self, gain: float) -> bool:
        """
        Set detector gain.

        Parameters
        ----------
        gain : float
            Gain value

        Returns
        -------
        bool
            True if successful
        """
        pass

    @abstractmethod
    def acquire_spectrum(self, trigger: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Acquire a spectrum.

        Parameters
        ----------
        trigger : bool
            Whether to trigger acquisition

        Returns
        -------
        wavelength : array
            Wavelength array in nm
        intensity : array
            Intensity array (counts or calibrated units)
        """
        pass

    @abstractmethod
    def calibrate_wavelength(self, reference_lines: Dict[float, float]) -> bool:
        """
        Calibrate wavelength axis using reference lines.

        Parameters
        ----------
        reference_lines : dict
            Dictionary mapping pixel to wavelength (nm)

        Returns
        -------
        bool
            True if calibration successful
        """
        pass

    @abstractmethod
    def get_detector_info(self) -> Dict[str, Any]:
        """
        Get detector information.

        Returns
        -------
        dict
            Detector information (resolution, size, etc.)
        """
        pass


class LaserInterface(HardwareComponent):
    """
    Abstract interface for laser hardware.

    This interface handles:
    - Laser power/energy control
    - Pulse parameters (duration, repetition rate)
    - Triggering
    - Safety interlocks
    """

    @abstractmethod
    def set_power(self, power_mW: float) -> bool:
        """
        Set laser power.

        Parameters
        ----------
        power_mW : float
            Power in milliwatts

        Returns
        -------
        bool
            True if successful
        """
        pass

    @abstractmethod
    def get_power(self) -> float:
        """
        Get current laser power.

        Returns
        -------
        float
            Power in milliwatts
        """
        pass

    @abstractmethod
    def set_pulse_energy(self, energy_mJ: float) -> bool:
        """
        Set pulse energy.

        Parameters
        ----------
        energy_mJ : float
            Pulse energy in millijoules

        Returns
        -------
        bool
            True if successful
        """
        pass

    @abstractmethod
    def set_repetition_rate(self, rate_Hz: float) -> bool:
        """
        Set repetition rate.

        Parameters
        ----------
        rate_Hz : float
            Repetition rate in Hz

        Returns
        -------
        bool
            True if successful
        """
        pass

    @abstractmethod
    def fire(self, n_pulses: int = 1) -> bool:
        """
        Fire laser pulses.

        Parameters
        ----------
        n_pulses : int
            Number of pulses to fire

        Returns
        -------
        bool
            True if successful
        """
        pass

    @abstractmethod
    def enable(self) -> bool:
        """
        Enable laser (safety interlock).

        Returns
        -------
        bool
            True if enabled
        """
        pass

    @abstractmethod
    def disable(self) -> bool:
        """
        Disable laser (safety interlock).

        Returns
        -------
        bool
            True if disabled
        """
        pass

    @abstractmethod
    def get_laser_info(self) -> Dict[str, Any]:
        """
        Get laser information.

        Returns
        -------
        dict
            Laser information (wavelength, max power, etc.)
        """
        pass


class MotionStageInterface(HardwareComponent):
    """
    Abstract interface for motion stage hardware.

    This interface handles:
    - Multi-axis positioning
    - Homing/referencing
    - Velocity control
    - Limit switches
    """

    @abstractmethod
    def get_position(self, axis: str) -> float:
        """
        Get current position of an axis.

        Parameters
        ----------
        axis : str
            Axis name (e.g., 'X', 'Y', 'Z')

        Returns
        -------
        float
            Position in mm
        """
        pass

    @abstractmethod
    def move_to(self, axis: str, position_mm: float, wait: bool = True) -> bool:
        """
        Move axis to position.

        Parameters
        ----------
        axis : str
            Axis name
        position_mm : float
            Target position in mm
        wait : bool
            Whether to wait for movement to complete

        Returns
        -------
        bool
            True if movement successful
        """
        pass

    @abstractmethod
    def move_relative(self, axis: str, distance_mm: float, wait: bool = True) -> bool:
        """
        Move axis relative to current position.

        Parameters
        ----------
        axis : str
            Axis name
        distance_mm : float
            Distance to move in mm
        wait : bool
            Whether to wait for movement to complete

        Returns
        -------
        bool
            True if movement successful
        """
        pass

    @abstractmethod
    def home(self, axis: Optional[str] = None) -> bool:
        """
        Home/reference axis or all axes.

        Parameters
        ----------
        axis : str, optional
            Specific axis to home, or None for all axes

        Returns
        -------
        bool
            True if homing successful
        """
        pass

    @abstractmethod
    def set_velocity(self, axis: str, velocity_mm_s: float) -> bool:
        """
        Set axis velocity.

        Parameters
        ----------
        axis : str
            Axis name
        velocity_mm_s : float
            Velocity in mm/s

        Returns
        -------
        bool
            True if successful
        """
        pass

    @abstractmethod
    def get_available_axes(self) -> List[str]:
        """
        Get list of available axes.

        Returns
        -------
        list
            List of axis names
        """
        pass


class FlowRegulatorInterface(HardwareComponent):
    """
    Abstract interface for powder flow regulator hardware.

    This interface handles:
    - Flow rate control
    - Flow monitoring
    - Valve control
    - Powder level sensing
    """

    @abstractmethod
    def set_flow_rate(self, rate_g_s: float) -> bool:
        """
        Set powder flow rate.

        Parameters
        ----------
        rate_g_s : float
            Flow rate in grams per second

        Returns
        -------
        bool
            True if successful
        """
        pass

    @abstractmethod
    def get_flow_rate(self) -> float:
        """
        Get current flow rate.

        Returns
        -------
        float
            Flow rate in grams per second
        """
        pass

    @abstractmethod
    def open_valve(self) -> bool:
        """
        Open flow valve.

        Returns
        -------
        bool
            True if successful
        """
        pass

    @abstractmethod
    def close_valve(self) -> bool:
        """
        Close flow valve.

        Returns
        -------
        bool
            True if successful
        """
        pass

    @abstractmethod
    def get_powder_level(self) -> float:
        """
        Get powder level in hopper.

        Returns
        -------
        float
            Powder level (0-100% or grams)
        """
        pass

    @abstractmethod
    def start_flow(self) -> bool:
        """
        Start powder flow.

        Returns
        -------
        bool
            True if successful
        """
        pass

    @abstractmethod
    def stop_flow(self) -> bool:
        """
        Stop powder flow.

        Returns
        -------
        bool
            True if successful
        """
        pass
