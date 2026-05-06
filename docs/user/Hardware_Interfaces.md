# Hardware Interfaces Documentation

## Overview

CF-LIBS provides a comprehensive hardware interface system for integrating with physical instrumentation. The system is designed to be modular and extensible, allowing easy integration with custom GUI applications and hardware control systems.

## Architecture

The hardware interface system follows these design principles:

1. **Abstract Base Classes**: All hardware components implement standardized interfaces
2. **Placeholder Implementations**: Default implementations for development and testing
3. **Factory Pattern**: Centralized component creation and registration
4. **Manager Pattern**: Coordinated management of multiple components
5. **Configuration-Driven**: Hardware setup via YAML/JSON configuration files

## Component Types

### 1. Spectrograph/Detector

**Interface**: `SpectrographInterface`

**Capabilities**:
- Exposure time control
- Gain adjustment
- Spectrum acquisition
- Wavelength calibration
- Detector information queries

**Example**:
```python
from cflibs.hardware import SpectrographHardware

# Create spectrograph
spec = SpectrographHardware(
    name="main_spectrograph",
    config={
        'model': 'Andor iDus',
        'resolution_nm': 0.1,
        'wavelength_range': (200.0, 800.0),
        'pixels': 2048
    }
)

# Connect and initialize
with spec:
    # Set exposure time
    spec.set_exposure_time(100.0)  # ms
    
    # Acquire spectrum
    wavelength, intensity = spec.acquire_spectrum()
```

### 2. Laser

**Interface**: `LaserInterface`

**Capabilities**:
- Power/energy control
- Repetition rate setting
- Pulse firing
- Safety interlock control
- Laser information queries

**Example**:
```python
from cflibs.hardware import LaserHardware

# Create laser
laser = LaserHardware(
    name="main_laser",
    config={
        'model': 'Nd:YAG',
        'wavelength_nm': 1064.0,
        'max_power_mW': 1000.0,
        'max_energy_mJ': 100.0
    }
)

# Connect and control
with laser:
    laser.enable()
    laser.set_pulse_energy(50.0)  # mJ
    laser.set_repetition_rate(10.0)  # Hz
    laser.fire(n_pulses=1)
```

### 3. Motion Stages

**Interface**: `MotionStageInterface`

**Capabilities**:
- Multi-axis positioning
- Relative movements
- Homing/referencing
- Velocity control
- Position queries

**Example**:
```python
from cflibs.hardware import MotionStageHardware

# Create motion stage
stage = MotionStageHardware(
    name="xyz_stage",
    config={
        'axes': ['X', 'Y', 'Z'],
        'travel_range': {
            'X': (0.0, 100.0),
            'Y': (0.0, 100.0),
            'Z': (0.0, 50.0)
        }
    }
)

# Connect and move
with stage:
    # Home all axes
    stage.home()
    
    # Move to position
    stage.move_to('X', 50.0, wait=True)
    
    # Relative movement
    stage.move_relative('Y', 10.0)
    
    # Get position
    pos = stage.get_position('X')
```

### 4. Powder Flow Regulator

**Interface**: `FlowRegulatorInterface`

**Capabilities**:
- Flow rate control
- Valve control
- Flow start/stop
- Powder level monitoring

**Example**:
```python
from cflibs.hardware import FlowRegulatorHardware

# Create flow regulator
flow = FlowRegulatorHardware(
    name="powder_feeder",
    config={
        'model': 'Vibra Screw',
        'max_flow_rate_g_s': 10.0,
        'min_flow_rate_g_s': 0.01
    }
)

# Connect and control
with flow:
    # Set flow rate
    flow.set_flow_rate(5.0)  # g/s
    
    # Open valve and start flow
    flow.open_valve()
    flow.start_flow()
    
    # Monitor powder level
    level = flow.get_powder_level()
    
    # Stop flow
    flow.stop_flow()
    flow.close_valve()
```

## Hardware Manager

The `HardwareManager` class provides centralized management of all hardware components:

```python
from cflibs.hardware import HardwareManager
from pathlib import Path

# Create manager from config file
manager = HardwareManager(config_path=Path("hardware_config.yaml"))

# Connect all components
with manager:
    # Get status of all components
    status = manager.get_all_status()
    
    # Get ready components
    ready = manager.get_ready_components()
    
    # Access specific components
    laser = manager.get_component("main_laser")
    spec = manager.get_component("main_spectrograph")
    
    # Coordinated operation
    laser.fire()
    wavelength, intensity = spec.acquire_spectrum()
```

## Factory Pattern

Use `HardwareFactory` to create components programmatically:

```python
from cflibs.hardware import HardwareFactory

# Create component
laser = HardwareFactory.create(
    component_type='laser',
    name='main_laser',
    config={
        'wavelength_nm': 1064.0,
        'max_power_mW': 1000.0
    }
)

# Register custom implementation
HardwareFactory.register('laser', 'custom_model', CustomLaserClass)

# List available components
components = HardwareFactory.list_components('laser')
```

## Configuration Files

Hardware can be configured via YAML or JSON files:

```yaml
hardware:
  spectrograph:
    name: "main_spectrograph"
    model: "Andor iDus"
    resolution_nm: 0.1
    wavelength_range: [200.0, 800.0]
    pixels: 2048
  
  laser:
    name: "main_laser"
    model: "Nd:YAG"
    wavelength_nm: 1064.0
    max_power_mW: 1000.0
```

Load configuration:
```python
from cflibs.hardware import HardwareManager
from pathlib import Path

manager = HardwareManager(config_path=Path("hardware_config.yaml"))
```

## Custom Implementations

To integrate custom hardware drivers:

1. **Create Implementation Class**:
```python
from cflibs.hardware.abc import LaserInterface

class CustomLaser(LaserInterface):
    def connect(self):
        # Custom connection logic
        pass
    
    def set_power(self, power_mW):
        # Custom power control
        pass
    
    # ... implement all required methods
```

2. **Register with Factory**:
```python
from cflibs.hardware import HardwareFactory

HardwareFactory.register('laser', 'custom_model', CustomLaser)
```

3. **Use in Configuration**:
```yaml
hardware:
  laser:
    name: "custom_laser"
    model: "custom_model"
    # ... custom config
```

## Status Monitoring

All components provide status information:

```python
# Get component status
status = component.get_status()
print(status)
# {
#     'name': 'main_laser',
#     'status': 'ready',
#     'power_mW': 500.0,
#     'enabled': True
# }

# Check connection status
if component.is_connected:
    print("Component is connected")

if component.is_ready:
    print("Component is ready for operation")
```

## Error Handling

Components provide error information:

```python
if component.status == HardwareStatus.ERROR:
    error_msg = component.error_message
    print(f"Error: {error_msg}")
```

## Context Managers

All components support context managers for automatic cleanup:

```python
with component:
    # Component automatically connects and initializes
    component.do_operation()
# Component automatically disconnects
```

## Integration with GUI

The hardware interfaces are designed for easy GUI integration:

1. **Status Updates**: Components provide status dictionaries that can be displayed
2. **Event Callbacks**: (Future) Components can emit events for GUI updates
3. **Thread Safety**: Components can be accessed from GUI threads
4. **Configuration**: GUI can load/save hardware configurations

## Placeholder Implementations

Current implementations are placeholders that:
- Log all operations
- Return placeholder data (zeros, default values)
- Do not connect to actual hardware
- Are suitable for development and testing

Replace with actual hardware drivers when integrating with physical systems.

## Future Enhancements

Planned features:
- [ ] Event/callback system for status updates
- [ ] Thread-safe operation queues
- [ ] Hardware simulation mode
- [ ] Calibration data persistence
- [ ] Multi-device synchronization
- [ ] Hardware health monitoring
- [ ] Automatic reconnection on failure

## Examples

See `examples/hardware_config_example.yaml` for a complete configuration example.

## API Reference

See module docstrings for detailed API documentation:
- `cflibs.hardware.abc` - Abstract base classes
- `cflibs.hardware.spectrograph` - Spectrograph implementation
- `cflibs.hardware.laser` - Laser implementation
- `cflibs.hardware.stages` - Motion stage implementation
- `cflibs.hardware.flow` - Flow regulator implementation
- `cflibs.hardware.factory` - Factory pattern
- `cflibs.hardware.manager` - Hardware manager

