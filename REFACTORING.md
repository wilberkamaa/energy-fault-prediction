# Energy Fault Prediction System Refactoring

## Overview
This document tracks the refactoring process of centralizing all hardcoded parameters into a configuration system.

## Goals
- Centralize all hardcoded parameters into `src/config.py`
- Make the simulation more configurable
- Improve code maintainability
- Add grid power inclusion/exclusion capability

## Progress

### Completed
- ✅ Created centralized `config.py` with all system parameters

### In Progress
- 🔄 Updating component files to use centralized configuration

### Pending
- Battery System (`battery_system.py`)
- Diesel Generator (`diesel_generator.py`)
- Fault Injection (`fault_injection.py`)
- Load Profile (`load_profile.py`)
- Solar PV (`solar_pv.py`)
- Validation (`validation.py`)
- Weather (`weather.py`)

### Not Included
- `grid_connection.py` (scheduled for removal)
- `data_generator.py` (updates postponed)

## Implementation Details

### Configuration Structure
The new configuration system is organized by component:
- Battery parameters
- Diesel generator parameters
- Fault injection parameters
- Grid connection parameters
- Load profile parameters
- Solar PV parameters
- Validation parameters
- Weather parameters

### Usage
To use the configuration in component files:
```python
from src.config import config

# Access component-specific config
component_config = config['component_name']
```

## Testing Strategy
- Each component will be tested after updating to use the new configuration
- System-wide integration testing will be performed after all components are updated

## Notes
- `grid_connection.py` will be removed in a future update
- Grid power can be disabled using `include_grid` flag in configuration