# Energy Fault Prediction System Refactoring

## Overview
This document tracks the refactoring process of centralizing all hardcoded parameters into a configuration system and removing grid connection functionality.

## Goals
- Centralize all hardcoded parameters into `src/config.py`
- Make the simulation more configurable
- Improve code maintainability
- Remove grid power functionality to transform from four-component to three-component architecture

## Progress

### Completed
- ✅ Created centralized `config.py` with all system parameters
- ✅ Updated battery_system.py to use config
- ✅ Updated diesel_generator.py to use config
- ✅ Updated fault_injection.py to use config
- ✅ Updated load_profile.py to use config
- ✅ Updated solar_pv.py to use config
- ✅ Updated validation.py to use config
- ✅ Updated weather.py to use config
- ✅ Removed grid-related fault types from fault_injection.py
- ✅ Updated test_fault_injection_system.py to work with removed grid fault types
- ✅ Modified validation.py to exclude grid power from calculations

### In Progress
- 🔄 Testing all components after refactoring
- 🔄 Cleaning data_generator.py - removing commented grid code
- 🔄 Adding defensive programming for any remaining grid references

### Not Included
- `grid_connection.py` (scheduled for removal)

## Implementation Details

### Configuration Structure
The new configuration system is organized by component:
- Battery parameters
- Diesel generator parameters
- Fault injection parameters
- Grid connection parameters (to be removed)
- Load profile parameters
  - Base and peak loads
  - Weekday/weekend factors
  - Seasonal adjustments
  - Holiday dates
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
- Holidays are now configured as month-day tuples for better reusability across years
- Defensive programming has been added to handle cases where grid parameters are not available