# Testing Documentation

## Overview
This document tracks the testing progress, issues encountered, and future improvements for the energy fault prediction system.

## Current Test Status

### Completed
- Configuration integration tests for refactored components

### In Progress
- Testing all components after refactoring

### Pending
- Integration tests between components
- Performance tests
- Edge case handling tests

## Issues and Fixes

### 2025-08-27
- **Issue**: `TypeError: generate_output() missing 1 required positional argument: 'power_request_series'` in `tests/unit/battery/test_battery_system.py` and `tests/unit/diesel_generator/test_diesel_generator_system.py`.
- **Fix**: Modified `generate_output` in `src/battery_system.py` and `src/diesel_generator.py` to handle both single numeric values and pandas Series for the `power_request_series` argument. This was done by checking the type of the input and converting single values to a Series. The methods now also return a dictionary of single values if the input was a single value.
- **Details**: The tests were calling `generate_output` with a single numeric value, while the method expected a pandas Series. The fix makes the method more flexible and allows the tests to pass without modification.
- **Issue**: `AssertionError` in `test_charging_state_update` and `test_discharging_state_update` in `tests/unit/battery/test_battery_system.py`.
- **Fix**: Inverted the charging/discharging logic in `update_state` and `generate_output` methods in `src/battery_system.py`.
- **Details**: The convention in the tests is that negative power means charging and positive power means discharging. The implementation had the opposite logic.

### 2023-10-12
- **Issue**: Import error in test_config_integration.py - `FaultInjector` class not found
- **Fix**: Updated import to use `FaultInjectionSystem` which is the actual class name in fault_injection.py
- **Details**: The test was trying to import a non-existent class. Also removed assertion for `thresholds` attribute which is not directly stored on the class instance.

### 2025-08-22
- **Issue**: `AttributeError: 'WeatherSimulator' object has no attribute 'temperature_peak_hour'` in `tests/test_config_integration.py`.
- **Fix**: Added initialization of `temperature_peak_hour` in the `WeatherSimulator`'s `__init__` method in `src/weather.py`.
- **Details**: The test expected `temperature_peak_hour` to be an instance attribute of the `WeatherSimulator` object, but it was only being used in the `generate_weather` method without being initialized in the constructor.
- **Issue**: `AttributeError: 'DataValidator' object has no attribute 'power_balance_tolerance'` in `tests/test_config_integration.py`.
- **Fix**: Initialized `power_balance_tolerance` in the `DataValidator`'s `__init__` method in `src/validation.py`.
- **Details**: The test expected `power_balance_tolerance` to be an attribute of the `DataValidator` object, but it was not being initialized in the constructor.

## Future Improvements

### Test Coverage
- Add unit tests for each component method
- Add more comprehensive integration tests
- Implement automated test coverage reporting

### Test Infrastructure
- Set up continuous integration for automated testing
- Create test fixtures for common test scenarios
- Implement property-based testing for edge cases

### Documentation
- Document test procedures for new components
- Create test data generation utilities
- Document expected behavior for fault scenarios