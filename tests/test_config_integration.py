import pytest
import numpy as np
from src.config import config
from src.battery_system import BatterySystemSimulator
from src.diesel_generator import DieselGeneratorSimulator
from src.fault_injection import FaultInjectionSystem
from src.load_profile import LoadProfileGenerator
from src.solar_pv import SolarPVSimulator
from src.validation import DataValidator
from src.weather import WeatherSimulator

# Test Battery System Configuration Integration
class TestBatterySystem:
    def test_battery_initialization(self):
        """Test that battery system correctly loads configuration values."""
        battery = BatterySystemSimulator(seed=42)
        
        # Verify configuration values were correctly loaded
        assert battery.capacity_kwh == config['battery']['capacity_kwh']
        assert battery.max_power_kw == config['battery']['max_power_kw']
        assert battery.min_soc == config['battery']['min_soc']
        assert battery.max_soc == config['battery']['max_soc']
        assert battery.charging_efficiency == config['battery']['charging_efficiency']
        assert battery.discharging_efficiency == config['battery']['discharging_efficiency']
    
    def test_battery_power_limits(self):
        """Test that power limits respect configuration constraints."""
        battery = BatterySystemSimulator(seed=42)
        limits = battery.calculate_power_limits()
        
        # Verify power limits don't exceed configured max power
        assert limits['charge_limit'] <= config['battery']['max_power_kw']
        assert limits['discharge_limit'] <= config['battery']['max_power_kw']

# Test Diesel Generator Configuration Integration
class TestDieselGenerator:
    def test_generator_initialization(self):
        """Test that generator correctly loads configuration values."""
        generator = DieselGeneratorSimulator(seed=42)
        
        # Verify configuration values were correctly loaded
        assert generator.capacity_kva == config['diesel_generator']['capacity_kva']
        assert generator.fuel_tank_capacity == config['diesel_generator']['fuel_tank_capacity']
        assert generator.min_load_percent == config['diesel_generator']['min_load_percent']

# Test Fault Injection Configuration Integration
class TestFaultInjection:
    def test_fault_injector_initialization(self):
        """Test that fault injector correctly loads configuration values."""
        fault_injector = FaultInjectionSystem(seed=42)
        
        # Verify configuration values were correctly loaded
        assert fault_injector.fault_probabilities == config['fault_injection']['fault_probabilities']
        assert fault_injector.fault_durations == config['fault_injection']['fault_durations']
        #assert fault_injector.thresholds == config['fault_injection']['thresholds']

# Test Load Profile Configuration Integration
class TestLoadProfile:
    def test_load_profile_initialization(self):
        """Test that load profile generator correctly loads configuration values."""
        load_profile = LoadProfileGenerator(seed=42)
        
        # Verify configuration values were correctly loaded
        assert load_profile.base_load_kw == config['load_profile']['base_load_kw']
        assert load_profile.peak_load_kw == config['load_profile']['peak_load_kw']
        assert load_profile.weekday_factors == config['load_profile']['weekday_factors']
        assert load_profile.weekend_reduction == config['load_profile']['weekend_reduction']
        assert load_profile.holidays == config['load_profile']['holidays']

# Test Solar PV Configuration Integration
class TestSolarPV:
    def test_solar_pv_initialization(self):
        """Test that solar PV simulator correctly loads configuration values."""
        solar_pv = SolarPVSimulator(seed=42)
        
        # Verify configuration values were correctly loaded
        assert solar_pv.capacity_kw == config['solar_pv']['capacity_kw']
        assert solar_pv.nominal_efficiency == config['solar_pv']['nominal_efficiency']
        assert solar_pv.temp_coefficient == config['solar_pv']['temp_coefficient']
        assert solar_pv.dust_loss_rate == config['solar_pv']['dust_loss_rate']

# Test Validation Configuration Integration
class TestValidation:
    def test_validator_initialization(self):
        """Test that data validator correctly loads configuration values."""
        validator = DataValidator()
        
        # Verify configuration values were correctly loaded
        assert validator.valid_ranges == config['validation']['valid_ranges']
        assert validator.power_balance_tolerance == config['validation']['power_balance_tolerance']
        assert validator.nan_fill_value == config['validation']['nan_fill_value']

# Test Weather Configuration Integration
class TestWeather:
    def test_weather_initialization(self):
        """Test that weather simulator correctly loads configuration values."""
        weather = WeatherSimulator(seed=42)
        
        # Verify configuration values were correctly loaded
        assert weather.season_params == config['weather']['season_params']
        assert weather.base_temperature == config['weather']['base_temperature']
        assert weather.temperature_amplitude == config['weather']['temperature_amplitude']
        assert weather.temperature_peak_hour == config['weather']['temperature_peak_hour']

# Test System-wide Integration
class TestSystemIntegration:
    def test_config_modification_propagation(self):
        """Test that changes to config are reflected in components."""
        # Save original value
        original_capacity = config['battery']['capacity_kwh']
        
        try:
            # Modify config
            config['battery']['capacity_kwh'] = 200
            
            # Create new instance and verify it uses updated config
            battery = BatterySystemSimulator(seed=42)
            assert battery.capacity_kwh == 200
        finally:
            # Restore original value
            config['battery']['capacity_kwh'] = original_capacity