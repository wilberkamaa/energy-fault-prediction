import pytest
import numpy as np
from src.battery_system import BatterySystemSimulator
from src.config import config

class TestBatterySystemInitialization:
    """Test the initialization of the battery system."""
    
    def test_initialization_with_default_seed(self):
        """Test that battery system initializes correctly with default seed."""
        battery = BatterySystemSimulator()
        
        # Verify configuration values were correctly loaded
        assert battery.capacity_kwh == config['battery']['capacity_kwh']
        assert battery.max_power_kw == config['battery']['max_power_kw']
        assert battery.min_soc == config['battery']['min_soc']
        assert battery.max_soc == config['battery']['max_soc']
        assert battery.charging_efficiency == config['battery']['charging_efficiency']
        assert battery.discharging_efficiency == config['battery']['discharging_efficiency']
        assert battery.self_discharge_rate == config['battery']['self_discharge_rate']
        assert battery.nominal_voltage == config['battery']['nominal_voltage']
        assert battery.degradation_per_cycle == config['battery']['degradation_per_cycle']
        assert battery.charge_rate_factor == config['battery']['charge_rate_factor']
        assert battery.discharge_rate_factor == config['battery']['discharge_rate_factor']
        
        # Verify initial state
        assert battery.state_of_charge == battery.max_soc
        assert battery.cycle_count == 0
        assert battery.remaining_capacity == battery.capacity_kwh

    def test_initialization_with_custom_seed(self):
        """Test that battery system initializes correctly with custom seed."""
        battery = BatterySystemSimulator(seed=123)
        
        # Verify configuration values were correctly loaded
        assert battery.capacity_kwh == config['battery']['capacity_kwh']
        assert battery.max_soc == config['battery']['max_soc']
        
        # Verify initial state
        assert battery.state_of_charge == battery.max_soc


class TestBatterySystemPowerLimits:
    """Test the power limit calculations of the battery system."""
    
    def test_power_limits_at_full_charge(self):
        """Test power limits when battery is fully charged."""
        battery = BatterySystemSimulator()
        battery.state_of_charge = battery.max_soc
        limits = battery.calculate_power_limits()
        
        # At full charge, charging limit should be near zero
        assert limits['charge_limit'] < 0.01
        # Discharge limit should be at maximum
        assert limits['discharge_limit'] == pytest.approx(
            battery.max_power_kw * battery.discharge_rate_factor, 
            rel=1e-5
        )
    
    def test_power_limits_at_min_charge(self):
        """Test power limits when battery is at minimum charge."""
        battery = BatterySystemSimulator()
        battery.state_of_charge = battery.min_soc
        limits = battery.calculate_power_limits()
        
        # At minimum charge, discharge limit should be near zero
        assert limits['discharge_limit'] < 0.01
        # Charge limit should be at maximum
        assert limits['charge_limit'] == pytest.approx(
            battery.max_power_kw * battery.charge_rate_factor,
            rel=1e-5
        )
    
    def test_power_limits_at_mid_charge(self):
        """Test power limits when battery is at 50% charge."""
        battery = BatterySystemSimulator()
        battery.state_of_charge = (battery.min_soc + battery.max_soc) / 2
        limits = battery.calculate_power_limits()
        
        # Both limits should be non-zero
        assert limits['charge_limit'] > 0
        assert limits['discharge_limit'] > 0


class TestBatterySystemStateUpdate:
    """Test the state update functionality of the battery system."""
    
    def test_charging_state_update(self):
        """Test state update during charging."""
        battery = BatterySystemSimulator()
        battery.state_of_charge = 0.5  # Start at 50%
        initial_soc = battery.state_of_charge
        
        # Charge with negative power (system convention)
        battery.update_state(-10, 1.0)  # -10 kW for 1 hour
        
        # SOC should increase
        assert battery.state_of_charge > initial_soc
        # Cycle count should increase
        assert battery.cycle_count > 0
    
    def test_discharging_state_update(self):
        """Test state update during discharging."""
        battery = BatterySystemSimulator()
        battery.state_of_charge = 0.8  # Start at 80%
        initial_soc = battery.state_of_charge
        
        # Discharge with positive power (system convention)
        battery.update_state(10, 1.0)  # 10 kW for 1 hour
        
        # SOC should decrease
        assert battery.state_of_charge < initial_soc
        # Cycle count should increase
        assert battery.cycle_count > 0
    
    def test_self_discharge(self):
        """Test self-discharge over time with no power flow."""
        battery = BatterySystemSimulator()
        battery.state_of_charge = 0.8  # Start at 80%
        initial_soc = battery.state_of_charge
        
        # No power flow, just self-discharge
        battery.update_state(0, 24.0)  # 0 kW for 24 hours
        
        # SOC should decrease due to self-discharge
        assert battery.state_of_charge < initial_soc
        # Cycle count should not change
        assert battery.cycle_count == 0
    
    def test_soc_limits(self):
        """Test that SOC stays within min/max limits."""
        battery = BatterySystemSimulator()
        
        # Try to discharge below minimum
        battery.state_of_charge = battery.min_soc + 0.01
        battery.update_state(100, 10.0)  # Heavy discharge
        
        # SOC should not go below minimum
        assert battery.state_of_charge >= battery.min_soc
        
        # Try to charge above maximum
        battery.state_of_charge = battery.max_soc - 0.01
        battery.update_state(-100, 10.0)  # Heavy charge
        
        # SOC should not go above maximum
        assert battery.state_of_charge <= battery.max_soc


class TestBatterySystemOutput:
    """Test the output generation of the battery system."""
    
    def test_output_format(self):
        """Test that output contains all required fields."""
        battery = BatterySystemSimulator()
        output = battery.generate_output(10)  # 10 kW discharge request
        
        # Check that all required fields are present
        assert 'power' in output
        assert 'current' in output
        assert 'voltage' in output
        assert 'soc' in output
        assert 'temperature' in output
        assert 'remaining_capacity' in output
        assert 'cycle_count' in output
    
    def test_power_limiting_charge(self):
        """Test that charge power is limited correctly."""
        battery = BatterySystemSimulator()
        battery.state_of_charge = 0.5  # 50% SOC
        
        # Request more power than maximum
        max_power = battery.max_power_kw * 2
        output = battery.generate_output(-max_power)  # Charging (negative)
        
        # Power should be limited to maximum
        assert abs(output['power']) <= battery.max_power_kw
    
    def test_power_limiting_discharge(self):
        """Test that discharge power is limited correctly."""
        battery = BatterySystemSimulator()
        battery.state_of_charge = 0.5  # 50% SOC
        
        # Request more power than maximum
        max_power = battery.max_power_kw * 2
        output = battery.generate_output(max_power)  # Discharging (positive)
        
        # Power should be limited to maximum
        assert output['power'] <= battery.max_power_kw
    
    def test_current_calculation(self):
        """Test that current is calculated correctly from power."""
        battery = BatterySystemSimulator()
        output = battery.generate_output(10)  # 10 kW discharge
        
        # Current = Power * 1000 / Voltage
        expected_current = 10 * 1000 / battery.nominal_voltage
        assert output['current'] == pytest.approx(expected_current, rel=1e-5)
        
        # Test zero power case
        output = battery.generate_output(0)
        assert output['current'] == 0


class TestBatterySystemDegradation:
    """Test the degradation modeling of the battery system."""
    
    def test_capacity_degradation(self):
        """Test that capacity degrades with cycling."""
        battery = BatterySystemSimulator()
        initial_capacity = battery.remaining_capacity
        
        # Simulate many cycles
        for _ in range(100):
            # Discharge
            battery.update_state(10, 1.0)
            # Charge
            battery.update_state(-10, 1.0)
        
        # Capacity should decrease
        assert battery.remaining_capacity < initial_capacity