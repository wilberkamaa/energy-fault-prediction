import pytest
import numpy as np
import pandas as pd
from src.fault_injection import FaultInjectionSystem, FaultType, FaultEvent
from src.config import config

class TestFaultInjectionInitialization:
    """Test the initialization of the fault injection system."""
    
    def test_initialization_with_default_seed(self):
        """Test that fault injection system initializes correctly with default seed."""
        fault_injector = FaultInjectionSystem()
        
        # Verify configuration values were correctly loaded
        assert fault_injector.fault_probabilities == config['fault_injection']['fault_probabilities']
        assert fault_injector.fault_durations == config['fault_injection']['fault_durations']
    
    def test_initialization_with_custom_seed(self):
        """Test that fault injection system initializes correctly with custom seed."""
        custom_seed = 123
        fault_injector = FaultInjectionSystem(seed=custom_seed)
        
        # Verify configuration values were correctly loaded
        assert fault_injector.fault_probabilities == config['fault_injection']['fault_probabilities']
        assert fault_injector.fault_durations == config['fault_injection']['fault_durations']

class TestFaultConditionChecking:
    """Test the fault condition checking functionality."""
    
    def test_check_fault_conditions_no_conditions_met(self):
        """Test that no faults are detected when conditions aren't met."""
        fault_injector = FaultInjectionSystem(seed=42)
        
        # Create system state with all parameters above thresholds
        system_state = {
            'grid_voltage': np.array([config['fault_injection']['thresholds']['grid_voltage'] * 1.2]),
            'inverter_temp': np.array([config['fault_injection']['thresholds']['inverter_temp'] - 10]),
            'generator_runtime': np.array([config['fault_injection']['thresholds']['generator_runtime'] - 10]),
            'battery_soc': np.array([config['fault_injection']['thresholds']['battery_soc'] + 0.3])
        }
        
        potential_faults = fault_injector.check_fault_conditions(system_state, 0)
        assert len(potential_faults) == 0
    
    def test_check_fault_conditions_grid_voltage_low(self):
        """Test that line short circuit fault is detected when grid voltage is low."""
        fault_injector = FaultInjectionSystem(seed=42)
        
        # Create system state with low grid voltage
        system_state = {
            'grid_voltage': np.array([config['fault_injection']['thresholds']['grid_voltage'] * 0.9]),
            'inverter_temp': np.array([config['fault_injection']['thresholds']['inverter_temp'] - 10]),
            'generator_runtime': np.array([config['fault_injection']['thresholds']['generator_runtime'] - 10]),
            'battery_soc': np.array([config['fault_injection']['thresholds']['battery_soc'] + 0.3])
        }
        
        potential_faults = fault_injector.check_fault_conditions(system_state, 0)
        assert len(potential_faults) == 1
        assert potential_faults[0][0] == FaultType.LINE_SHORT_CIRCUIT
        assert potential_faults[0][1] > config['fault_injection']['fault_probabilities']['LINE_SHORT_CIRCUIT']
    
    def test_check_fault_conditions_inverter_temp_high(self):
        """Test that inverter IGBT failure is detected when inverter temperature is high."""
        fault_injector = FaultInjectionSystem(seed=42)
        
        # Create system state with high inverter temperature
        system_state = {
            'grid_voltage': np.array([config['fault_injection']['thresholds']['grid_voltage'] * 1.2]),
            'inverter_temp': np.array([config['fault_injection']['thresholds']['inverter_temp'] + 10]),
            'generator_runtime': np.array([config['fault_injection']['thresholds']['generator_runtime'] - 10]),
            'battery_soc': np.array([config['fault_injection']['thresholds']['battery_soc'] + 0.3])
        }
        
        potential_faults = fault_injector.check_fault_conditions(system_state, 0)
        assert len(potential_faults) == 1
        assert potential_faults[0][0] == FaultType.INVERTER_IGBT_FAILURE
        assert potential_faults[0][1] > config['fault_injection']['fault_probabilities']['INVERTER_IGBT_FAILURE']
    
    def test_check_fault_conditions_battery_soc_low(self):
        """Test that battery overdischarge is detected when battery SOC is low."""
        fault_injector = FaultInjectionSystem(seed=42)
        
        # Create system state with low battery SOC
        system_state = {
            'grid_voltage': np.array([config['fault_injection']['thresholds']['grid_voltage'] * 1.2]),
            'inverter_temp': np.array([config['fault_injection']['thresholds']['inverter_temp'] - 10]),
            'generator_runtime': np.array([config['fault_injection']['thresholds']['generator_runtime'] - 10]),
            'battery_soc': np.array([config['fault_injection']['thresholds']['battery_soc'] - 0.05])
        }
        
        potential_faults = fault_injector.check_fault_conditions(system_state, 0)
        assert len(potential_faults) == 1
        assert potential_faults[0][0] == FaultType.BATTERY_OVERDISCHARGE
        assert potential_faults[0][1] > config['fault_injection']['fault_probabilities']['BATTERY_OVERDISCHARGE']

class TestFaultEventGeneration:
    """Test the generation of fault events."""
    
    def test_generate_fault_events_no_faults(self):
        """Test that no fault events are generated when conditions aren't met."""
        fault_injector = FaultInjectionSystem(seed=42)
        
        # Create dataframe and system state with no fault conditions
        df = pd.DataFrame(index=range(24))  # 24 hours
        system_state = {
            'grid_voltage': np.array([config['fault_injection']['thresholds']['grid_voltage'] * 1.2] * 24),
            'inverter_temp': np.array([config['fault_injection']['thresholds']['inverter_temp'] - 10] * 24),
            'generator_runtime': np.array([config['fault_injection']['thresholds']['generator_runtime'] - 10] * 24),
            'battery_soc': np.array([config['fault_injection']['thresholds']['battery_soc'] + 0.3] * 24)
        }
        
        fault_events = fault_injector.generate_fault_events(df, system_state)
        
        # Verify no faults occurred
        assert not np.any(fault_events['occurred'])
        assert np.all(fault_events['type'] == 'NO_FAULT')
        assert not np.any(fault_events['severity'])
        assert not np.any(fault_events['start'])
        assert not np.any(fault_events['duration'])
    
    def test_generate_fault_events_with_forced_fault(self, monkeypatch):
        """Test fault event generation with a forced fault (mocking random)."""
        fault_injector = FaultInjectionSystem(seed=42)
        
        # Mock random.random to always return 0 (ensuring fault occurs)
        monkeypatch.setattr(np.random, 'random', lambda: 0)
        
        # Mock random.randint to return consistent duration
        monkeypatch.setattr(np.random, 'randint', lambda *args: 3)
        
        # Mock random.uniform to return consistent severity
        monkeypatch.setattr(np.random, 'uniform', lambda *args: 0.5)
        
        # Create dataframe and system state with fault condition
        df = pd.DataFrame(index=range(24))  # 24 hours
        system_state = {
            'grid_voltage': np.array([config['fault_injection']['thresholds']['grid_voltage'] * 0.9] * 24),
            'inverter_temp': np.array([config['fault_injection']['thresholds']['inverter_temp'] - 10] * 24),
            'generator_runtime': np.array([config['fault_injection']['thresholds']['generator_runtime'] - 10] * 24),
            'battery_soc': np.array([config['fault_injection']['thresholds']['battery_soc'] + 0.3] * 24)
        }
        
        fault_events = fault_injector.generate_fault_events(df, system_state)
        
        # Verify fault occurred
        assert fault_events['occurred'][0]
        assert fault_events['type'][0] == 'LINE_SHORT_CIRCUIT'
        assert fault_events['severity'][0] == 0.5
        assert fault_events['start'][0]
        assert fault_events['duration'][0] == 3

class TestFaultEffects:
    """Test the generation of fault effects."""
    
    def test_generate_fault_effects_line_short_circuit(self):
        """Test that appropriate effects are generated for line short circuit."""
        fault_injector = FaultInjectionSystem(seed=42)
        
        effects = fault_injector._generate_fault_effects(FaultType.LINE_SHORT_CIRCUIT, 0.5)
        
        # Verify effects contain expected keys
        assert 'voltage_drop' in effects
        assert 'current_spike' in effects
        
        # Verify effect values are calculated correctly
        fault_config = config['fault_injection']['fault_effects']
        expected_voltage_drop = fault_config['line_short_circuit']['voltage_drop_base'] + \
                              fault_config['line_short_circuit']['voltage_drop_factor'] * 0.5
        expected_current_spike = fault_config['line_short_circuit']['current_spike_base'] + \
                               fault_config['line_short_circuit']['current_spike_factor'] * 0.5
        
        assert effects['voltage_drop'] == expected_voltage_drop
        assert effects['current_spike'] == expected_current_spike
    
    def test_generate_fault_effects_battery_overdischarge(self):
        """Test that appropriate effects are generated for battery overdischarge."""
        fault_injector = FaultInjectionSystem(seed=42)
        
        effects = fault_injector._generate_fault_effects(FaultType.BATTERY_OVERDISCHARGE, 0.7)
        
        # Verify effects contain expected keys
        assert 'capacity_loss' in effects
        assert 'internal_resistance' in effects
        
        # Verify effect values are calculated correctly
        fault_config = config['fault_injection']['fault_effects']
        expected_capacity_loss = fault_config['battery_overdischarge']['capacity_loss_factor'] * 0.7
        expected_internal_resistance = fault_config['battery_overdischarge']['internal_resistance_base'] + \
                                     fault_config['battery_overdischarge']['internal_resistance_factor'] * 0.7
        
        assert effects['capacity_loss'] == expected_capacity_loss
        assert effects['internal_resistance'] == expected_internal_resistance