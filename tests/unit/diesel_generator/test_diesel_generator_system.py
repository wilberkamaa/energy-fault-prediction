import pytest
import numpy as np
from src.diesel_generator import DieselGeneratorSimulator
from src.config import config

class TestDieselGeneratorInitialization:
    """Test the initialization of the diesel generator system."""
    
    def test_initialization_with_default_seed(self):
        """Test that diesel generator initializes correctly with default seed."""
        generator = DieselGeneratorSimulator()
        
        # Verify configuration values were correctly loaded
        assert generator.capacity_kva == config['diesel_generator']['capacity_kva']
        assert generator.fuel_tank_capacity == config['diesel_generator']['fuel_tank_capacity']
        assert generator.min_load_percent == config['diesel_generator']['min_load_percent']
        assert generator.min_runtime_hours == config['diesel_generator']['min_runtime_hours']
        assert generator.fuel_consumption_rate == config['diesel_generator']['fuel_consumption_rate']
        assert generator.maintenance_interval == config['diesel_generator']['maintenance_interval']
        
        # Verify initial state
        assert generator.running == False
        assert generator.runtime_hours == 0
        assert generator.fuel_level == generator.fuel_tank_capacity
        assert generator.last_maintenance == 0
        assert generator.temperature == 25
    
    def test_initialization_with_custom_seed(self):
        """Test that diesel generator initializes correctly with custom seed."""
        custom_seed = 123
        generator = DieselGeneratorSimulator(seed=custom_seed)
        
        # Verify configuration values were correctly loaded
        assert generator.capacity_kva == config['diesel_generator']['capacity_kva']
        assert generator.fuel_tank_capacity == config['diesel_generator']['fuel_tank_capacity']


class TestDieselGeneratorFuelConsumption:
    """Test the fuel consumption calculation of the diesel generator."""
    
    def test_fuel_consumption_when_not_running(self):
        """Test that fuel consumption is zero when generator is not running."""
        generator = DieselGeneratorSimulator()
        generator.running = False
        
        consumption = generator.calculate_fuel_consumption(0.5)  # 50% load
        assert consumption == 0.0
    
    def test_fuel_consumption_at_idle(self):
        """Test fuel consumption at idle load."""
        generator = DieselGeneratorSimulator()
        generator.running = True
        
        consumption = generator.calculate_fuel_consumption(0.0)  # 0% load
        assert consumption == generator.fuel_consumption_rate['idle']
    
    def test_fuel_consumption_at_full_load(self):
        """Test fuel consumption at full load."""
        generator = DieselGeneratorSimulator()
        generator.running = True
        
        consumption = generator.calculate_fuel_consumption(1.0)  # 100% load
        assert consumption == generator.fuel_consumption_rate['full_load']
    
    def test_fuel_consumption_at_partial_load(self):
        """Test fuel consumption at partial load."""
        generator = DieselGeneratorSimulator()
        generator.running = True
        
        # Test at 50% load
        load_percent = 0.5
        expected_consumption = (
            generator.fuel_consumption_rate['idle'] +
            (generator.fuel_consumption_rate['full_load'] - generator.fuel_consumption_rate['idle']) *
            load_percent
        )
        
        consumption = generator.calculate_fuel_consumption(load_percent)
        assert consumption == expected_consumption


class TestDieselGeneratorMaintenance:
    """Test the maintenance functionality of the diesel generator."""
    
    def test_needs_maintenance_when_new(self):
        """Test that a new generator doesn't need maintenance."""
        generator = DieselGeneratorSimulator()
        assert generator.needs_maintenance() == False
    
    def test_needs_maintenance_after_interval(self):
        """Test that generator needs maintenance after running for the maintenance interval."""
        generator = DieselGeneratorSimulator()
        generator.runtime_hours = generator.maintenance_interval
        assert generator.needs_maintenance() == True
    
    def test_needs_maintenance_after_maintenance(self):
        """Test that generator doesn't need maintenance right after maintenance is performed."""
        generator = DieselGeneratorSimulator()
        generator.runtime_hours = generator.maintenance_interval
        generator.last_maintenance = generator.runtime_hours
        assert generator.needs_maintenance() == False


class TestDieselGeneratorStartStop:
    """Test the start and stop functionality of the diesel generator."""
    
    def test_start_generator_normal(self):
        """Test starting the generator under normal conditions."""
        generator = DieselGeneratorSimulator()
        result = generator.start_generator()
        assert result == True
        assert generator.running == True
    
    def test_start_generator_no_fuel(self):
        """Test that generator won't start without fuel."""
        generator = DieselGeneratorSimulator()
        generator.fuel_level = 0
        result = generator.start_generator()
        assert result == False
        assert generator.running == False
    
    def test_start_generator_needs_maintenance(self):
        """Test that generator won't start when maintenance is needed."""
        generator = DieselGeneratorSimulator()
        generator.runtime_hours = generator.maintenance_interval
        result = generator.start_generator()
        assert result == False
        assert generator.running == False
    
    def test_stop_generator(self):
        """Test stopping the generator."""
        generator = DieselGeneratorSimulator()
        generator.running = True
        generator.stop_generator()
        assert generator.running == False


class TestDieselGeneratorOutput:
    """Test the output generation of the diesel generator."""
    
    def test_output_format(self):
        """Test that output contains all required fields."""
        generator = DieselGeneratorSimulator()
        output = generator.generate_output(500)  # 500 kW request
        
        # Check that all required fields are present
        assert 'running' in output
        assert 'power' in output
        assert 'frequency' in output
        assert 'temperature' in output
        assert 'fuel_level' in output
        assert 'runtime' in output
        assert 'needs_maintenance' in output
    
    def test_output_when_not_running_below_min_load(self):
        """Test output when power request is below minimum load."""
        generator = DieselGeneratorSimulator()
        min_power = generator.capacity_kva * generator.min_load_percent - 1  # Just below min load
        output = generator.generate_output(min_power)
        
        assert output['running'] == False
        assert output['power'] == 0
        assert output['frequency'] == 0
    
    def test_output_when_running_above_min_load(self):
        """Test output when power request is above minimum load."""
        generator = DieselGeneratorSimulator()
        power_request = generator.capacity_kva * generator.min_load_percent + 10  # Above min load
        output = generator.generate_output(power_request)
        
        assert output['running'] == True
        assert output['power'] == power_request
        assert output['frequency'] > 0
        assert output['temperature'] > 25  # Should increase from ambient
        assert output['fuel_level'] < generator.fuel_tank_capacity  # Should consume fuel
        assert output['runtime'] > 0
    
    def test_output_with_no_fuel(self):
        """Test output when generator has no fuel."""
        generator = DieselGeneratorSimulator()
        generator.fuel_level = 0
        power_request = generator.capacity_kva * 0.5  # 50% load
        output = generator.generate_output(power_request)
        
        assert output['running'] == False
        assert output['power'] == 0
        assert output['fuel_level'] == 0
    
    def test_output_with_maintenance_needed(self):
        """Test output when generator needs maintenance."""
        generator = DieselGeneratorSimulator()
        generator.runtime_hours = generator.maintenance_interval
        power_request = generator.capacity_kva * 0.5  # 50% load
        output = generator.generate_output(power_request)
        
        assert output['needs_maintenance'] == True
        # Generator should not start if it needs maintenance
        assert output['running'] == False
        assert output['power'] == 0
    
    def test_minimum_runtime_enforcement(self):
        """Test that generator keeps running to meet minimum runtime."""
        generator = DieselGeneratorSimulator()
        
        # First start the generator with sufficient load
        high_power = generator.capacity_kva * 0.5  # 50% load
        generator.generate_output(high_power)
        assert generator.running == True
        
        # Now request low power, but generator should keep running due to minimum runtime
        low_power = generator.capacity_kva * (generator.min_load_percent - 0.1)  # Below min load
        output = generator.generate_output(low_power)
        
        # Generator should still be running due to minimum runtime requirement
        assert output['running'] == True