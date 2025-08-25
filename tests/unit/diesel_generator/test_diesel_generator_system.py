import pytest
import numpy as np
from src.diesel_generator import DieselGeneratorSimulator
from src.config import config

class TestDieselGeneratorInitialization:
    """Test the initialization of the diesel generator simulator."""
    
    def test_initialization_with_default_seed(self):
        """Test that diesel generator simulator initializes correctly with default seed."""
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
        """Test that diesel generator simulator initializes correctly with custom seed."""
        custom_seed = 123
        generator = DieselGeneratorSimulator(seed=custom_seed)
        
        # Verify configuration values were correctly loaded
        assert generator.capacity_kva == config['diesel_generator']['capacity_kva']
        assert generator.fuel_tank_capacity == config['diesel_generator']['fuel_tank_capacity']

class TestFuelConsumptionCalculation:
    """Test the fuel consumption calculation functionality."""
    
    def test_fuel_consumption_when_not_running(self):
        """Test that fuel consumption is zero when generator is not running."""
        generator = DieselGeneratorSimulator()
        # Generator is not running by default
        assert generator.running == False
        
        # Test fuel consumption at different loads
        assert generator.calculate_fuel_consumption(0.0) == 0.0
        assert generator.calculate_fuel_consumption(0.5) == 0.0
        assert generator.calculate_fuel_consumption(1.0) == 0.0
    
    def test_fuel_consumption_at_idle(self):
        """Test fuel consumption calculation at idle load."""
        generator = DieselGeneratorSimulator()
        generator.running = True
        
        # Test fuel consumption at idle (0% load)
        idle_consumption = generator.calculate_fuel_consumption(0.0)
        assert idle_consumption == config['diesel_generator']['fuel_consumption_rate']['idle']
    
    def test_fuel_consumption_at_full_load(self):
        """Test fuel consumption calculation at full load."""
        generator = DieselGeneratorSimulator()
        generator.running = True
        
        # Test fuel consumption at full load (100% load)
        full_load_consumption = generator.calculate_fuel_consumption(1.0)
        assert full_load_consumption == config['diesel_generator']['fuel_consumption_rate']['full_load']
    
    def test_fuel_consumption_at_partial_load(self):
        """Test fuel consumption calculation at partial load."""
        generator = DieselGeneratorSimulator()
        generator.running = True
        
        # Test fuel consumption at 50% load (should be linear interpolation)
        idle_rate = config['diesel_generator']['fuel_consumption_rate']['idle']
        full_load_rate = config['diesel_generator']['fuel_consumption_rate']['full_load']
        expected_consumption = idle_rate + (full_load_rate - idle_rate) * 0.5
        
        partial_load_consumption = generator.calculate_fuel_consumption(0.5)
        assert partial_load_consumption == expected_consumption

class TestMaintenanceFunctionality:
    """Test the maintenance functionality."""
    
    def test_new_generator_doesnt_need_maintenance(self):
        """Test that a new generator doesn't need maintenance."""
        generator = DieselGeneratorSimulator()
        assert generator.needs_maintenance() == False
    
    def test_generator_needs_maintenance_after_interval(self):
        """Test that generator needs maintenance after running for the maintenance interval."""
        generator = DieselGeneratorSimulator()
        generator.runtime_hours = generator.maintenance_interval
        assert generator.needs_maintenance() == True
    
    def test_generator_doesnt_need_maintenance_after_service(self):
        """Test that generator doesn't need maintenance after being serviced."""
        generator = DieselGeneratorSimulator()
        generator.runtime_hours = generator.maintenance_interval
        assert generator.needs_maintenance() == True
        
        # Simulate maintenance service
        generator.last_maintenance = generator.runtime_hours
        assert generator.needs_maintenance() == False

class TestStartStopFunctionality:
    """Test the generator start/stop functionality."""
    
    def test_start_generator_normal_conditions(self):
        """Test starting the generator under normal conditions."""
        generator = DieselGeneratorSimulator()
        result = generator.start_generator()
        
        assert result == True
        assert generator.running == True
    
    def test_start_generator_no_fuel(self):
        """Test starting the generator with no fuel."""
        generator = DieselGeneratorSimulator()
        generator.fuel_level = 0
        
        result = generator.start_generator()
        assert result == False
        assert generator.running == False
    
    def test_start_generator_needs_maintenance(self):
        """Test starting the generator when maintenance is needed."""
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

class TestOutputGeneration:
    """Test the generator output generation functionality."""
    
    def test_output_format(self):
        """Test that the output has the correct format."""
        generator = DieselGeneratorSimulator()
        output = generator.generate_output(0)
        
        # Check that all expected keys are present
        expected_keys = ['running', 'power', 'frequency', 'temperature', 
                         'fuel_level', 'runtime', 'needs_maintenance']
        for key in expected_keys:
            assert key in output
    
    def test_output_when_not_running(self):
        """Test output generation when generator is not running."""
        generator = DieselGeneratorSimulator()
        output = generator.generate_output(0)
        
        assert output['running'] == False
        assert output['power'] == 0
        assert output['frequency'] == 0
    
    def test_output_when_running(self):
        """Test output generation when generator is running."""
        generator = DieselGeneratorSimulator()
        # Request enough power to start the generator
        power_request = generator.capacity_kva * generator.min_load_percent * 1.1
        output = generator.generate_output(power_request)
        
        assert output['running'] == True
        assert output['power'] == power_request
        assert output['frequency'] > 0
        assert output['fuel_level'] < generator.fuel_tank_capacity
    
    def test_output_when_no_fuel(self):
        """Test output generation when generator has no fuel."""
        generator = DieselGeneratorSimulator()
        generator.fuel_level = 0
        output = generator.generate_output(generator.capacity_kva)
        
        assert output['running'] == False
        assert output['power'] == 0
        assert output['fuel_level'] == 0
    
    def test_output_when_maintenance_needed(self):
        """Test output generation when generator needs maintenance."""
        generator = DieselGeneratorSimulator()
        generator.runtime_hours = generator.maintenance_interval
        output = generator.generate_output(generator.capacity_kva)
        
        assert output['running'] == False
        assert output['power'] == 0
        assert output['needs_maintenance'] == True
    
    def test_minimum_runtime_enforcement(self):
        """Test that minimum runtime is enforced."""
        generator = DieselGeneratorSimulator()
        
        # Start the generator with sufficient load
        power_request = generator.capacity_kva * generator.min_load_percent * 1.1
        generator.generate_output(power_request)
        assert generator.running == True
        
        # Try to stop by requesting low power before minimum runtime is reached
        output = generator.generate_output(0)
        assert output['running'] == True  # Should still be running