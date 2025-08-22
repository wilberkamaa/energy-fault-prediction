import numpy as np
from typing import Dict, Any
from src.config import config

class DieselGeneratorSimulator:
    """Simulates a diesel generator with realistic behavior."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
        # Load generator parameters from config
        gen_config = config['diesel_generator']
        self.capacity_kva = gen_config['capacity_kva']
        self.fuel_tank_capacity = gen_config['fuel_tank_capacity']
        self.min_load_percent = gen_config['min_load_percent']
        self.min_runtime_hours = gen_config['min_runtime_hours']
        self.fuel_consumption_rate = gen_config['fuel_consumption_rate']
        self.maintenance_interval = gen_config['maintenance_interval']
        
        # Initialize state
        self.running = False
        self.runtime_hours = 0
        self.fuel_level = self.fuel_tank_capacity
        self.last_maintenance = 0
        self.temperature = 25  # Starting at ambient temperature
    
    def calculate_fuel_consumption(self, load_percent: float) -> float:
        """Calculate fuel consumption based on load."""
        if not self.running:
            return 0.0
        
        # Linear interpolation between idle and full load consumption
        fuel_rate = (
            self.fuel_consumption_rate['idle'] +
            (self.fuel_consumption_rate['full_load'] - self.fuel_consumption_rate['idle']) *
            load_percent
        )
        
        return fuel_rate
    
    def needs_maintenance(self) -> bool:
        """Check if maintenance is needed based on runtime."""
        return (self.runtime_hours - self.last_maintenance) >= self.maintenance_interval
    
    def start_generator(self) -> bool:
        """Attempt to start the generator."""
        if self.fuel_level <= 0:
            return False
        
        if self.needs_maintenance():
            return False
        
        self.running = True
        return True
    
    def stop_generator(self) -> None:
        """Stop the generator."""
        self.running = False
    
    def generate_output(self, power_request: float, time_step_hours: float = 1.0) -> Dict[str, Any]:
        """Generate generator output parameters."""
        # Calculate load percentage
        load_percent = abs(power_request) / self.capacity_kva
        
        # Check if generator should be running
        should_run = load_percent >= self.min_load_percent
        
        # Start or stop generator based on load
        if should_run and not self.running:
            self.start_generator()
        elif not should_run and self.running:
            if self.runtime_hours % self.min_runtime_hours < 1:
                should_run = True  # Keep running to meet minimum runtime
            else:
                self.stop_generator()
        
        # Calculate output parameters
        if self.running and self.fuel_level > 0:
            # Update runtime
            self.runtime_hours += time_step_hours
            
            # Calculate fuel consumption
            fuel_consumption = self.calculate_fuel_consumption(load_percent) * time_step_hours
            self.fuel_level = max(0, self.fuel_level - fuel_consumption)
            
            # Calculate temperature (simplified model)
            target_temp = 25 + 60 * load_percent  # Higher load = higher temp
            self.temperature += (target_temp - self.temperature) * 0.1
            
            # Calculate frequency variation (simplified model)
            frequency = 50 + np.random.normal(0, 0.1) - 0.5 * load_percent
            
            # Set output power based on available fuel
            if self.fuel_level > 0:
                output_power = power_request
            else:
                output_power = 0
                self.stop_generator()
        else:
            output_power = 0
            frequency = 0
            
        return {
            'running': self.running,
            'power': output_power,
            'frequency': frequency if self.running else 0,
            'temperature': self.temperature,
            'fuel_level': self.fuel_level,
            'runtime': self.runtime_hours,
            'needs_maintenance': self.needs_maintenance()
        }
