import numpy as np
from typing import Dict, Any

class DieselGeneratorSimulator:
    """Simulates a 2 MVA diesel generator system."""
    
    def __init__(self, capacity_kva: float = 2000, seed: int = 42):
        self.capacity_kva = capacity_kva
        self.fuel_tank_capacity = 5000  # Liters
        np.random.seed(seed)
        
        # Operating parameters
        self.min_load_percent = 0.20  # Minimum loading (20%)
        self.fuel_consumption_rate = {
            'idle': 15,      # L/hour at idle
            'full_load': 120 # L/hour at full load
        }
        self.maintenance_interval = 500  # hours
        
    def calculate_fuel_consumption(self, load_percent: float) -> float:
        """Calculate fuel consumption based on load percentage."""
        if load_percent < self.min_load_percent:
            return self.fuel_consumption_rate['idle']
        
        # Linear interpolation between idle and full load consumption
        consumption = (self.fuel_consumption_rate['idle'] + 
                      (self.fuel_consumption_rate['full_load'] - 
                       self.fuel_consumption_rate['idle']) * load_percent)
        return consumption
    
    def calculate_efficiency(self, load_percent: float) -> float:
        """Calculate generator efficiency based on load percentage."""
        # Typical diesel generator efficiency curve
        if load_percent < self.min_load_percent:
            return 0.25
        elif load_percent > 0.9:
            return 0.38
        else:
            # Peak efficiency around 75% load
            return 0.35 * np.sin(np.pi * (load_percent - 0.2) / 1.4) + 0.3
    
    def generate_output(self, df, load_demand: np.ndarray, 
                       pv_output: np.ndarray,
                       battery_output: np.ndarray) -> Dict[str, Any]:
        """Generate generator output parameters."""
        # Initialize arrays
        hours = len(df)
        fuel_level = np.zeros(hours)
        output_power = np.zeros(hours)
        frequency = np.zeros(hours)
        temperature = np.zeros(hours)
        running_hours = np.zeros(hours)
        
        # Initial conditions
        fuel_level[0] = self.fuel_tank_capacity
        last_maintenance = 0
        cumulative_running_hours = 0
        
        for i in range(hours):
            # Calculate required power
            required_power = max(0, load_demand[i] - pv_output[i] - battery_output[i])
            
            # Calculate load percentage
            load_percent = min(required_power / self.capacity_kva, 1.0)
            
            if load_percent > 0:
                # Generator is running
                cumulative_running_hours += 1
                running_hours[i] = cumulative_running_hours
                
                # Calculate fuel consumption
                fuel_consumption = self.calculate_fuel_consumption(load_percent)
                
                # Update fuel level
                if i > 0:
                    fuel_level[i] = fuel_level[i-1] - fuel_consumption
                
                # Automatic refill when below 20%
                if fuel_level[i] < 0.2 * self.fuel_tank_capacity:
                    fuel_level[i] = self.fuel_tank_capacity
                
                # Calculate output
                efficiency = self.calculate_efficiency(load_percent)
                output_power[i] = required_power * efficiency
                
                # Calculate frequency
                base_freq = 60 + 0.1 * (load_percent - 0.5)
                frequency[i] = base_freq + np.random.normal(0, 0.01)
                
                # Calculate temperature
                base_temp = 80 + 40 * load_percent
                temperature[i] = base_temp + np.random.normal(0, 2)
                
                # Maintenance effect
                hours_since_maintenance = cumulative_running_hours - last_maintenance
                if hours_since_maintenance >= self.maintenance_interval:
                    last_maintenance = cumulative_running_hours
                    efficiency *= 1.05  # Efficiency boost after maintenance
            else:
                # Generator is off
                if i > 0:
                    fuel_level[i] = fuel_level[i-1]
                frequency[i] = 0
                temperature[i] = df['temperature'][i]  # Ambient temperature
        
        return {
            'fuel_level': fuel_level,
            'output_power': output_power,
            'frequency': frequency,
            'temperature': temperature,
            'running_hours': running_hours
        }
