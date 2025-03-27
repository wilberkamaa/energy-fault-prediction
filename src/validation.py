import numpy as np
from typing import Dict, Any

class DataValidator:
    """Validates and enforces physical constraints on system data."""
    
    def __init__(self):
        # System constraints
        self.constraints = {
            'battery': {
                'soc_range': (0.2, 0.95),
                'voltage_range': (400, 600),
                'temperature_range': (10, 45)
            },
            'solar': {
                'irradiance_range': (0, 1200),
                'efficiency_range': (0.1, 0.2),
                'temperature_range': (10, 85)
            },
            'generator': {
                'frequency_range': (59.5, 60.5),
                'temperature_range': (60, 95),
                'fuel_level_range': (0, 5000)
            },
            'grid': {
                'voltage_range': (22500, 27500),  # ±10% of 25kV
                'frequency_range': (49.5, 50.5)
            },
            'load': {
                'power_factor_range': (0.8, 1.0)
            }
        }
    
    def validate_and_clip(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clip data to ensure it meets physical constraints.
        Returns cleaned data and a list of validation messages.
        """
        messages = []
        cleaned_data = data.copy()
        
        # Battery validation
        if 'battery' in data:
            battery = data['battery']
            if 'soc' in battery:
                cleaned_data['battery']['soc'] = np.clip(
                    battery['soc'],
                    *self.constraints['battery']['soc_range']
                )
            if 'voltage' in battery:
                cleaned_data['battery']['voltage'] = np.clip(
                    battery['voltage'],
                    *self.constraints['battery']['voltage_range']
                )
            if 'temperature' in battery:
                cleaned_data['battery']['temperature'] = np.clip(
                    battery['temperature'],
                    *self.constraints['battery']['temperature_range']
                )
        
        # Solar validation
        if 'solar' in data:
            solar = data['solar']
            if 'irradiance' in solar:
                cleaned_data['solar']['irradiance'] = np.clip(
                    solar['irradiance'],
                    *self.constraints['solar']['irradiance_range']
                )
            if 'efficiency' in solar:
                cleaned_data['solar']['efficiency'] = np.clip(
                    solar['efficiency'],
                    *self.constraints['solar']['efficiency_range']
                )
            if 'cell_temperature' in solar:
                cleaned_data['solar']['cell_temperature'] = np.clip(
                    solar['cell_temperature'],
                    *self.constraints['solar']['temperature_range']
                )
        
        # Generator validation
        if 'generator' in data:
            generator = data['generator']
            if 'frequency' in generator:
                cleaned_data['generator']['frequency'] = np.clip(
                    generator['frequency'],
                    *self.constraints['generator']['frequency_range']
                )
            if 'temperature' in generator:
                cleaned_data['generator']['temperature'] = np.clip(
                    generator['temperature'],
                    *self.constraints['generator']['temperature_range']
                )
            if 'fuel_level' in generator:
                cleaned_data['generator']['fuel_level'] = np.clip(
                    generator['fuel_level'],
                    *self.constraints['generator']['fuel_level_range']
                )
        
        # Grid validation
        if 'grid' in data:
            grid = data['grid']
            if 'voltage' in grid:
                cleaned_data['grid']['voltage'] = np.clip(
                    grid['voltage'],
                    *self.constraints['grid']['voltage_range']
                )
            if 'frequency' in grid:
                cleaned_data['grid']['frequency'] = np.clip(
                    grid['frequency'],
                    *self.constraints['grid']['frequency_range']
                )
        
        # Load validation
        if 'load' in data:
            load = data['load']
            if 'power_factor' in load:
                cleaned_data['load']['power_factor'] = np.clip(
                    load['power_factor'],
                    *self.constraints['load']['power_factor_range']
                )
        
        return cleaned_data
    
    def check_power_balance(self, data: Dict[str, Any], tolerance: float = 0.01) -> bool:
        """
        Verify that power generation matches load demand within tolerance.
        Returns True if balance is maintained, False otherwise.
        """
        total_generation = 0
        
        # Sum up all generation sources
        if 'solar' in data and 'power_output' in data['solar']:
            total_generation += data['solar']['power_output']
        
        if 'generator' in data and 'output_power' in data['generator']:
            total_generation += data['generator']['output_power']
        
        if 'grid' in data and 'available' in data['grid']:
            grid_power = np.where(
                data['grid']['available'],
                data['load']['active_power'] - total_generation,
                0
            )
            total_generation += grid_power
        
        if 'battery' in data and 'power_output' in data['battery']:
            total_generation += data['battery']['power_output']
        
        # Get load demand
        load_demand = data['load']['active_power']
        
        # Check balance
        imbalance = np.abs(total_generation - load_demand)
        max_allowed_imbalance = load_demand * tolerance
        
        return np.all(imbalance <= max_allowed_imbalance)
