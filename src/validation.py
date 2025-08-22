import numpy as np
import pandas as pd
from typing import Dict, Any
from src.config import config

class DataValidator:
    """Validates and cleans the generated data."""
    
    def __init__(self):
        # Get validation ranges from config
        self.valid_ranges = config['validation']['valid_ranges']
        self.power_balance_tolerance = config['validation']['power_balance_tolerance']
    
    def validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the data, ensuring all values are within valid ranges."""
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Clip numerical values to valid ranges
        for column, (min_val, max_val) in self.valid_ranges.items():
            if column in df.columns:
                df[column] = df[column].clip(min_val, max_val)
        
        # Validate power balance
        total_generation = (
            df['solar_power'] +
            df['generator_power'] +
            df['grid_power'] +
            df['battery_power']
        )
        total_load = df['load_demand']
        
        # Allow for small imbalances (from config)
        tolerance = config['validation']['power_balance_tolerance'] * df['load_demand'].max()
        df['power_balanced'] = (total_generation - total_load).abs() <= tolerance
        
        # Check for NaN values
        if df.isna().any().any():
            print("Warning: NaN values found in dataset")
            df = df.fillna(config['validation']['nan_fill_value'])  # Replace NaN with configured value
        
        return df
    
    def check_power_balance(self, data: Dict[str, Any], tolerance: float = None) -> bool:
        """Verify that power generation matches load demand within tolerance.
        Returns True if balance is maintained, False otherwise."""
        if tolerance is None:
            tolerance = config['validation']['power_balance_tolerance']
            
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
