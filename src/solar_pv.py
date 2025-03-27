import numpy as np
from typing import Dict, Any
import pandas as pd

class SolarPVSimulator:
    """Simulates a 500 kW Solar PV system."""
    
    def __init__(self, capacity_kw: float = 500, seed: int = 42):
        self.capacity_kw = capacity_kw
        np.random.seed(seed)
        
        # System parameters
        self.nominal_efficiency = 0.15
        self.temp_coefficient = -0.004  # Efficiency loss per degree C above 25°C
        self.dust_loss_rate = 0.001    # Daily loss rate due to dust
        self.noct = 45  # Nominal Operating Cell Temperature
        self.base_efficiency = 0.18  # Base efficiency at 25°C
        
    def calculate_irradiance(self, df) -> np.ndarray:
        """Calculate solar irradiance considering time and weather."""
        # Base irradiance pattern (clear sky)
        hour_shifted = (df['weather_hour'] - 6) % 24  # Shift to start at 6 AM
        base_irradiance = 1000 * np.abs(np.sin(np.pi * hour_shifted / 12))
        
        # Seasonal variation (Kenya's position near equator)
        seasonal_factor = 1 - 0.3 * np.sin(2 * np.pi * (df['weather_day_of_year'] + 81) / 365)
        
        # Apply cloud effects
        cloud_effect = 1 - df['weather_cloud_cover']
        
        # Combine factors
        irradiance = base_irradiance * seasonal_factor * cloud_effect
        
        # Add random noise
        noise = np.random.normal(0, 0.05, len(df))
        irradiance = irradiance * (1 + noise)
        
        return np.clip(irradiance, 0, 1200)  # Max 1200 W/m²
        
    def calculate_cell_temperature(self, ambient_temp: np.ndarray, 
                                 irradiance: np.ndarray) -> np.ndarray:
        """Calculate PV cell temperature based on ambient temperature and irradiance."""
        # NOCT method for cell temperature
        return ambient_temp + (self.noct - 20) * irradiance / 800

    def calculate_efficiency(self, cell_temp: np.ndarray) -> np.ndarray:
        """Calculate temperature-dependent efficiency."""
        return self.base_efficiency * (1 - self.temp_coefficient * (cell_temp - 25))
        
    def generate_output(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate PV system output parameters."""
        # Calculate irradiance
        irradiance = self.calculate_irradiance(df)
        
        # Calculate cell temperature
        cell_temp = self.calculate_cell_temperature(df['weather_temperature'], irradiance)
        
        # Simulate dust accumulation (cleaned every 30 days)
        days = (df.index - df.index[0]).days
        dust_factor = 1 - 0.002 * (days % 30)  # 0.2% loss per day
        
        # Calculate efficiency
        efficiency = self.calculate_efficiency(cell_temp)
        
        # Calculate power output
        power = (
            self.capacity_kw 
            * irradiance / 1000  # Convert W/m² to ratio
            * efficiency 
            * dust_factor
        )
        
        return {
            'irradiance': irradiance,
            'cell_temp': cell_temp,
            'efficiency': efficiency,
            'power': power
        }
