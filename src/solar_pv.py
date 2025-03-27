import numpy as np
from typing import Dict, Any

class SolarPVSimulator:
    """Simulates a 500 kW Solar PV system."""
    
    def __init__(self, capacity_kw: float = 500, seed: int = 42):
        self.capacity_kw = capacity_kw
        np.random.seed(seed)
        
        # System parameters
        self.nominal_efficiency = 0.15
        self.temp_coefficient = -0.004  # Efficiency loss per degree C above 25°C
        self.dust_loss_rate = 0.001    # Daily loss rate due to dust
        
    def calculate_irradiance(self, df) -> np.ndarray:
        """Calculate solar irradiance considering time and weather."""
        # Base irradiance pattern (clear sky)
        hour_shifted = (df['hour'] - 6) % 24  # Shift to start at 6 AM
        base_irradiance = 1000 * np.abs(np.sin(np.pi * hour_shifted / 12))
        
        # Seasonal variation (Kenya's position near equator)
        seasonal_factor = 1 - 0.3 * np.sin(2 * np.pi * (df['day_of_year'] + 81) / 365)
        
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
        # NOCT method (Nominal Operating Cell Temperature)
        return ambient_temp + (irradiance / 800) * 30  # Simplified model
        
    def calculate_efficiency(self, cell_temp: np.ndarray, 
                           days_since_cleaning: np.ndarray) -> np.ndarray:
        """Calculate overall system efficiency."""
        # Temperature derating
        temp_derating = 1 + self.temp_coefficient * (cell_temp - 25)
        
        # Dust derating
        dust_derating = 1 - self.dust_loss_rate * days_since_cleaning
        
        # Combined efficiency
        efficiency = self.nominal_efficiency * temp_derating * dust_derating
        
        return np.clip(efficiency, 0.05, self.nominal_efficiency)
        
    def generate_output(self, df) -> Dict[str, Any]:
        """Generate PV system output parameters."""
        # Calculate irradiance
        irradiance = self.calculate_irradiance(df)
        
        # Calculate cell temperature
        cell_temp = self.calculate_cell_temperature(df['temperature'], irradiance)
        
        # Simulate dust accumulation (cleaned every 30 days)
        days = (df.index - df.index[0]).days
        days_since_cleaning = days % 30
        
        # Calculate system efficiency
        efficiency = self.calculate_efficiency(cell_temp, days_since_cleaning)
        
        # Calculate power output (kW)
        power_output = (irradiance * self.capacity_kw * efficiency) / 1000
        
        # Add inverter efficiency (96-98%)
        inverter_efficiency = 0.97 + np.random.normal(0, 0.005, len(df))
        power_output *= inverter_efficiency
        
        # DC voltage (V)
        dc_voltage = 480 + 20 * np.sin(2 * np.pi * df['hour'] / 24) + \
                    np.random.normal(0, 2, len(df))
        
        return {
            'irradiance': irradiance,
            'cell_temperature': cell_temp,
            'efficiency': efficiency,
            'power_output': power_output,
            'dc_voltage': dc_voltage
        }
