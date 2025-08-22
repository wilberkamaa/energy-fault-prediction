import numpy as np
from typing import Dict, Any
import pandas as pd
from src.config import config

class LoadProfileGenerator:
    """Generates realistic load profiles for a hybrid energy system."""
    
    def __init__(self, seed: int = 42):
        # Load configuration
        load_config = config['load_profile']
        self.base_load_kw = load_config['base_load_kw']
        self.peak_load_kw = load_config['peak_load_kw']
        np.random.seed(seed)
        
        # Load parameters from config
        self.weekday_factors = load_config['weekday_factors']
        self.weekend_reduction = load_config['weekend_reduction']
        self.holidays = load_config['holidays']
        self.seasonal_factors = load_config['seasonal_factors']
        
    def is_holiday(self, date: pd.Timestamp) -> bool:
        """Check if date is a holiday."""
        return (date.month, date.day) in self.holidays
    
    def get_time_factor(self, hour: int, is_weekend: bool) -> float:
        """Calculate load factor based on time of day and week."""
        if is_weekend:
            # Weekend pattern
            if 8 <= hour <= 20:  # Active hours
                return 0.9
            else:  # Night hours
                return 0.6
        else:
            # Weekday pattern
            for period, info in self.weekday_factors.items():
                start, end = info['hours']
                if start <= hour < end:
                    return info['factor']
            return 1.0  # Default factor
    
    def get_seasonal_factor(self, season: str) -> float:
        """Calculate load factor based on season."""
        return self.seasonal_factors.get(season, 1.0)
    
    def generate_load(self, df) -> Dict[str, Any]:
        """Generate load profile with various factors."""
        hours = len(df)
        
        # Initialize arrays
        load_demand = np.zeros(hours)
        power_factor = np.zeros(hours)
        
        # Generate base load with random walk
        random_walk = np.cumsum(np.random.normal(0, 0.02, hours))
        random_walk = (random_walk - random_walk.min()) / (random_walk.max() - random_walk.min())
        
        for i in range(hours):
            current_time = df.index[i]
            hour = current_time.hour
            is_weekend = current_time.weekday() >= 5
            is_holiday = self.is_holiday(current_time)
            season = df['weather_season'][i]  # Use weather_season for consistency
            
            # Calculate various factors
            time_factor = self.get_time_factor(hour, is_weekend or is_holiday)
            seasonal_factor = self.get_seasonal_factor(season)
            
            # Calculate base load
            base_pattern = self.base_load_kw + \
                         (self.peak_load_kw - self.base_load_kw) * \
                         (0.5 + 0.5 * np.sin(np.pi * (hour - 6) / 12))
            
            # Combine all factors
            load = base_pattern * time_factor * seasonal_factor
            
            # Add random variations
            load *= (1 + 0.1 * random_walk[i])
            
            # Apply weekend reduction if applicable
            if is_weekend or is_holiday:
                load *= self.weekend_reduction
            
            load_demand[i] = load
            
            # Generate power factor using config values
            pf_config = config['load_profile']['power_factor']
            base_pf = pf_config['base'] + pf_config['variation'] * np.sin(2 * np.pi * hour / 24)
            power_factor[i] = base_pf + np.random.normal(0, pf_config['noise_std'])
        
        # Clip power factor to realistic range
        power_factor = np.clip(power_factor, 
                              config['load_profile']['power_factor']['min'],
                              config['load_profile']['power_factor']['max'])
        
        return {
            'demand': load_demand,
            'power_factor': power_factor
        }
