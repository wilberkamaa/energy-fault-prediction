import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from .weather import WeatherSimulator
from .solar_pv import SolarPVSimulator
from .diesel_generator import DieselGeneratorSimulator
from .battery_system import BatterySystemSimulator
from .grid_connection import GridConnectionSimulator
from .load_profile import LoadProfileGenerator
from .fault_injection import FaultInjectionSystem
from .validation import DataValidator

class HybridSystemDataGenerator:
    """Main class for generating synthetic data for the hybrid energy system."""
    
    def __init__(self, seed: int = 42):
        """Initialize all system components."""
        self.seed = seed
        np.random.seed(seed)
        
        # Initialize components
        self.weather_sim = WeatherSimulator(seed=seed)
        self.solar_sim = SolarPVSimulator(seed=seed)
        self.generator_sim = DieselGeneratorSimulator(seed=seed)
        self.battery_sim = BatterySystemSimulator(seed=seed)
        self.grid_sim = GridConnectionSimulator(seed=seed)
        self.load_gen = LoadProfileGenerator(seed=seed)
        self.fault_sim = FaultInjectionSystem(seed=seed)
        self.validator = DataValidator()
        
    def generate_dataset(self, 
                        start_date: str = '2023-01-01',
                        periods_years: int = 2,
                        output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Generate a complete dataset for the specified time period.
        
        Args:
            start_date: Start date for the dataset
            periods_years: Number of years to simulate
            output_file: Optional path to save the dataset
            
        Returns:
            DataFrame containing all system parameters and fault labels
        """
        print("Generating time series base...")
        # Generate time series
        hours = periods_years * 365 * 24
        dates = pd.date_range(start=start_date, periods=hours, freq='H')
        df = pd.DataFrame(index=dates)
        
        # Add temporal features
        df['hour'] = df.index.hour
        df['day_of_year'] = df.index.dayofyear
        df['is_weekend'] = df.index.weekday >= 5
        
        # Add cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Add Kenyan seasons
        def get_season(date):
            month = date.month
            if 3 <= month <= 5:
                return 'long_rains'
            elif 10 <= month <= 12:
                return 'short_rains'
            else:
                return 'dry'
        
        df['season'] = df.index.map(get_season)
        
        print("Generating weather conditions...")
        # Generate weather conditions FIRST
        weather_data = self.weather_sim.generate_weather(df)
        for key, value in weather_data.items():
            df[f'weather_{key}'] = value
        
        print("Generating load profile...")
        # Generate load profile
        load_data = self.load_gen.generate_load(df)
        for key, value in load_data.items():
            df[f'load_{key}'] = value
        
        print("Simulating solar PV system...")
        # Generate solar PV output (now has access to weather data)
        solar_data = self.solar_sim.generate_output(df)
        for key, value in solar_data.items():
            df[f'solar_{key}'] = value
        
        print("Simulating grid connection...")
        # Generate grid parameters
        grid_data = self.grid_sim.generate_output(df)
        for key, value in grid_data.items():
            df[f'grid_{key}'] = value
        
        print("Simulating battery system...")
        # Generate battery parameters
        battery_data = self.battery_sim.generate_output(
            df,
            solar_data['power_output'],
            load_data['active_power']
        )
        for key, value in battery_data.items():
            df[f'battery_{key}'] = value
        
        print("Simulating diesel generator...")
        # Generate generator parameters
        generator_data = self.generator_sim.generate_output(
            df,
            load_data['active_power'],
            solar_data['power_output'],
            battery_data['power_output']
        )
        for key, value in generator_data.items():
            df[f'generator_{key}'] = value
        
        print("Injecting faults...")
        # Generate fault events
        system_state = {
            'grid_voltage': df['grid_voltage'],
            'inverter_temp': df['solar_cell_temperature'],
            'generator_runtime': df['generator_running_hours'],
            'battery_soc': df['battery_soc']
        }
        fault_data = self.fault_sim.generate_fault_events(df, system_state)
        for key, value in fault_data.items():
            df[f'fault_{key}'] = value
        
        print("Validating data...")
        # Validate and clean data
        data_dict = {
            'solar': {k.replace('solar_', ''): v for k, v in df.items() if k.startswith('solar_')},
            'battery': {k.replace('battery_', ''): v for k, v in df.items() if k.startswith('battery_')},
            'generator': {k.replace('generator_', ''): v for k, v in df.items() if k.startswith('generator_')},
            'grid': {k.replace('grid_', ''): v for k, v in df.items() if k.startswith('grid_')},
            'load': {k.replace('load_', ''): v for k, v in df.items() if k.startswith('load_')}
        }
        
        cleaned_data = self.validator.validate_and_clip(data_dict)
        
        # Update DataFrame with cleaned data
        for component, params in cleaned_data.items():
            for param, values in params.items():
                df[f'{component}_{param}'] = values
        
        # Add power balance check
        df['system_balanced'] = self.validator.check_power_balance(data_dict)
        
        if output_file:
            # Create directory if it doesn't exist
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            
            # Save to parquet format
            df.to_parquet(output_file, compression='snappy')
            print(f"Dataset saved to {output_file}")
        
        return df
