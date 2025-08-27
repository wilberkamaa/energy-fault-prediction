import numpy as np
from typing import Dict, Any
from src.config import config

class WeatherSimulator:
    """Simulates weather conditions for Kenya's climate."""
    
    def __init__(self, seed: int = None):
        if seed is None:
            seed = config['weather']['seed']
        np.random.seed(seed)
        
        # Get seasonal parameters from config
        self.season_params = config['weather']['season_params']
        self.base_temperature = config['weather']['base_temperature']
        self.temperature_amplitude = config['weather']['temperature_amplitude']
        self.temperature_peak_hour = config['weather']['temperature_peak_hour']  # Add this line

    def generate_weather(self, df) -> Dict[str, Any]:
        """
        Generate weather conditions based on time and season.
        
        Args:
            df: DataFrame with datetime index and 'weather_season' column
            
        Returns:
            Dictionary containing weather parameters
        """
        weather_config = config['weather']
        
        # Base temperature pattern (daily cycle)
        base_temp = weather_config['base_temperature']
        temp_amplitude = weather_config['temperature_amplitude']
        peak_hour = weather_config['temperature_peak_hour']
        temp_base = base_temp + temp_amplitude * np.sin(2 * np.pi * (df['weather_hour'] - peak_hour) / 24)
        
        # Add seasonal variation
        season_temp_offset = weather_config['season_temp_offset']
        temp_seasonal = df['weather_season'].map(season_temp_offset)
        
        # Add random variations
        temp_noise = np.random.normal(0, weather_config['temperature_noise_std'], len(df))
        temperature = temp_base + temp_seasonal + temp_noise
        
        # Generate cloud cover based on season and time of day
        cloud_cover = np.zeros(len(df))
        for i in range(len(df)):
            season = df['weather_season'].iloc[i]
            base_prob = np.random.uniform(*self.season_params[season]['cloud_cover'])
            # More clouds in early morning and late afternoon
            hour = df['weather_hour'].iloc[i]
            cloud_hour_amplitude = weather_config['cloud_hour_amplitude']
            cloud_peak_hour = weather_config['cloud_peak_hour']
            hour_factor = cloud_hour_amplitude * np.sin(2 * np.pi * (hour - cloud_peak_hour) / 12)
            cloud_cover[i] = np.clip(base_prob + hour_factor, 0, 1)
        
        # Generate humidity
        humidity_base = weather_config['humidity_base']
        humidity_amplitude = weather_config['humidity_amplitude']
        humidity_base = humidity_base + humidity_amplitude * np.sin(2 * np.pi * df['weather_hour'] / 24)
        humidity_cloud_factor = weather_config['humidity_cloud_factor']
        humidity_noise_std = weather_config['humidity_noise_std']
        humidity = humidity_base + humidity_cloud_factor * cloud_cover + np.random.normal(0, humidity_noise_std, len(df))
        humidity = np.clip(humidity, weather_config['humidity_min'], weather_config['humidity_max'])
        
        # Generate wind speed (m/s)
        wind_shape = weather_config['wind_weibull_shape']
        wind_scale = weather_config['wind_weibull_scale']
        wind_speed = np.random.weibull(wind_shape, len(df)) * wind_scale
        
        return {
            'temperature': temperature,
            'cloud_cover': cloud_cover,
            'humidity': humidity,
            'wind_speed': wind_speed
        }
