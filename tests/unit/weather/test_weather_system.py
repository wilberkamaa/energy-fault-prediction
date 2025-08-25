import pytest
import numpy as np
import pandas as pd
from src.weather import WeatherSimulator
from src.config import config

class TestWeatherInitialization:
    """Test the initialization of the weather simulator."""
    
    def test_initialization_with_default_seed(self):
        """Test that weather simulator initializes correctly with default seed."""
        weather = WeatherSimulator()
        
        # Verify configuration values were correctly loaded
        assert weather.season_params == config['weather']['season_params']
        assert weather.base_temperature == config['weather']['base_temperature']
        assert weather.temperature_amplitude == config['weather']['temperature_amplitude']
        assert weather.temperature_peak_hour == config['weather']['temperature_peak_hour']
    
    def test_initialization_with_custom_seed(self):
        """Test that weather simulator initializes correctly with custom seed."""
        custom_seed = 123
        weather = WeatherSimulator(seed=custom_seed)
        
        # Verify configuration values were correctly loaded
        assert weather.season_params == config['weather']['season_params']
        assert weather.base_temperature == config['weather']['base_temperature']

class TestWeatherGeneration:
    """Test the weather generation functionality."""
    
    def setup_method(self):
        """Set up test data."""
        # Create a simple dataframe with required columns
        self.df = pd.DataFrame({
            'weather_hour': [0, 6, 12, 18],
            'weather_season': ['dry', 'long_rains', 'short_rains', 'dry']
        })
    
    def test_weather_output_format(self):
        """Test that the weather output has the correct format."""
        weather = WeatherSimulator(seed=42)
        output = weather.generate_weather(self.df)
        
        # Check that all expected keys are present
        expected_keys = ['temperature', 'cloud_cover', 'humidity', 'wind_speed']
        for key in expected_keys:
            assert key in output
        
        # Check that output arrays have the correct length
        for key in expected_keys:
            assert len(output[key]) == len(self.df)
    
    def test_temperature_generation(self):
        """Test temperature generation based on hour and season."""
        weather = WeatherSimulator(seed=42)
        output = weather.generate_weather(self.df)
        
        # Check that temperature values are within expected ranges
        for i, season in enumerate(self.df['weather_season']):
            temp_range = config['weather']['season_params'][season]['temp_range']
            # Allow for some variation due to random noise
            assert output['temperature'][i] >= temp_range[0] - 5
            assert output['temperature'][i] <= temp_range[1] + 5
        
        # Check that temperature follows daily cycle (higher at peak hours)
        noon_index = self.df[self.df['weather_hour'] == 12].index[0]
        midnight_index = self.df[self.df['weather_hour'] == 0].index[0]
        # Assuming same season, noon should be warmer than midnight
        if self.df.loc[noon_index, 'weather_season'] == self.df.loc[midnight_index, 'weather_season']:
            assert output['temperature'][noon_index] > output['temperature'][midnight_index]
    
    def test_cloud_cover_generation(self):
        """Test cloud cover generation based on season."""
        weather = WeatherSimulator(seed=42)
        output = weather.generate_weather(self.df)
        
        # Check that cloud cover values are within expected range (0-1)
        assert np.all(output['cloud_cover'] >= 0)
        assert np.all(output['cloud_cover'] <= 1)
        
        # Check that cloud cover follows seasonal patterns
        for i, season in enumerate(self.df['weather_season']):
            cloud_range = config['weather']['season_params'][season]['cloud_cover']
            # Allow for some variation due to time of day factor
            assert output['cloud_cover'][i] >= cloud_range[0] - 0.3
            assert output['cloud_cover'][i] <= cloud_range[1] + 0.3
    
    def test_humidity_generation(self):
        """Test humidity generation."""
        weather = WeatherSimulator(seed=42)
        output = weather.generate_weather(self.df)
        
        # Check that humidity values are within expected range
        humidity_min = config['weather']['humidity_min']
        humidity_max = config['weather']['humidity_max']
        assert np.all(output['humidity'] >= humidity_min)
        assert np.all(output['humidity'] <= humidity_max)
        
        # Check correlation between cloud cover and humidity
        # Higher cloud cover should generally mean higher humidity
        cloud_cover = output['cloud_cover']
        humidity = output['humidity']
        if len(cloud_cover) > 1:  # Only test if we have enough data points
            correlation = np.corrcoef(cloud_cover, humidity)[0, 1]
            assert correlation > 0  # Positive correlation expected
    
    def test_wind_speed_generation(self):
        """Test wind speed generation."""
        weather = WeatherSimulator(seed=42)
        output = weather.generate_weather(self.df)
        
        # Check that wind speed values are positive
        assert np.all(output['wind_speed'] >= 0)
        
        # Check that wind speed values are within reasonable range for Kenya
        # (typically 0-10 m/s, but allowing for occasional stronger winds)
        assert np.all(output['wind_speed'] <= 20)

class TestWeatherSeasonality:
    """Test the seasonal variations in weather generation."""
    
    def setup_method(self):
        """Set up test data for different seasons."""
        # Create dataframes for different seasons (same hour to isolate seasonal effect)
        self.hour = 12  # Noon
        self.dry_df = pd.DataFrame({
            'weather_hour': [self.hour] * 10,
            'weather_season': ['dry'] * 10
        })
        self.long_rains_df = pd.DataFrame({
            'weather_hour': [self.hour] * 10,
            'weather_season': ['long_rains'] * 10
        })
        self.short_rains_df = pd.DataFrame({
            'weather_hour': [self.hour] * 10,
            'weather_season': ['short_rains'] * 10
        })
    
    def test_seasonal_temperature_differences(self):
        """Test that temperature varies by season."""
        weather = WeatherSimulator(seed=42)
        
        dry_output = weather.generate_weather(self.dry_df)
        long_rains_output = weather.generate_weather(self.long_rains_df)
        
        # Dry season should be warmer than long rains season
        assert np.mean(dry_output['temperature']) > np.mean(long_rains_output['temperature'])
    
    def test_seasonal_cloud_cover_differences(self):
        """Test that cloud cover varies by season."""
        weather = WeatherSimulator(seed=42)
        
        dry_output = weather.generate_weather(self.dry_df)
        long_rains_output = weather.generate_weather(self.long_rains_df)
        
        # Long rains season should have more cloud cover than dry season
        assert np.mean(long_rains_output['cloud_cover']) > np.mean(dry_output['cloud_cover'])