import pytest
import numpy as np
import pandas as pd
from src.load_profile import LoadProfileGenerator
from src.config import config

class TestLoadProfileInitialization:
    """Test the initialization of the load profile generator."""
    
    def test_initialization_with_default_seed(self):
        """Test that load profile generator initializes correctly with default seed."""
        load_profile = LoadProfileGenerator()
        
        # Verify configuration values were correctly loaded
        assert load_profile.base_load_kw == config['load_profile']['base_load_kw']
        assert load_profile.peak_load_kw == config['load_profile']['peak_load_kw']
        assert load_profile.weekday_factors == config['load_profile']['weekday_factors']
        assert load_profile.weekend_reduction == config['load_profile']['weekend_reduction']
        assert load_profile.holidays == config['load_profile']['holidays']
        assert load_profile.seasonal_factors == config['load_profile']['seasonal_factors']
    
    def test_initialization_with_custom_seed(self):
        """Test that load profile generator initializes correctly with custom seed."""
        custom_seed = 123
        load_profile = LoadProfileGenerator(seed=custom_seed)
        
        # Verify configuration values were correctly loaded
        assert load_profile.base_load_kw == config['load_profile']['base_load_kw']
        assert load_profile.peak_load_kw == config['load_profile']['peak_load_kw']

class TestHolidayChecking:
    """Test the holiday checking functionality."""
    
    def test_is_holiday_on_holiday(self):
        """Test that is_holiday correctly identifies holidays."""
        load_profile = LoadProfileGenerator()
        
        # Test known holidays
        assert load_profile.is_holiday(pd.Timestamp('2023-01-01')) == True  # New Year's Day
        assert load_profile.is_holiday(pd.Timestamp('2023-05-01')) == True  # Labour Day
        assert load_profile.is_holiday(pd.Timestamp('2023-12-25')) == True  # Christmas Day
    
    def test_is_holiday_on_non_holiday(self):
        """Test that is_holiday correctly identifies non-holidays."""
        load_profile = LoadProfileGenerator()
        
        # Test known non-holidays
        assert load_profile.is_holiday(pd.Timestamp('2023-01-02')) == False
        assert load_profile.is_holiday(pd.Timestamp('2023-07-15')) == False
        assert load_profile.is_holiday(pd.Timestamp('2023-11-11')) == False

class TestTimeFactorCalculation:
    """Test the time factor calculation functionality."""
    
    def test_get_time_factor_weekday_morning_peak(self):
        """Test time factor calculation for weekday morning peak hours."""
        load_profile = LoadProfileGenerator()
        
        # Test morning peak hours (6-9)
        for hour in range(6, 9):
            factor = load_profile.get_time_factor(hour, is_weekend=False)
            assert factor == config['load_profile']['weekday_factors']['morning_peak']['factor']
    
    def test_get_time_factor_weekday_evening_peak(self):
        """Test time factor calculation for weekday evening peak hours."""
        load_profile = LoadProfileGenerator()
        
        # Test evening peak hours (18-22)
        for hour in range(18, 22):
            factor = load_profile.get_time_factor(hour, is_weekend=False)
            assert factor == config['load_profile']['weekday_factors']['evening_peak']['factor']
    
    def test_get_time_factor_weekday_night_valley(self):
        """Test time factor calculation for weekday night valley hours."""
        load_profile = LoadProfileGenerator()
        
        # Test night valley hours (23-5)
        for hour in [23, 0, 1, 2, 3, 4]:
            factor = load_profile.get_time_factor(hour, is_weekend=False)
            assert factor == config['load_profile']['weekday_factors']['night_valley']['factor']
    
    def test_get_time_factor_weekday_default(self):
        """Test time factor calculation for weekday default hours."""
        load_profile = LoadProfileGenerator()
        
        # Test default hours (not in any specific period)
        for hour in [10, 11, 12, 13, 14, 15, 16, 17]:
            factor = load_profile.get_time_factor(hour, is_weekend=False)
            assert factor == 1.0
    
    def test_get_time_factor_weekend_active_hours(self):
        """Test time factor calculation for weekend active hours."""
        load_profile = LoadProfileGenerator()
        
        # Test weekend active hours (8-20)
        for hour in range(8, 21):
            factor = load_profile.get_time_factor(hour, is_weekend=True)
            assert factor == 0.9
    
    def test_get_time_factor_weekend_night_hours(self):
        """Test time factor calculation for weekend night hours."""
        load_profile = LoadProfileGenerator()
        
        # Test weekend night hours (0-7, 21-23)
        for hour in list(range(0, 8)) + list(range(21, 24)):
            factor = load_profile.get_time_factor(hour, is_weekend=True)
            assert factor == 0.6

class TestSeasonalFactorCalculation:
    """Test the seasonal factor calculation functionality."""
    
    def test_get_seasonal_factor_known_seasons(self):
        """Test seasonal factor calculation for known seasons."""
        load_profile = LoadProfileGenerator()
        
        # Test known seasons
        assert load_profile.get_seasonal_factor('long_rains') == config['load_profile']['seasonal_factors']['long_rains']
        assert load_profile.get_seasonal_factor('short_rains') == config['load_profile']['seasonal_factors']['short_rains']
        assert load_profile.get_seasonal_factor('dry') == config['load_profile']['seasonal_factors']['dry']
    
    def test_get_seasonal_factor_unknown_season(self):
        """Test seasonal factor calculation for unknown season."""
        load_profile = LoadProfileGenerator()
        
        # Test unknown season
        assert load_profile.get_seasonal_factor('unknown_season') == 1.0

class TestLoadGeneration:
    """Test the load generation functionality."""
    
    def test_generate_load_basic_functionality(self):
        """Test basic functionality of load generation."""
        load_profile = LoadProfileGenerator(seed=42)
        
        # Create a simple test dataframe
        index = pd.date_range('2023-01-01', periods=24, freq='H')
        df = pd.DataFrame(index=index)
        df['weather_season'] = 'dry'
        
        # Generate load
        result = load_profile.generate_load(df)
        
        # Check that the result contains the expected keys
        assert 'demand' in result
        assert 'power_factor' in result
        
        # Check that the arrays have the expected length
        assert len(result['demand']) == len(df)
        assert len(result['power_factor']) == len(df)
        
        # Check that the values are within expected ranges
        assert np.all(result['demand'] >= 0)
        assert np.all(result['demand'] <= config['load_profile']['peak_load_kw'] * 2)  # Allow for some variation
        assert np.all(result['power_factor'] >= config['validation']['valid_ranges']['load_power_factor'][0])
        assert np.all(result['power_factor'] <= config['validation']['valid_ranges']['load_power_factor'][1])
    
    def test_generate_load_weekend_reduction(self):
        """Test that weekend reduction is applied correctly."""
        load_profile = LoadProfileGenerator(seed=42)
        
        # Create test dataframes for weekday and weekend
        weekday_index = pd.date_range('2023-01-02', periods=24, freq='H')  # Monday
        weekend_index = pd.date_range('2023-01-07', periods=24, freq='H')  # Saturday
        
        weekday_df = pd.DataFrame(index=weekday_index)
        weekend_df = pd.DataFrame(index=weekend_index)
        
        weekday_df['weather_season'] = 'dry'
        weekend_df['weather_season'] = 'dry'
        
        # Generate loads
        weekday_result = load_profile.generate_load(weekday_df)
        weekend_result = load_profile.generate_load(weekend_df)
        
        # Calculate average demand
        weekday_avg = np.mean(weekday_result['demand'])
        weekend_avg = np.mean(weekend_result['demand'])
        
        # Weekend should be lower than weekday due to weekend_reduction
        assert weekend_avg < weekday_avg
        
        # The ratio should be approximately the weekend_reduction factor
        # Allow for some variation due to other factors and randomness
        ratio = weekend_avg / weekday_avg
        assert 0.7 <= ratio <= 0.9  # weekend_reduction is 0.8
    
    def test_generate_load_seasonal_variation(self):
        """Test that seasonal factors affect load generation."""
        load_profile = LoadProfileGenerator(seed=42)
        
        # Create test dataframes for different seasons
        index = pd.date_range('2023-01-01', periods=24, freq='H')
        
        dry_df = pd.DataFrame(index=index)
        long_rains_df = pd.DataFrame(index=index)
        
        dry_df['weather_season'] = 'dry'
        long_rains_df['weather_season'] = 'long_rains'
        
        # Generate loads
        dry_result = load_profile.generate_load(dry_df)
        long_rains_result = load_profile.generate_load(long_rains_df)
        
        # Calculate average demand
        dry_avg = np.mean(dry_result['demand'])
        long_rains_avg = np.mean(long_rains_result['demand'])
        
        # Dry season should have higher demand than long rains
        assert dry_avg > long_rains_avg
        
        # The ratio should be approximately the ratio of seasonal factors
        # dry (1.1) / long_rains (0.9) ≈ 1.22
        ratio = dry_avg / long_rains_avg
        assert 1.1 <= ratio <= 1.3