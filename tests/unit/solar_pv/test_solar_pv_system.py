import pytest
import numpy as np
import pandas as pd
from src.solar_pv import SolarPVSimulator
from src.config import config

class TestSolarPVInitialization:
    """Test the initialization of the solar PV system."""
    
    def test_initialization_with_default_seed(self):
        """Test that solar PV simulator initializes correctly with default seed."""
        solar_pv = SolarPVSimulator()
        
        # Verify configuration values were correctly loaded
        assert solar_pv.capacity_kw == config['solar_pv']['capacity_kw']
        assert solar_pv.nominal_efficiency == config['solar_pv']['nominal_efficiency']
        assert solar_pv.temp_coefficient == config['solar_pv']['temp_coefficient']
        assert solar_pv.dust_loss_rate == config['solar_pv']['dust_loss_rate']
        assert solar_pv.noct == config['solar_pv']['noct']
        assert solar_pv.base_efficiency == config['solar_pv']['base_efficiency']
        assert solar_pv.system_efficiency == config['solar_pv']['system_efficiency']
        assert solar_pv.rated_power == solar_pv.capacity_kw

    def test_initialization_with_custom_seed(self):
        """Test that solar PV simulator initializes correctly with custom seed."""
        solar_pv = SolarPVSimulator(seed=123)
        
        # Verify configuration values were correctly loaded
        assert solar_pv.capacity_kw == config['solar_pv']['capacity_kw']
        assert solar_pv.system_efficiency == config['solar_pv']['system_efficiency']


class TestSolarPVIrradianceCalculation:
    """Test the irradiance calculation functionality of the solar PV system."""
    
    def test_irradiance_calculation(self):
        """Test irradiance calculation based on time and weather."""
        solar_pv = SolarPVSimulator(seed=42)
        
        # Create a test dataframe with required columns
        index = pd.date_range('2023-01-01', periods=24, freq='H')
        df = pd.DataFrame(index=index)
        df['weather_cloud_cover'] = 0.2  # 20% cloud cover
        df['weather_day_of_year'] = df.index.dayofyear
        
        # Calculate irradiance
        irradiance = solar_pv.calculate_irradiance(df)
        
        # Verify irradiance is calculated correctly
        assert len(irradiance) == len(df)
        assert isinstance(irradiance, np.ndarray)
        
        # Verify daytime hours have positive irradiance
        daytime_hours = (df.index.hour >= 6) & (df.index.hour <= 18)
        assert all(irradiance[daytime_hours] > 0)
        
        # Verify nighttime hours have zero irradiance
        nighttime_hours = ~daytime_hours
        assert all(irradiance[nighttime_hours] == 0)


class TestSolarPVCellTemperature:
    """Test the cell temperature calculation of the solar PV system."""
    
    def test_cell_temperature_calculation(self):
        """Test cell temperature calculation based on ambient temperature and irradiance."""
        solar_pv = SolarPVSimulator(seed=42)
        
        # Test with various ambient temperatures and irradiance levels
        test_cases = [
            (25, 1000),  # Standard test conditions
            (35, 800),   # Hot day, good irradiance
            (15, 400),   # Cool day, moderate irradiance
            (40, 1200),  # Very hot day, high irradiance
            (10, 0)      # Cold day, no irradiance
        ]
        
        for ambient_temp, irradiance in test_cases:
            cell_temp = solar_pv.calculate_cell_temperature(ambient_temp, irradiance)
            
            # Cell temperature should be higher than ambient when there's irradiance
            if irradiance > 0:
                assert cell_temp > ambient_temp
            else:
                assert cell_temp == ambient_temp
            
            # Verify the calculation formula
            expected_temp = ambient_temp + (solar_pv.noct - 20) * irradiance / 800
            assert cell_temp == pytest.approx(expected_temp, rel=1e-10)


class TestSolarPVPowerCalculation:
    """Test the power calculation functionality of the solar PV system."""
    
    def test_power_calculation(self):
        """Test power calculation based on irradiance and cell temperature."""
        solar_pv = SolarPVSimulator(seed=42)
        
        # Test with various irradiance and cell temperature combinations
        test_cases = [
            (1000, 25),  # Standard test conditions
            (800, 30),   # Good irradiance, elevated temperature
            (400, 20),   # Moderate irradiance, cooler temperature
            (1200, 45),  # High irradiance, very hot
            (0, 15)      # No irradiance, cold
        ]
        
        for irradiance, cell_temp in test_cases:
            power = solar_pv.calculate_power(irradiance, cell_temp)
            
            # Power should be non-negative and not exceed rated power
            assert power >= 0
            assert power <= solar_pv.rated_power
            
            # Verify calculation logic
            temp_factor = 1 + solar_pv.temp_coefficient * (cell_temp - 25)
            expected_power = (
                solar_pv.rated_power * 
                (irradiance / 1000) * 
                temp_factor * 
                solar_pv.system_efficiency
            )
            expected_power = np.clip(expected_power, 0, solar_pv.rated_power)
            
            assert power == pytest.approx(expected_power, rel=1e-10)
            
            # Verify temperature coefficient effect
            if cell_temp > 25 and irradiance > 0:
                # Higher temperature should reduce power (negative coefficient)
                assert temp_factor < 1
            elif cell_temp < 25 and irradiance > 0:
                # Lower temperature should increase power
                assert temp_factor > 1


class TestSolarPVOutputGeneration:
    """Test the output generation functionality of the solar PV system."""
    
    def test_generate_output(self):
        """Test generation of PV system output parameters."""
        solar_pv = SolarPVSimulator(seed=42)
        
        # Create a test dataframe with required columns
        index = pd.date_range('2023-01-01', periods=24, freq='H')
        df = pd.DataFrame(index=index)
        df['weather_cloud_cover'] = 0.2  # 20% cloud cover
        df['weather_day_of_year'] = df.index.dayofyear
        df['weather_temperature'] = 25  # 25°C ambient temperature
        
        # Generate output
        output = solar_pv.generate_output(df)
        
        # Verify output structure
        assert isinstance(output, dict)
        assert 'irradiance' in output
        assert 'cell_temp' in output
        assert 'power' in output
        
        # Verify output data types and lengths
        assert isinstance(output['irradiance'], np.ndarray)
        assert isinstance(output['cell_temp'], np.ndarray)
        assert isinstance(output['power'], np.ndarray)
        assert len(output['irradiance']) == len(df)
        assert len(output['cell_temp']) == len(df)
        assert len(output['power']) == len(df)
        
        # Verify relationships between outputs
        # Power should be zero when irradiance is zero
        zero_irradiance = output['irradiance'] == 0
        assert all(output['power'][zero_irradiance] == 0)
        
        # Cell temperature should be higher when irradiance is higher
        # (This is a general trend, not an exact relationship)
        high_irradiance = output['irradiance'] > 500
        if any(high_irradiance):
            assert np.mean(output['cell_temp'][high_irradiance]) > np.mean(df['weather_temperature'])