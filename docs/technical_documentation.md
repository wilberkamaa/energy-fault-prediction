# Hybrid Energy System Synthetic Data Generator

## Overview

This codebase generates synthetic data for a hybrid energy system located in Kenya, consisting of:
- 500 kW Solar PV System
- 2 MVA Diesel Generator
- 1 MWh Battery Storage
- 25 kV Grid Connection

The data generator creates realistic time series data with weather patterns, load profiles, and system faults typical of Kenyan conditions.

## Installation & Setup

### Dependencies
```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Generation Methodology

### 1. Weather Simulation

#### Parameters
- Temperature: Daily cycle with seasonal variations
  - Base: 25°C ± 5°C daily variation
  - Seasonal offset: -2°C (long rains), 0°C (short rains), +2°C (dry)
  
#### Mathematical Model
```python
temp_base = 25 + 5 * sin(2π * (hour - 14) / 24)  # Peak at 2 PM
temp_seasonal = season_offset[current_season]
temp_noise = normal(μ=0, σ=0.5)
temperature = temp_base + temp_seasonal + temp_noise
```

### 2. Solar PV Generation

#### Parameters
- Nominal capacity: 500 kW
- Base efficiency: 18% at 25°C
- Temperature coefficient: -0.4%/°C
- NOCT (Nominal Operating Cell Temperature): 45°C

#### Mathematical Model
```python
# Irradiance calculation
irradiance = base_irradiance * seasonal_factor * (1 - cloud_cover)

# Cell temperature
cell_temp = ambient_temp + (NOCT - 20) * irradiance / 800

# Efficiency
efficiency = base_efficiency * (1 - temp_coefficient * (cell_temp - 25))

# Power output
power = capacity * (irradiance / 1000) * efficiency * dust_factor
```

### 3. Load Profile Generation

#### Parameters
- Base load: 500 kW
- Peak load: 2000 kW
- Time factors:
  - Morning peak (6-9): 1.3×
  - Evening peak (18-22): 1.5×
  - Night valley (23-5): 0.7×
  - Weekend reduction: 0.8×

#### Mathematical Model
```python
base_pattern = base_load + (peak_load - base_load) * (0.5 + 0.5 * sin(π * (hour - 6) / 12))
load = base_pattern * time_factor * seasonal_factor * (1 + random_walk)
```

### 4. Grid Connection

#### Parameters
- Nominal voltage: 25 kV
- Base reliability: 98%
- Voltage variation: ±2%
- Peak hours: 18:00-22:00
- Maintenance schedule: Every 90 days, 8:00-16:00

#### Mathematical Model
```python
# Grid availability
reliability = base_reliability * season_factor * time_factor
is_available = random() < reliability and not is_maintenance

# Voltage calculation
base_voltage = nominal_voltage + 500 * sin(2π * hour / 24)
voltage = base_voltage + normal(0, voltage_variation * nominal_voltage)

# Power calculation
grid_power = load_demand - (solar_power + battery_power + generator_power)
```

Key features:
1. Realistic daily voltage patterns
2. Seasonal reliability variations
3. Scheduled maintenance periods
4. Power quality metrics
5. Automatic power balancing

### 5. Battery System

#### Parameters
- Capacity: 1 MWh
- Maximum power: 200 kW
- Minimum state of charge (SOC): 20%
- Charge efficiency: 95%
- Discharge efficiency: 95%
- Self-discharge rate: 0.1% per hour
- Temperature coefficient: -0.2% capacity per °C above 25°C

#### Mathematical Model
```python
# Temperature effect on capacity
temp_factor = 1 - 0.002 * (temperature - 25)  # -0.2% per °C above 25°C
effective_capacity = nominal_capacity * temp_factor

# Charging
power_charge = min(
    power_balance,
    max_power,
    (1 - soc) * effective_capacity * charge_efficiency
)

# Discharging
power_discharge = min(
    -power_balance,
    max_power,
    (soc - min_soc) * effective_capacity / discharge_efficiency
)

# State of charge update
if charging:
    soc_new = soc + (power * charge_efficiency) / effective_capacity
else:
    soc_new = soc - (power / discharge_efficiency) / effective_capacity

# Temperature model
temp_rise = 0.05 * abs(power)  # Temperature rise proportional to power flow
battery_temp = ambient_temp + temp_rise
```

The battery system maintains consistent power flow direction:
- Negative power = Charging (absorbing power)
- Positive power = Discharging (providing power)

Key features:
1. Temperature-dependent capacity
2. Efficiency losses during charge/discharge
3. Self-discharge over time
4. SOC limits (20% minimum)
5. Power limits based on SOC and capacity

### 6. Diesel Generator

#### Parameters
- Capacity: 2 MVA
- Minimum load: 30%
- Fuel consumption curve: Quadratic function
- Maintenance interval: 500 hours

#### Mathematical Model
```python
fuel_rate = a * power² + b * power + c  # Quadratic consumption curve
efficiency = power_output / (fuel_rate * fuel_energy_density)
```

### 7. Fault Injection

#### Fault Types
1. LINE_SHORT_CIRCUIT
2. LINE_PROLONGED_UNDERVOLTAGE
3. INVERTER_IGBT_FAILURE
4. GENERATOR_FIELD_FAILURE
5. GRID_VOLTAGE_SAG
6. GRID_OUTAGE
7. BATTERY_OVERDISCHARGE

#### Fault Parameters
- Occurrence probability: 0.001-0.005 per hour
- Duration: 1-48 hours depending on type
- Severity: 0.3-1.0 (normalized)
- Condition-based triggers

#### Mathematical Model
```python
# Base probability adjustment
prob = base_prob * condition_factor * severity

# Fault duration
duration = uniform(min_duration, max_duration)

# Fault effects
voltage_drop = 0.8 + 0.2 * severity
efficiency_drop = 0.3 * severity
temperature_rise = 20 * severity
```

Key features:
1. Condition-based fault triggering
2. Multiple concurrent faults possible
3. Severity-based effects
4. Component-specific impacts
5. Time-varying fault durations

### 8. Data Validation

#### Valid Parameter Ranges
- Temperature: -10°C to 45°C
- Humidity: 0-100%
- Cloud cover: 0-100%
- Wind speed: 0-30 m/s
- Solar power: 0-1500 kW
- Battery SOC: 0-100%
- Grid voltage: ±20% nominal
- Load demand: 0-2000 kW

#### Power Balance
```python
total_generation = solar + battery + generator + grid
total_load = demand
is_balanced = abs(total_generation - total_load) <= tolerance
```

Key features:
1. Automated range validation
2. Power balance checking
3. NaN handling
4. Data type verification
5. Consistency enforcement

## Data Structure

### Output Format
The generator produces a pandas DataFrame with hourly resolution containing:

| Category | Features | Description |
|----------|----------|-------------|
| Weather | temperature, humidity, cloud_cover, wind_speed | Ambient conditions |
| Solar | irradiance, cell_temp, power, efficiency | PV system performance |
| Load | demand, power_factor | Consumer demand profile |
| Grid | voltage, frequency, available, power_quality | Grid parameters |
| Battery | soc, power, voltage, temperature | Battery state |
| Generator | power, fuel_consumption, runtime, efficiency | Generator operation |
| Faults | type, severity, duration, component | System faults |

## Example Usage

```python
from src.data_generator import HybridSystemDataGenerator

# Initialize generator
generator = HybridSystemDataGenerator(seed=42)

# Generate 2 years of data
df = generator.generate_dataset(
    start_date='2023-01-01',
    periods_years=2,
    output_file='data/hybrid_system_data.parquet'
)
```

## Code Optimization

1. Vectorized operations using NumPy for performance
2. Efficient data storage using parquet format
3. Modular design for easy maintenance and testing
4. Configurable parameters via class initialization

## Further Enhancements

1. Add more sophisticated fault models
2. Implement machine learning-based anomaly detection
3. Add real-time data generation capabilities
4. Include economic parameters (costs, revenue, etc.)
5. Extend weather patterns with climate change scenarios

## References

1. Kenya Meteorological Department - Weather patterns
2. IEEE 1547 - Grid interconnection standards
3. IEC 61724 - PV system performance monitoring
4. Battery storage system standards (IEC 62619)
