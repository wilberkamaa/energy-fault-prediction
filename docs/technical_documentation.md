# Hybrid Energy System Synthetic Data Generator

## Overview

This codebase generates synthetic data for a hybrid energy system located in Kenya, consisting of:
- 1500 kW Solar PV System
- 1 MVA Diesel Generator
- 3 MWh Battery Storage
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

## System Components

### 1. Solar PV System
- Capacity: 1500 kW
- High-efficiency panels (23% base efficiency)
- Modern inverters (85% system efficiency)
- Advanced thermal design (NOCT: 42°C)
- Improved cleaning schedule
- Expected to provide 30-50% of daily load
- **Priority Level: 1 (Highest)** - Maximized usage when available

### 2. Battery Energy Storage
- Capacity: 3 MWh (2 hours of solar capacity)
- Maximum power: 750 kW (50% of solar capacity)
- Modern lithium-ion technology
- High round-trip efficiency (95%)
- Low self-discharge (0.05% per hour)
- Extended cycle life
- Usable capacity: 90% of rated
- Dynamic SOC range (15%-95%)
- Balanced charging/discharging cycles
- **Priority Level: 2** - Used after solar to meet demand or store excess solar

### 3. Grid Connection
- Voltage: 25 kV
- Reliability: 98% (base)
- Seasonal reliability factors
  - Long rains: 95%
  - Short rains: 97%
  - Dry season: 99%
- Bi-directional power flow (import/export)
- Peak hour limitations (30% reduction during peak hours)
- **Priority Level: 3** - Used after solar and battery

### 4. Diesel Generator
- Capacity: 1 MVA
- Fuel tank: 2000 liters
- Minimum load: 40% (improved efficiency)
- Minimum runtime: 2 hours once started
- Modern engine with improved efficiency
- Maintenance interval: 750 hours
- **Priority Level: 4 (Lowest)** - Last resort, only used when other sources insufficient

## Power Dispatch Strategy

The system implements a hierarchical power dispatch strategy to optimize renewable energy usage and minimize operational costs:

1. **Solar PV (First Priority)**
   - Always used when available
   - Excess solar production can charge batteries or be exported to grid

2. **Battery Storage (Second Priority)**
   - Charges during solar surplus
   - Discharges to meet demand when solar is insufficient
   - Maintains dynamic SOC range (15%-95%)
   - Follows daily charge/discharge patterns based on time of day

3. **Grid Connection (Third Priority)**
   - Supplies remaining power needs after solar and battery
   - Reduced usage during peak hours (6 PM - 10 PM)
   - Can accept excess power from solar when batteries are full

4. **Diesel Generator (Last Resort)**
   - Only activates when all other sources are insufficient
   - Requires minimum load (40%) for efficient operation
   - Runs for minimum of 2 consecutive hours once started
   - Automatically activates during grid outages with significant load

This dispatch strategy ensures:
- Maximum utilization of renewable energy
- Optimal battery cycling behavior
- Reduced operational costs and emissions
- Reliable power supply under all conditions

## Design Principles

### Power Management Strategy
1. **Solar Priority**
   - Solar PV is the primary power source
   - Sized to meet 30-50% of average daily load
   - Excess solar charges battery

2. **Battery Management**
   - Charges primarily from solar excess
   - Discharges during peak demand or low solar
   - Maintains 10-90% SOC range
   - Smart charge/discharge algorithms

3. **Generator Control**
   - Last resort power source
   - Operates only when:
     * Battery SOC < 20%
     * Load > Solar + Battery + Grid
     * Grid unavailable
   - Maintains minimum 30% loading

4. **Grid Integration**
   - Supplements solar during low generation
   - Absorbs excess solar when battery full
   - Provides frequency regulation
   - Backup during system maintenance

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
- Nominal capacity: 1500 kW
- Base efficiency: 23% at 25°C
- Temperature coefficient: -0.4%/°C
- NOCT (Nominal Operating Cell Temperature): 42°C

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
- Capacity: 3 MWh
- Maximum power: 750 kW
- Minimum state of charge (SOC): 20%
- Charge efficiency: 95%
- Discharge efficiency: 95%
- Self-discharge rate: 0.05% per hour
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
- Capacity: 1 MVA
- Minimum load: 30%
- Fuel consumption curve: Quadratic function
- Maintenance interval: 750 hours

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

### Column Naming Convention
All columns follow a consistent naming pattern:
- Weather data: `weather_*` (e.g., `weather_temperature`, `weather_cloud_cover`)
- Solar PV: `solar_*` (e.g., `solar_power`, `solar_irradiance`)
- Grid: `grid_*` (e.g., `grid_power`, `grid_voltage`)
- Battery: `battery_*` (e.g., `battery_power`, `battery_soc`)
- Generator: `generator_*` (e.g., `generator_power`, `generator_fuel_rate`)
- Load: `load_*` (e.g., `load_demand`, `load_reactive_power`)
- Faults: `fault_*` (e.g., `fault_type`, `fault_severity`)

### Key Parameters
1. **Weather Parameters**
   - `weather_temperature`: Ambient temperature (°C)
   - `weather_cloud_cover`: Cloud cover ratio (0-1)
   - `weather_humidity`: Relative humidity (%)
   - `weather_wind_speed`: Wind speed (m/s)

2. **Solar PV Parameters**
   - `solar_irradiance`: Solar irradiance (W/m²)
   - `solar_cell_temp`: PV cell temperature (°C)
   - `solar_power`: Power output (kW)

3. **Grid Parameters**
   - `grid_voltage`: Grid voltage (V)
   - `grid_frequency`: Grid frequency (Hz)
   - `grid_power`: Power exchange with grid (kW, positive=import)
   - `grid_available`: Grid availability status (bool)
   - `grid_power_quality`: Power quality metric (0-1)

4. **Battery Parameters**
   - `battery_power`: Power exchange (kW, positive=discharge)
   - `battery_soc`: State of charge (%)
   - `battery_voltage`: Battery voltage (V)
   - `battery_temperature`: Battery temperature (°C)

5. **Generator Parameters**
   - `generator_power`: Power output (kW)
   - `generator_fuel_rate`: Fuel consumption rate (L/h)
   - `generator_temperature`: Engine temperature (°C)
   - `generator_runtime`: Cumulative runtime (h)

6. **Load Parameters**
   - `load_demand`: Active power demand (kW)
   - `load_reactive_power`: Reactive power (kVAR)
   - `load_power_factor`: Power factor (0-1)

7. **Fault Parameters**
   - `fault_type`: Type of fault (string)
   - `fault_severity`: Fault severity (0-1)
   - `fault_duration`: Duration in hours (float)

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
