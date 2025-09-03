# Hybrid Energy System Synthetic Data Generator

## Overview

This codebase generates synthetic data for a hybrid energy system located in Kenya. The system is designed as an off-grid or microgrid solution consisting of three main components:

- A Solar PV System
- A Diesel Generator
- A Battery Storage System

The data generator creates realistic time series data with weather patterns, load profiles, and system faults. All parameters for the simulation are defined in the centralized configuration file at `src/config.py`, making the system highly modular and configurable.

## Project Structure

```
.
├── dashboards
├── data
├── docs
│   └── technical_documentation.md
├── environment.yml
├── fault_analysis
├── kaggle_integration
├── notebooks
├── output
├── src
│   ├── __init__.py
│   ├── battery_system.py
│   ├── config.py
│   ├── data_generator.py
│   └── ... (component modules)
└── tests
    ├── e2e
    ├── integration
    ├── property
    └── unit
```

## Installation & Setup

### Environment Setup
The project uses `conda` for environment management to ensure consistency.

```bash
# Create the conda environment from the environment.yml file
conda env create -f environment.yml

# Activate the new environment
conda activate energy-fault
```

## System Components

The system consists of three core components whose behaviors and parameters are defined in `src/config.py`.

### 1. Solar PV System
- **Description:** Simulates the power output from a solar photovoltaic array, considering factors like solar irradiance, cloud cover, and cell temperature.
- **Configurable Parameters:** `capacity_kw`, `nominal_efficiency`, `temp_coefficient`, `noct`, etc.
- **Priority Level: 1 (Highest)** - Maximized usage when available.
- *Note: All parameters are set in `src/config.py` under the `solar_pv` key.*

### 2. Battery Energy Storage
- **Description:** Models a battery system that can store excess energy and discharge to meet demand. The simulation includes charging/discharging efficiency, state of charge (SOC) limits, and degradation.
- **Configurable Parameters:** `capacity_kwh`, `max_power_kw`, `min_soc`, `max_soc`, `charging_efficiency`, etc.
- **Priority Level: 2** - Used after solar to meet demand or store excess solar generation.
- *Note: All parameters are set in `src/config.py` under the `battery` key.*

### 3. Diesel Generator
- **Description:** Simulates a diesel generator as a backup power source. The model includes fuel consumption based on load, minimum runtime constraints, and maintenance intervals.
- **Configurable Parameters:** `capacity_kva`, `fuel_tank_capacity`, `min_load_percent`, `fuel_consumption_rate`, etc.
- **Priority Level: 3 (Lowest)** - Last resort, used only when solar and battery are insufficient.
- *Note: All parameters are set in `src/config.py` under the `diesel_generator` key.*

## Power Dispatch Strategy

The system implements a hierarchical power dispatch strategy to optimize renewable energy usage:

1.  **Solar PV (First Priority)**
    - Always used when available. Excess solar production is used to charge the battery.

2.  **Battery Storage (Second Priority)**
    - Discharges to meet energy demand when solar generation is insufficient.

3.  **Diesel Generator (Last Resort)**
    - Activates only when both solar and battery cannot meet the load demand.

## Dataset Generation Methodology

The generation process is a sequence of simulations, with all numerical parameters sourced from `src/config.py`.

### 1. Weather Simulation
- Simulates ambient temperature, cloud cover, humidity, and wind speed based on daily and seasonal patterns typical for the region.

### 2. Solar PV Generation
- Calculates power output based on the simulated weather (irradiance, temperature) and the PV system's configured parameters.

### 3. Load Profile Generation
- Generates a realistic load demand profile considering time of day (peaks/valleys), day of the week, holidays, and seasonal factors.

### 4. Battery System
- Simulates the battery's response to the power surplus or deficit after accounting for solar generation and load demand.

### 5. Diesel Generator
- Simulates the generator's activation to cover any remaining load not met by solar and battery.

### 6. Fault Injection
- Injects faults into the system based on predefined probabilities and system state conditions (e.g., high temperature, low SOC).
- **Configurable Faults:** `LINE_SHORT_CIRCUIT`, `INVERTER_IGBT_FAILURE`, `GENERATOR_FIELD_FAILURE`, `BATTERY_OVERDISCHARGE`, etc.

### 7. Data Validation
- **Range Validation:** Clips all generated data to fall within realistic physical ranges.
- **Power Balance:** Ensures that `Total Generation ≈ Load Demand` at each timestep.

## Data Structure

- **Column Naming:** Follows a `component_*` convention (e.g., `solar_power`, `battery_soc`).
- **Key Parameters:** The dataset includes time series data for weather, load, component status, and fault conditions.

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
1.  Vectorized operations using NumPy for performance.
2.  Efficient data storage using the Parquet format.
3.  Modular, component-based design.
4.  All simulation parameters are centralized in `src/config.py` for easy configuration.
