# Energy System Fault Prediction & Demand Forecasting

A comprehensive hybrid energy system analysis tool for demand forecasting and fault prediction using synthetic data and machine learning models.

## Project Overview

This project focuses on two main aspects of hybrid energy systems:
1. **Demand Forecasting**: Predicting energy demand patterns using time series analysis and machine learning
2. **Fault Detection**: Early detection and classification of system faults using deep learning techniques

The system uses synthetic data that simulates a hybrid energy setup with solar PV, battery storage, diesel generator, and grid connection components. This data is used to train and evaluate various machine learning models.

## Repository Structure

```
energy-fault-prediction/
├── data/                  # Synthetic datasets and raw data
├── dashboards/            # Streamlit dashboards
│   ├── demand_forecast/   # Demand forecasting visualization dashboard
│   └── fault_detection/   # Fault detection and analysis dashboard
├── docs/                  # Documentation and technical specifications
├── fault_analysis/        # Fault analysis and prediction modules
├── kaggle_integration/    # Scripts for Kaggle model training
├── models/                # Model definitions and saved trained models
│   └── trained/           # Pre-trained model files from Kaggle
├── notebooks/             # Jupyter notebooks for analysis and exploration
├── output/                # Generated visualizations and analysis results
├── src/                   # Source code for data generation and simulation
│   ├── battery_system.py  # Battery system simulation
│   ├── data_generator.py  # Main data generation module
│   ├── diesel_generator.py # Diesel generator simulation
│   ├── fault_injection.py # Fault simulation module
│   ├── grid_connection.py # Grid connection simulation
│   ├── load_profile.py    # Load demand profile generation
│   ├── solar_pv.py        # Solar PV system simulation
│   ├── validation.py      # Data validation utilities
│   └── weather.py         # Weather simulation module
└── tests/                 # Unit tests
```

## Key Features

- **Synthetic Data Generation**: Creates realistic time-series data for hybrid energy systems
- **Power Dispatch Strategy**: Implements hierarchical power dispatch with renewable energy prioritization
- **Fault Injection**: Simulates various system faults with configurable severity and duration
- **Machine Learning Models**: Includes models for demand forecasting and fault prediction
- **Deep Learning Integration**: Uses deep learning for complex fault pattern recognition
- **Interactive Dashboards**: Provides Streamlit dashboards for visualization and analysis
- **Kaggle Integration**: Supports training models on Kaggle and importing trained models

## Setup Instructions

### Local Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/energy-fault-prediction.git
   cd energy-fault-prediction
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Generate synthetic data:
   ```bash
   python -c "from src.data_generator import HybridSystemDataGenerator; gen = HybridSystemDataGenerator(); gen.generate_dataset(output_file='data/synthetic_data.csv')"
   ```

### Kaggle Integration for Model Training

1. Set up Kaggle API credentials:
   - Download your Kaggle API token from your Kaggle account settings
   - Place `kaggle.json` in the `~/.kaggle/` directory

2. Use the provided scripts to train models on Kaggle:
   ```bash
   python kaggle_integration/prepare_dataset.py
   python kaggle_integration/train_model.py
   ```

3. Import trained models:
   ```bash
   python kaggle_integration/import_model.py
   ```

### Running the Dashboards

1. Start the demand forecasting dashboard:
   ```bash
   streamlit run dashboards/demand_forecast/app.py
   ```

2. Start the fault detection dashboard:
   ```bash
   streamlit run dashboards/fault_detection/app.py
   ```

## Model Training Workflow

1. Generate synthetic data locally
2. Upload data to Kaggle using the integration scripts
3. Train models on Kaggle using the provided notebooks
4. Download trained model files
5. Use the models for prediction and analysis locally

## Dashboards

### Demand Forecasting Dashboard
- Historical demand visualization
- Forecasting model performance metrics
- Interactive prediction for different time horizons
- Factor analysis for demand drivers

### Fault Detection Dashboard
- Real-time system monitoring visualization
- Fault prediction with confidence scores
- Historical fault analysis
- Early warning system with configurable thresholds

## Contributing

This project is part of a final year academic project. Contributions, suggestions, and feedback are welcome.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
