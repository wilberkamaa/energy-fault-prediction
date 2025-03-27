"""
Prepare and upload datasets to Kaggle for model training.

This script:
1. Generates synthetic data using the HybridSystemDataGenerator
2. Prepares the data for Kaggle (feature engineering, splitting, etc.)
3. Creates a Kaggle dataset and uploads the prepared data
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import kaggle
import json
from datetime import datetime

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import project modules
from src.data_generator import HybridSystemDataGenerator
from fault_analysis.data_preparation import (
    create_time_features, create_lag_features, create_window_features
)

def generate_synthetic_data(output_file, periods_years=1):
    """Generate synthetic data for training."""
    print(f"Generating synthetic data ({periods_years} years)...")
    
    # Create data generator
    data_gen = HybridSystemDataGenerator(seed=42)
    
    # Generate dataset
    df = data_gen.generate_dataset(
        start_date='2023-01-01',
        periods_years=periods_years,
        output_file=output_file
    )
    
    print(f"Generated dataset with shape: {df.shape}")
    return df

def prepare_data_for_kaggle(df, output_dir):
    """Prepare data for Kaggle training by creating features."""
    print("Preparing data for Kaggle...")
    
    # Create time features
    df = create_time_features(df)
    
    # Select columns for feature engineering
    system_cols = [
        'solar_power', 'solar_cell_temp', 'solar_efficiency',
        'battery_power', 'battery_soc', 'battery_temperature',
        'grid_power', 'grid_voltage', 'grid_frequency',
        'generator_power', 'generator_temperature', 'generator_fuel_level',
        'load_demand', 'weather_temperature', 'weather_humidity'
    ]
    
    # Create lag and window features
    print("Creating lag features...")
    df = create_lag_features(df, system_cols, lag_periods=[1, 3, 6, 12, 24])
    
    print("Creating window features...")
    df = create_window_features(df, system_cols, window_sizes=[6, 12, 24])
    
    # Split data into train/validation/test
    train_end = int(len(df) * 0.7)
    val_end = int(len(df) * 0.85)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    # Save datasets
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(f"{output_dir}/train.csv")
    val_df.to_csv(f"{output_dir}/validation.csv")
    test_df.to_csv(f"{output_dir}/test.csv")
    
    # Create a metadata file
    metadata = {
        "dataset_info": {
            "generated_date": datetime.now().strftime("%Y-%m-%d"),
            "train_shape": train_df.shape,
            "validation_shape": val_df.shape,
            "test_shape": test_df.shape,
            "features": list(train_df.columns),
            "target_columns": ["fault_type", "fault_severity", "load_demand"]
        }
    }
    
    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Prepared datasets saved to {output_dir}")
    return train_df, val_df, test_df

def upload_to_kaggle(dataset_dir, dataset_name, dataset_description):
    """Upload the prepared dataset to Kaggle."""
    try:
        # Create dataset metadata
        metadata = {
            "title": dataset_name,
            "id": f"yourusername/{dataset_name}",
            "licenses": [{"name": "CC0-1.0"}],
            "description": dataset_description
        }
        
        # Write metadata file
        with open(f"{dataset_dir}/dataset-metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Upload to Kaggle
        kaggle.api.dataset_create_new(
            folder=dataset_dir,
            public=False,
            dir_mode="zip"
        )
        
        print(f"Successfully uploaded dataset to Kaggle as '{dataset_name}'")
        return True
    except Exception as e:
        print(f"Error uploading to Kaggle: {e}")
        return False

if __name__ == "__main__":
    # Define paths
    raw_data_path = "data/synthetic_data_full.csv"
    prepared_data_dir = "data/kaggle_prepared"
    
    # Generate synthetic data if it doesn't exist
    if not os.path.exists(raw_data_path):
        df = generate_synthetic_data(raw_data_path)
    else:
        print(f"Loading existing data from {raw_data_path}")
        df = pd.read_csv(raw_data_path, index_col=0, parse_dates=True)
    
    # Prepare data for Kaggle
    train_df, val_df, test_df = prepare_data_for_kaggle(df, prepared_data_dir)
    
    # Upload to Kaggle (uncomment when ready to upload)
    # dataset_name = "energy-fault-prediction-data"
    # dataset_description = "Synthetic data for energy demand forecasting and fault prediction in hybrid energy systems"
    # upload_to_kaggle(prepared_data_dir, dataset_name, dataset_description)
    
    print("Data preparation complete!")
