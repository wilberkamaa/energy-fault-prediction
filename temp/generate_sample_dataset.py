import pandas as pd
import numpy as np
from src.data_generator import HybridSystemDataGenerator
from src.config import config
import os

# Ensure output directory exists
os.makedirs('data/sample', exist_ok=True)

# Create a smaller dataset for testing (1 month instead of 2 years)
print("Generating sample dataset with grid functionality disabled...")

# Verify grid is disabled in config
print(f"Grid inclusion status: {config['grid']['include_grid']}")

# Generate a small dataset (1 month)
generator = HybridSystemDataGenerator(seed=42)
dataset = generator.generate_dataset(
    start_date='2023-01-01',
    periods_years=1,  # 1 month
    output_file='data/sample/hybrid_system_sample.parquet'
)

# Print dataset info
print("\nDataset generated successfully!")
print(f"Shape: {dataset.shape}")
print("\nColumns:")
print(dataset.columns.tolist())

# Verify grid columns have placeholder values
print("\nGrid columns (should have placeholder values):")
grid_cols = [col for col in dataset.columns if col.startswith('grid_')]
for col in grid_cols:
    unique_vals = dataset[col].unique()
    print(f"{col}: unique values = {unique_vals}")

# Sample a few rows
print("\nSample rows:")
print(dataset.sample(5))