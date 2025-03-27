#!/usr/bin/env python3
"""
Test script for battery system behavior
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_generator import HybridSystemDataGenerator

# Set up the generator
generator = HybridSystemDataGenerator(seed=42)

# Generate a shorter dataset for testing (0.1 years = ~36 days)
print("Generating dataset...")
df = generator.generate_dataset(
    start_date='2023-01-01',
    periods_years=0.1,
    output_file='notebooks/synthetic_data_test.csv'
)
print(f"Dataset generated with shape: {df.shape}")

# Extract first week for visualization
week_data = df['2023-01-01':'2023-01-07']

# Create output directory
os.makedirs('output', exist_ok=True)

# Plot 1: Battery State of Charge
plt.figure(figsize=(15, 6))
plt.plot(week_data.index, week_data['battery_soc'] * 100, label='Battery SOC (%)', color='red')
plt.title('Battery State of Charge - First Week of 2023')
plt.xlabel('Date')
plt.ylabel('State of Charge (%)')
plt.grid(True)
plt.savefig('output/battery_soc.png')
plt.close()

# Plot 2: Battery Power
plt.figure(figsize=(15, 6))
plt.plot(week_data.index, week_data['battery_power'], label='Battery Power (kW)', color='purple')
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Battery Power Output - First Week of 2023')
plt.xlabel('Date')
plt.ylabel('Power (kW)')
plt.legend()
plt.grid(True)
plt.savefig('output/battery_power.png')
plt.close()

# Plot 3: Solar, Battery, and Load
plt.figure(figsize=(15, 6))
plt.plot(week_data.index, week_data['solar_power'], label='Solar Output (kW)', color='orange')
plt.plot(week_data.index, week_data['battery_power'], label='Battery Power (kW)', color='purple')
plt.plot(week_data.index, week_data['load_demand'], label='Load Demand (kW)', color='blue')
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Power Balance - First Week of 2023')
plt.xlabel('Date')
plt.ylabel('Power (kW)')
plt.legend()
plt.grid(True)
plt.savefig('output/power_balance.png')
plt.close()

# Print some statistics
print("\nBattery Statistics:")
print(f"Min SOC: {df['battery_soc'].min() * 100:.2f}%")
print(f"Max SOC: {df['battery_soc'].max() * 100:.2f}%")
print(f"Mean SOC: {df['battery_soc'].mean() * 100:.2f}%")
print(f"SOC Standard Deviation: {df['battery_soc'].std() * 100:.2f}%")

print("\nBattery Power Statistics:")
print(f"Min Power: {df['battery_power'].min():.2f} kW")
print(f"Max Power: {df['battery_power'].max():.2f} kW")
print(f"Mean Power: {df['battery_power'].mean():.2f} kW")
print(f"Power Standard Deviation: {df['battery_power'].std():.2f} kW")

print("\nDone! Check output directory for plots.")
