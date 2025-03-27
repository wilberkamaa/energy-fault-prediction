import numpy as np
from typing import Dict, Any

class BatterySystemSimulator:
    """Simulates a 3 MWh battery energy storage system (2 hours of solar capacity)."""
    
    def __init__(self, capacity_kwh: float = 3000, seed: int = 42):
        self.capacity_kwh = capacity_kwh
        self.max_power_kw = 750  # 50% of solar capacity
        self.min_soc = 0.1  # Lower minimum SOC for more usable capacity
        np.random.seed(seed)
        
        # System parameters
        self.charging_efficiency = 0.95  # Modern lithium-ion
        self.discharging_efficiency = 0.95  # Modern lithium-ion
        self.self_discharge_rate = 0.0005  # 0.05% per hour
        self.nominal_voltage = 800  # Higher voltage for better efficiency
        self.cycles = 0
        self.degradation_per_cycle = 0.00005  # Modern batteries degrade slower
        
    def temperature_effect(self, temperature: float) -> float:
        """Calculate temperature effect on capacity."""
        # Modern batteries have better temperature tolerance
        # Capacity decreases by 1% per 10°C deviation
        temp_diff = abs(temperature - 25)
        return 1 - (0.001 * temp_diff)
    
    def calculate_voltage(self, soc: float, current: float) -> float:
        """Calculate battery voltage based on SOC and current."""
        # Simple battery voltage model
        voltage = self.nominal_voltage * (0.9 + 0.2 * soc)
        # Add internal resistance effect
        voltage -= current * 0.1  # Simplified internal resistance effect
        return voltage
    
    def generate_output(self, df, pv_output: np.ndarray, 
                       load_demand: np.ndarray) -> Dict[str, Any]:
        """Generate battery system output parameters."""
        hours = len(df)
        
        # Initialize arrays
        soc = np.zeros(hours)  # State of charge
        power = np.zeros(hours)  # Power output (positive for discharge)
        voltage = np.zeros(hours)
        current = np.zeros(hours)
        temperature = np.zeros(hours)
        
        # Initial conditions
        soc[0] = 0.8  # Start at 80% charge
        last_soc = soc[0]
        
        for i in range(hours):
            # Temperature affects capacity
            temp_factor = self.temperature_effect(df['weather_temperature'][i])
            effective_capacity = self.capacity_kwh * temp_factor
            
            # Calculate power balance
            power_balance = pv_output[i] - load_demand[i]
            
            if power_balance > 0:  # Excess power, can charge battery
                # Calculate maximum charging power
                max_charge = min(
                    power_balance,
                    self.max_power_kw,
                    (1 - last_soc) * effective_capacity * self.charging_efficiency
                )
                power[i] = -min(power_balance, max_charge)  # Negative for charging
                
                # Account for charging efficiency
                energy_stored = max_charge * self.charging_efficiency
            else:  # Power deficit, need to discharge battery
                # Calculate maximum discharge power
                max_discharge = min(
                    -power_balance,
                    self.max_power_kw,
                    (last_soc - self.min_soc) * effective_capacity / self.discharging_efficiency
                )
                power[i] = min(-power_balance, max_discharge)  # Positive for discharging
                
                # Account for discharging efficiency
                energy_stored = -max_discharge / self.discharging_efficiency
            
            # Update state of charge
            if i < hours - 1:
                if power[i] < 0:  # Charging
                    soc[i + 1] = soc[i] + energy_stored / effective_capacity
                else:  # Discharging
                    soc[i + 1] = soc[i] - energy_stored / effective_capacity
                
                # Account for self-discharge
                soc[i + 1] *= (1 - self.self_discharge_rate)
                
                # Ensure SOC stays within bounds
                soc[i + 1] = np.clip(soc[i + 1], self.min_soc, 1.0)
                last_soc = soc[i + 1]
            
            # Calculate electrical parameters
            current[i] = power[i] * 1000 / self.nominal_voltage  # Convert kW to W
            voltage[i] = self.calculate_voltage(soc[i], current[i])
            
            # Battery temperature (simplified model)
            temp_rise = 0.05 * abs(power[i])  # Temperature rise due to power flow
            temperature[i] = df['weather_temperature'][i] + temp_rise
            
        return {
            'soc': soc,
            'power': power,  
            'voltage': voltage,
            'current': current,
            'temperature': temperature
        }
