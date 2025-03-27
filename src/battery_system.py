import numpy as np
from typing import Dict, Any

class BatterySystemSimulator:
    """Simulates a 1 MWh battery energy storage system."""
    
    def __init__(self, capacity_kwh: float = 1000, seed: int = 42):
        self.capacity_kwh = capacity_kwh
        self.max_power_kw = 200  # Maximum charge/discharge rate
        np.random.seed(seed)
        
        # System parameters
        self.charging_efficiency = 0.92
        self.discharging_efficiency = 0.94
        self.self_discharge_rate = 0.001  # 0.1% per hour
        self.nominal_voltage = 480
        self.cycles = 0
        self.degradation_per_cycle = 0.0001  # 0.01% capacity loss per cycle
        
    def temperature_effect(self, temperature: float) -> float:
        """Calculate temperature effect on capacity."""
        # Optimal temperature is 25°C
        # Capacity decreases by 2% per 10°C deviation
        temp_diff = abs(temperature - 25)
        return 1 - (0.002 * temp_diff)
    
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
        soc = np.zeros(hours)  # State of Charge
        power_output = np.zeros(hours)
        voltage = np.zeros(hours)
        current = np.zeros(hours)
        temperature = np.zeros(hours)
        cycles = np.zeros(hours)
        
        # Initial conditions
        soc[0] = 0.8  # Start at 80% SOC
        cumulative_cycles = 0
        last_soc = soc[0]
        
        for i in range(hours):
            # Temperature affects capacity
            temp_factor = self.temperature_effect(df['temperature'][i])
            effective_capacity = self.capacity_kwh * temp_factor
            
            # Calculate power balance
            power_balance = pv_output[i] - load_demand[i]
            
            if power_balance > 0:  # Excess power - charge battery
                max_charge = min(
                    power_balance,
                    self.max_power_kw,
                    (1 - soc[i]) * effective_capacity
                )
                power_output[i] = -max_charge  # Negative means charging
                
                # Account for charging efficiency
                energy_stored = max_charge * self.charging_efficiency
                
            else:  # Power deficit - discharge battery
                max_discharge = min(
                    -power_balance,
                    self.max_power_kw,
                    soc[i] * effective_capacity
                )
                power_output[i] = max_discharge
                
                # Account for discharging efficiency
                energy_stored = -max_discharge / self.discharging_efficiency
            
            # Update SOC
            if i < hours - 1:
                soc[i + 1] = soc[i] + energy_stored / effective_capacity
                
                # Account for self-discharge
                soc[i + 1] *= (1 - self.self_discharge_rate)
                
                # Calculate partial cycles
                cycle_fraction = abs(soc[i + 1] - last_soc) / 2
                cumulative_cycles += cycle_fraction
                last_soc = soc[i + 1]
            
            # Calculate electrical parameters
            current[i] = power_output[i] * 1000 / self.nominal_voltage  # Convert kW to W
            voltage[i] = self.calculate_voltage(soc[i], current[i])
            
            # Battery temperature (simplified model)
            temp_rise = abs(power_output[i]) / self.max_power_kw * 15  # Max 15°C rise
            temperature[i] = df['temperature'][i] + temp_rise
            
            cycles[i] = cumulative_cycles
            
        return {
            'soc': soc,
            'power_output': power_output,
            'voltage': voltage,
            'current': current,
            'temperature': temperature,
            'cycles': cycles,
            'capacity_degradation': 1 - (cycles * self.degradation_per_cycle)
        }
