import numpy as np
from typing import Dict, Any

class BatterySystemSimulator:
    """Simulates a 3 MWh battery energy storage system (2 hours of solar capacity)."""
    
    def __init__(self, capacity_kwh: float = 3000, seed: int = 42):
        self.capacity_kwh = capacity_kwh
        self.max_power_kw = 750  # 50% of solar capacity
        self.min_soc = 0.1  # Lower minimum SOC for more usable capacity
        self.max_soc = 0.95  # Maximum SOC to prevent constant 100%
        np.random.seed(seed)
        
        # System parameters
        self.charging_efficiency = 0.95  # Modern lithium-ion
        self.discharging_efficiency = 0.95  # Modern lithium-ion
        self.self_discharge_rate = 0.0005  # 0.05% per hour
        self.nominal_voltage = 800  # Higher voltage for better efficiency
        self.cycles = 0
        self.degradation_per_cycle = 0.00005  # Modern batteries degrade slower
        
        # Dynamic behavior parameters
        self.charge_rate_factor = {
            'low': 1.0,    # Full power below 30% SOC
            'mid': 0.8,    # 80% power between 30-70% SOC
            'high': 0.5    # 50% power above 70% SOC
        }
        self.discharge_rate_factor = {
            'low': 0.5,    # 50% power below 30% SOC
            'mid': 0.8,    # 80% power between 30-70% SOC
            'high': 1.0    # Full power above 70% SOC
        }
        
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
    
    def get_charge_rate_factor(self, soc: float) -> float:
        """Get charging rate factor based on SOC."""
        if soc < 0.3:
            return self.charge_rate_factor['low']
        elif soc < 0.7:
            return self.charge_rate_factor['mid']
        else:
            return self.charge_rate_factor['high']
    
    def get_discharge_rate_factor(self, soc: float) -> float:
        """Get discharging rate factor based on SOC."""
        if soc < 0.3:
            return self.discharge_rate_factor['low']
        elif soc < 0.7:
            return self.discharge_rate_factor['mid']
        else:
            return self.discharge_rate_factor['high']
    
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
        
        # Initial conditions - start at 60% for more dynamic behavior
        soc[0] = 0.6
        last_soc = soc[0]
        
        # Add some daily variation to load and solar to create more charging opportunities
        day_of_year = df.index.dayofyear.values
        hour_of_day = df.index.hour.values
        
        # Track charge/discharge cycles
        charge_mode = True  # Start with charging mode
        mode_duration = 0   # Track how long we've been in current mode
        
        for i in range(hours):
            # Temperature affects capacity
            temp_factor = self.temperature_effect(df['weather_temperature'].iloc[i])
            effective_capacity = self.capacity_kwh * temp_factor
            
            # Calculate power balance with time-of-day factor
            # Early morning and evening prioritize charging when solar is lower
            tod_factor = 1.0
            if 5 <= hour_of_day[i] <= 8:  # Early morning
                tod_factor = 0.8  # Reduce load demand to allow charging
            elif 17 <= hour_of_day[i] <= 20:  # Evening
                tod_factor = 0.8  # Reduce load demand to allow charging
                
            # Add seasonal variation
            season_factor = 1.0 + 0.1 * np.sin(2 * np.pi * day_of_year[i] / 365)
            
            # Calculate adjusted power balance
            adjusted_load = load_demand[i] * tod_factor * season_factor
            power_balance = pv_output[i] - adjusted_load
            
            # Add small random variations to create more dynamic behavior
            power_balance += np.random.normal(0, 50)  # +/- 50 kW random noise
            
            # Determine if we should switch modes based on SOC and duration
            if charge_mode and (last_soc > 0.9 or mode_duration > 12):
                # Switch to discharge mode if battery is full or we've been charging for 12+ hours
                charge_mode = False
                mode_duration = 0
            elif not charge_mode and (last_soc < 0.3 or mode_duration > 12):
                # Switch to charge mode if battery is low or we've been discharging for 12+ hours
                charge_mode = True
                mode_duration = 0
            
            # Increment mode duration
            mode_duration += 1
            
            # Force mode based on time of day (charge during day, discharge at night)
            if 8 <= hour_of_day[i] <= 16 and power_balance > 100:
                # Good solar hours - prioritize charging
                charge_mode = True
            elif 18 <= hour_of_day[i] <= 22:
                # Evening peak - prioritize discharging
                charge_mode = False
            
            # Sometimes randomly switch modes for more variability
            if np.random.random() < 0.05:  # 5% chance to switch
                charge_mode = not charge_mode
                
            # Apply the current mode
            if charge_mode and last_soc < self.max_soc - 0.05:  # Charge battery if not too full
                # Get charge rate factor based on SOC
                charge_factor = self.get_charge_rate_factor(last_soc)
                
                # Calculate maximum charging power
                max_charge = min(
                    abs(power_balance) if power_balance > 0 else self.max_power_kw * 0.3,
                    self.max_power_kw * charge_factor,
                    (self.max_soc - last_soc) * effective_capacity / self.charging_efficiency
                )
                
                # Ensure some minimum charging when SOC is low
                if last_soc < 0.3:
                    max_charge = max(max_charge, 100)
                
                # Add randomness to charging power
                charge_power = max_charge * (0.5 + 0.5 * np.random.random())
                
                # Negative power means charging
                power[i] = -charge_power
                
                # Calculate energy stored (accounting for efficiency)
                energy_stored = charge_power * self.charging_efficiency
                
            elif not charge_mode and last_soc > self.min_soc + 0.05:  # Discharge battery if not too empty
                # Get discharge rate factor based on SOC
                discharge_factor = self.get_discharge_rate_factor(last_soc)
                
                # Calculate maximum discharge power
                max_discharge = min(
                    abs(power_balance) if power_balance < 0 else self.max_power_kw * 0.3,
                    self.max_power_kw * discharge_factor,
                    (last_soc - self.min_soc) * effective_capacity * self.discharging_efficiency
                )
                
                # Ensure some minimum discharge when SOC is high
                if last_soc > 0.7:
                    max_discharge = max(max_discharge, 100)
                
                # Add randomness to discharge power
                discharge_power = max_discharge * (0.5 + 0.5 * np.random.random())
                
                # Positive power means discharging
                power[i] = discharge_power
                
                # Calculate energy used (accounting for efficiency)
                energy_stored = -discharge_power / self.discharging_efficiency
                
            else:  # Idle - small self-discharge only
                power[i] = 0
                energy_stored = 0
            
            # Update state of charge
            if i < hours - 1:
                # Update SOC based on energy flow
                soc[i + 1] = last_soc - energy_stored / effective_capacity
                
                # Account for self-discharge
                soc[i + 1] *= (1 - self.self_discharge_rate)
                
                # Ensure SOC stays within bounds
                soc[i + 1] = np.clip(soc[i + 1], self.min_soc, self.max_soc)
                
                # Add small random fluctuations for realism
                soc[i + 1] += np.random.normal(0, 0.005)  # +/- 0.5% random noise
                soc[i + 1] = np.clip(soc[i + 1], self.min_soc, self.max_soc)  # Re-clip after noise
                
                last_soc = soc[i + 1]
            
            # Calculate electrical parameters
            current[i] = power[i] * 1000 / self.nominal_voltage  # Convert kW to W
            voltage[i] = self.calculate_voltage(soc[i], current[i])
            
            # Battery temperature (simplified model)
            temp_rise = 0.05 * abs(power[i])  # Temperature rise due to power flow
            temperature[i] = df['weather_temperature'].iloc[i] + temp_rise
        
        return {
            'soc': soc,
            'power': power,  
            'voltage': voltage,
            'current': current,
            'temperature': temperature
        }
