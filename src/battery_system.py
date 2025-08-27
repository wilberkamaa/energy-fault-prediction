import numpy as np
from typing import Dict, Any
from src.config import config

class BatterySystemSimulator:
    """Simulates a battery energy storage system."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
        # Load battery parameters from config
        battery_config = config['battery']
        self.capacity_kwh = battery_config['capacity_kwh']
        self.max_power_kw = battery_config['max_power_kw']
        self.min_soc = battery_config['min_soc']
        self.max_soc = battery_config['max_soc']
        self.charging_efficiency = battery_config['charging_efficiency']
        self.discharging_efficiency = battery_config['discharging_efficiency']
        self.self_discharge_rate = battery_config['self_discharge_rate']
        self.nominal_voltage = battery_config['nominal_voltage']
        self.degradation_per_cycle = battery_config['degradation_per_cycle']
        self.charge_rate_factor = battery_config['charge_rate_factor']
        self.discharge_rate_factor = battery_config['discharge_rate_factor']
        
        # Initialize state
        self.state_of_charge = self.max_soc
        self.cycle_count = 0
        self.remaining_capacity = self.capacity_kwh
        
    def calculate_power_limits(self) -> Dict[str, float]:
        """Calculate current charge/discharge power limits based on SOC."""
        # Charging limit
        soc_headroom = self.max_soc - self.state_of_charge
        charge_limit = min(
            self.max_power_kw * self.charge_rate_factor,
            (soc_headroom * self.capacity_kwh) / self.charging_efficiency
        )
        
        # Discharging limit
        available_energy = (self.state_of_charge - self.min_soc) * self.capacity_kwh
        discharge_limit = min(
            self.max_power_kw * self.discharge_rate_factor,
            available_energy * self.discharging_efficiency
        )
        
        return {
            'charge_limit': charge_limit,
            'discharge_limit': discharge_limit
        }
    
    def update_state(self, power_kw: float, time_step_hours: float = 1.0) -> None:
        """Update battery state based on power flow and time step."""
        # Calculate energy change
        if power_kw > 0:  # Charging
            energy_change = power_kw * time_step_hours * self.charging_efficiency
        else:  # Discharging
            energy_change = power_kw * time_step_hours / self.discharging_efficiency
        
        # Update SOC
        energy_capacity = self.capacity_kwh * (1 - self.degradation_per_cycle * self.cycle_count)
        soc_change = energy_change / energy_capacity
        self.state_of_charge = np.clip(
            self.state_of_charge + soc_change - self.self_discharge_rate * time_step_hours,
            self.min_soc,
            self.max_soc
        )
        
        # Update cycle count (partial cycles)
        if power_kw != 0:
            self.cycle_count += abs(soc_change) / 2  # Half cycle for each direction
        
        # Update remaining capacity
        self.remaining_capacity = energy_capacity
    
    def generate_output(self, df: pd.DataFrame, power_request_series: pd.Series) -> Dict[str, Any]:
        """Generate battery system output parameters for a time series."""
        num_steps = len(df)
        
        # Initialize arrays to store results
        power_output = np.zeros(num_steps)
        current_output = np.zeros(num_steps)
        voltage_output = np.full(num_steps, self.nominal_voltage)
        soc_output = np.zeros(num_steps)
        temperature_output = np.zeros(num_steps)
        capacity_output = np.zeros(num_steps)
        cycle_count_output = np.zeros(num_steps)

        for i in range(num_steps):
            power_request = power_request_series.iloc[i]
            
            # Get current power limits
            limits = self.calculate_power_limits()
            
            # Limit power request to available capacity
            if power_request > 0:  # Charging
                power = min(power_request, limits['charge_limit'])
            else:  # Discharging
                power = max(power_request, -limits['discharge_limit'])
            
            # Update battery state
            self.update_state(power)
            
            # Calculate current
            current = power * 1000 / self.nominal_voltage if power != 0 else 0
            
            # Store results
            power_output[i] = power
            current_output[i] = current
            soc_output[i] = self.state_of_charge
            temperature_output[i] = 25 + abs(current) * 0.1  # Simple temperature model
            capacity_output[i] = self.remaining_capacity
            cycle_count_output[i] = self.cycle_count

        return {
            'power': power_output,
            'current': current_output,
            'voltage': voltage_output,
            'soc': soc_output,
            'temperature': temperature_output,
            'remaining_capacity': capacity_output,
            'cycle_count': cycle_count_output
        }
