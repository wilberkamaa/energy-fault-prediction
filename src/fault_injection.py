import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum, auto

class FaultType(Enum):
    """Enumeration of possible fault types."""
    LINE_SHORT_CIRCUIT = auto()
    LINE_PROLONGED_UNDERVOLTAGE = auto()
    INVERTER_IGBT_FAILURE = auto()
    GENERATOR_FIELD_FAILURE = auto()
    GRID_VOLTAGE_SAG = auto()
    GRID_OUTAGE = auto()
    BATTERY_OVERDISCHARGE = auto()
    NO_FAULT = auto()

@dataclass
class FaultEvent:
    """Data class for fault events."""
    fault_type: FaultType
    start_time: int
    duration: int
    severity: float
    affected_parameters: Dict[str, float]

class FaultInjectionSystem:
    """Simulates various faults in the hybrid energy system."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
        # Base fault probabilities (per hour)
        self.fault_probabilities = {
            FaultType.LINE_SHORT_CIRCUIT: 0.001,
            FaultType.LINE_PROLONGED_UNDERVOLTAGE: 0.002,
            FaultType.INVERTER_IGBT_FAILURE: 0.001,
            FaultType.GENERATOR_FIELD_FAILURE: 0.001,
            FaultType.GRID_VOLTAGE_SAG: 0.005,
            FaultType.GRID_OUTAGE: 0.002,
            FaultType.BATTERY_OVERDISCHARGE: 0.001
        }
        
        # Typical fault durations (hours)
        self.fault_durations = {
            FaultType.LINE_SHORT_CIRCUIT: (1, 4),
            FaultType.LINE_PROLONGED_UNDERVOLTAGE: (2, 8),
            FaultType.INVERTER_IGBT_FAILURE: (4, 24),
            FaultType.GENERATOR_FIELD_FAILURE: (8, 48),
            FaultType.GRID_VOLTAGE_SAG: (1, 6),
            FaultType.GRID_OUTAGE: (2, 12),
            FaultType.BATTERY_OVERDISCHARGE: (1, 4)
        }
    
    def check_fault_conditions(self, system_state: Dict[str, np.ndarray], 
                             hour: int) -> List[Tuple[FaultType, float]]:
        """Check if conditions are met for different types of faults."""
        potential_faults = []
        
        # Line faults
        if system_state.get('grid_voltage') is not None:
            if system_state['grid_voltage'][hour] < 0.8 * 25000:  # 80% of nominal
                potential_faults.append(
                    (FaultType.LINE_SHORT_CIRCUIT, 
                     self.fault_probabilities[FaultType.LINE_SHORT_CIRCUIT] * 2)
                )
        
        # Inverter faults
        if system_state.get('inverter_temp') is not None:
            if system_state['inverter_temp'][hour] > 80:  # °C
                potential_faults.append(
                    (FaultType.INVERTER_IGBT_FAILURE,
                     self.fault_probabilities[FaultType.INVERTER_IGBT_FAILURE] * 
                     (system_state['inverter_temp'][hour] - 80) / 10)
                )
        
        # Generator faults
        if system_state.get('generator_runtime') is not None:
            if system_state['generator_runtime'][hour] > 100:
                potential_faults.append(
                    (FaultType.GENERATOR_FIELD_FAILURE,
                     self.fault_probabilities[FaultType.GENERATOR_FIELD_FAILURE] * 
                     (system_state['generator_runtime'][hour] / 100))
                )
        
        # Battery faults
        if system_state.get('battery_soc') is not None:
            if system_state['battery_soc'][hour] < 0.2:  # 20% SOC
                potential_faults.append(
                    (FaultType.BATTERY_OVERDISCHARGE,
                     self.fault_probabilities[FaultType.BATTERY_OVERDISCHARGE] * 
                     (0.2 - system_state['battery_soc'][hour]) * 10)
                )
        
        return potential_faults
    
    def generate_fault_events(self, df, system_state: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Generate fault events based on system conditions."""
        hours = len(df)
        
        # Initialize arrays
        fault_occurred = np.zeros(hours, dtype=bool)
        fault_types = np.full(hours, FaultType.NO_FAULT)
        fault_severity = np.zeros(hours)
        active_faults: List[FaultEvent] = []
        all_fault_events: List[FaultEvent] = []
        
        for hour in range(hours):
            # Check for potential new faults
            potential_faults = self.check_fault_conditions(system_state, hour)
            
            # Remove expired faults
            active_faults = [
                fault for fault in active_faults
                if hour < fault.start_time + fault.duration
            ]
            
            # Process potential new faults
            for fault_type, probability in potential_faults:
                if np.random.random() < probability:
                    # Generate new fault
                    duration = np.random.randint(
                        *self.fault_durations[fault_type]
                    )
                    severity = np.random.uniform(0.3, 1.0)
                    
                    # Create fault event
                    fault_event = FaultEvent(
                        fault_type=fault_type,
                        start_time=hour,
                        duration=duration,
                        severity=severity,
                        affected_parameters=self._generate_fault_effects(
                            fault_type, severity
                        )
                    )
                    
                    active_faults.append(fault_event)
                    all_fault_events.append(fault_event)
            
            # Record current fault state
            if active_faults:
                # Take the most severe active fault
                current_fault = max(active_faults, key=lambda x: x.severity)
                fault_occurred[hour] = True
                fault_types[hour] = current_fault.fault_type
                fault_severity[hour] = current_fault.severity
        
        return {
            'fault_occurred': fault_occurred,
            'fault_types': fault_types,
            'fault_severity': fault_severity,
            'fault_events': all_fault_events
        }
    
    def _generate_fault_effects(self, fault_type: FaultType, 
                              severity: float) -> Dict[str, float]:
        """Generate the effects of a fault on system parameters."""
        effects = {}
        
        if fault_type == FaultType.LINE_SHORT_CIRCUIT:
            effects.update({
                'voltage_drop': 0.8 + 0.2 * severity,
                'current_spike': 1.5 + 0.5 * severity
            })
        elif fault_type == FaultType.INVERTER_IGBT_FAILURE:
            effects.update({
                'efficiency_drop': 0.3 * severity,
                'temperature_rise': 20 * severity
            })
        elif fault_type == FaultType.GENERATOR_FIELD_FAILURE:
            effects.update({
                'voltage_deviation': 0.1 * severity,
                'frequency_deviation': 0.05 * severity
            })
        elif fault_type == FaultType.BATTERY_OVERDISCHARGE:
            effects.update({
                'capacity_loss': 0.1 * severity,
                'internal_resistance': 1.2 + 0.3 * severity
            })
        
        return effects
