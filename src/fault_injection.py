import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum, auto
from src.config import config

class FaultType(Enum):
    """Enumeration of possible fault types."""
    LINE_SHORT_CIRCUIT = auto()
    LINE_PROLONGED_UNDERVOLTAGE = auto()
    INVERTER_IGBT_FAILURE = auto()
    GENERATOR_FIELD_FAILURE = auto()
    # Removed: GRID_VOLTAGE_SAG = auto()
    # Removed: GRID_OUTAGE = auto()
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
        
        # Load fault configuration
        fault_config = config['fault_injection']
        
        # Base fault probabilities (per hour)
        self.fault_probabilities = fault_config['fault_probabilities']
        
        # Typical fault durations (hours)
        self.fault_durations = fault_config['fault_durations']
    
    def check_fault_conditions(self, system_state: Dict[str, np.ndarray], 
                             hour: int) -> List[Tuple[FaultType, float]]:
        """Check if conditions are met for different types of faults."""
        potential_faults = []
        fault_config = config['fault_injection']
        
        # Line faults - Remove grid voltage check
        # The grid_voltage check has been removed as part of the grid functionality removal
        
        # Inverter faults
        if system_state.get('inverter_temp') is not None:
            if system_state['inverter_temp'][hour] > fault_config['thresholds'].get('inverter_temp', 80):
                potential_faults.append(
                    (FaultType.INVERTER_IGBT_FAILURE,
                     self.fault_probabilities.get(FaultType.INVERTER_IGBT_FAILURE.name, 0.02) * 
                     (system_state['inverter_temp'][hour] - fault_config['thresholds'].get('inverter_temp', 80)) / 10)
                )
        
        # Generator faults
        if system_state.get('generator_runtime') is not None:
            if system_state['generator_runtime'][hour] > fault_config['thresholds'].get('generator_runtime', 100):
                potential_faults.append(
                    (FaultType.GENERATOR_FIELD_FAILURE,
                     self.fault_probabilities.get(FaultType.GENERATOR_FIELD_FAILURE.name, 0.02) * 
                     (system_state['generator_runtime'][hour] / fault_config['thresholds'].get('generator_runtime', 100)))
                )
        
        # Battery faults
        if system_state.get('battery_soc') is not None:
            if system_state['battery_soc'][hour] < fault_config['thresholds'].get('battery_soc', 0.2):
                potential_faults.append(
                    (FaultType.BATTERY_OVERDISCHARGE,
                     self.fault_probabilities.get(FaultType.BATTERY_OVERDISCHARGE.name, 0.05) * 
                     (fault_config['thresholds'].get('battery_soc', 0.2) - system_state['battery_soc'][hour]) * 10)
                )
        
        return potential_faults
    
    def generate_fault_events(self, df, system_state: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Generate fault events based on system conditions."""
        hours = len(df)
        
        # Initialize arrays
        fault_occurred = np.zeros(hours, dtype=bool)
        fault_types = np.full(hours, 'NO_FAULT', dtype=object)  
        fault_severity = np.zeros(hours)
        active_faults: List[FaultEvent] = []
        
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
                        *self.fault_durations.get(fault_type.name, (1, 4))
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
            
            # Record current fault state
            if active_faults:
                # Take the most severe active fault
                current_fault = max(active_faults, key=lambda x: x.severity)
                fault_occurred[hour] = True
                fault_types[hour] = current_fault.fault_type.name  
                fault_severity[hour] = current_fault.severity
        
        # Convert fault events to a more usable format
        fault_starts = np.zeros(hours, dtype=bool)
        fault_durations = np.zeros(hours)
        
        for fault in active_faults:
            fault_starts[fault.start_time] = True
            fault_durations[fault.start_time] = fault.duration
        
        return {
            'occurred': fault_occurred,
            'type': fault_types,
            'severity': fault_severity,
            'start': fault_starts,
            'duration': fault_durations
        }
    
    def _generate_fault_effects(self, fault_type: FaultType, 
                              severity: float) -> Dict[str, float]:
        """Generate the effects of a fault on system parameters."""
        effects = {}
        fault_config = config.get('fault_injection', {}).get('fault_effects', {})
        
        if fault_type == FaultType.LINE_SHORT_CIRCUIT:
            line_short_circuit = fault_config.get('line_short_circuit', {
                'voltage_drop_base': 0.1,
                'voltage_drop_factor': 0.2,
                'current_spike_base': 1.5,
                'current_spike_factor': 2.0
            })
            effects.update({
                'voltage_drop': line_short_circuit.get('voltage_drop_base', 0.1) + 
                               line_short_circuit.get('voltage_drop_factor', 0.2) * severity,
                'current_spike': line_short_circuit.get('current_spike_base', 1.5) + 
                                line_short_circuit.get('current_spike_factor', 2.0) * severity
            })
        elif fault_type == FaultType.INVERTER_IGBT_FAILURE:
            inverter_failure = fault_config.get('inverter_failure', {
                'efficiency_drop_factor': 0.3,
                'temperature_rise_factor': 15
            })
            effects.update({
                'efficiency_drop': inverter_failure.get('efficiency_drop_factor', 0.3) * severity,
                'temperature_rise': inverter_failure.get('temperature_rise_factor', 15) * severity
            })
        elif fault_type == FaultType.GENERATOR_FIELD_FAILURE:
            generator_failure = fault_config.get('generator_failure', {
                'voltage_deviation_factor': 0.15,
                'frequency_deviation_factor': 5
            })
            effects.update({
                'voltage_deviation': generator_failure.get('voltage_deviation_factor', 0.15) * severity,
                'frequency_deviation': generator_failure.get('frequency_deviation_factor', 5) * severity
            })
        elif fault_type == FaultType.BATTERY_OVERDISCHARGE:
            battery_overdischarge = fault_config.get('battery_overdischarge', {
                'capacity_loss_factor': 0.05,
                'internal_resistance_base': 0.01,
                'internal_resistance_factor': 0.05
            })
            effects.update({
                'capacity_loss': battery_overdischarge.get('capacity_loss_factor', 0.05) * severity,
                'internal_resistance': battery_overdischarge.get('internal_resistance_base', 0.01) + 
                                      battery_overdischarge.get('internal_resistance_factor', 0.05) * severity
            })
        
        return effects
