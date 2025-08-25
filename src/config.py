config = {
    'battery': {
        'capacity_kwh': 100,
        'max_power_kw': 50,
        'min_soc': 0.2,
        'max_soc': 0.95,
        'charging_efficiency': 0.95,
        'discharging_efficiency': 0.95,
        'self_discharge_rate': 0.001,
        'nominal_voltage': 400,
        'degradation_per_cycle': 0.0001,
        'charge_rate_factor': 0.5,
        'discharge_rate_factor': 0.5
    },
    'diesel_generator': {
        'capacity_kva': 1000,
        'fuel_tank_capacity': 5000,
        'min_load_percent': 0.3,
        'min_runtime_hours': 1,
        'fuel_consumption_rate': {
            'idle': 10,
            'full_load': 250
        },
        'maintenance_interval': 500
    },
    'fault_injection': {
        'fault_probabilities': {
            'LINE_SHORT_CIRCUIT': 0.05,
            'INVERTER_IGBT_FAILURE': 0.02,
            'GRID_OUTAGE': 0.1
        },
        'fault_durations': {
            'LINE_SHORT_CIRCUIT': (1, 4),
            'INVERTER_IGBT_FAILURE': (2, 8),
            'GRID_OUTAGE': (1, 24)
        },
        'thresholds': {
            'grid_voltage': 0.8 * 25000,
            'inverter_temperature': 80,
            'generator_runtime': 100,
            'battery_soc': 0.2
        }
    },
    'grid': {
        'include_grid': False,
        'nominal_voltage': 25000,
        'base_reliability': 0.95,
        'voltage_variation': 0.1,
        'peak_hours': (8, 20),
        'season_factors': {
            'long_rains': 0.85,
            'short_rains': 0.9,
            'dry': 1.0
        },
        'max_export': 1000
    },
    'load_profile': {
        'base_load_kw': 500,
        'peak_load_kw': 2000,
        'weekday_factors': {
            'morning_peak': {'hours': (6, 9), 'factor': 1.3},
            'evening_peak': {'hours': (18, 22), 'factor': 1.5},
            'night_valley': {'hours': (23, 5), 'factor': 0.7}
        },
        'weekend_reduction': 0.8,
        'seasonal_factors': {
            'long_rains': 0.9,
            'short_rains': 0.95,
            'dry': 1.1
        },
        'holidays': {
            # Major Kenyan holidays
            (1, 1),    # New Year's Day
            (5, 1),    # Labour Day
            (6, 1),    # Madaraka Day
            (10, 20),  # Mashujaa Day
            (12, 12),  # Jamhuri Day
            (12, 25),  # Christmas Day
            (12, 26),  # Boxing Day
        }
    },
    'solar_pv': {
        'capacity_kw': 1500,
        'nominal_efficiency': 0.21,
        'temp_coefficient': -0.003,
        'dust_loss_rate': 0.0005,
        'noct': 42,
        'base_efficiency': 0.23,
        'system_efficiency': 0.85
    },
    'validation': {
        'valid_ranges': {
            'weather_temperature': (-10, 45),
            'weather_humidity': (0, 100),
            'weather_cloud_cover': (0, 100),
            'weather_wind_speed': (0, 30),
            'solar_power': (0, 1500),
            'solar_cell_temp': (0, 85),
            'battery_soc': (0, 1),
            'battery_power': (-200, 200),
            'battery_voltage': (350, 450),
            'battery_current': (-500, 500),
            'battery_temperature': (0, 60),
            'generator_power': (0, 2000),
            'generator_fuel_level': (0, 5000),
            'generator_frequency': (55, 65),
            'generator_temperature': (0, 120),
            'grid_voltage': (0.8 * 25000, 1.2 * 25000),
            'grid_frequency': (48, 52),
            'grid_power': (-2000, 2000),
            'load_demand': (0, 2000),
            'load_power_factor': (0.8, 1.0),
            'fault_severity': (0, 1)
        },
        'power_balance_tolerance': 0.01,
        'nan_fill_value': 0
    },
    'weather': {
        'season_params': {
            'long_rains': {'cloud_cover': (0.4, 0.6), 'temp_range': (20, 28)},
            'short_rains': {'cloud_cover': (0.3, 0.5), 'temp_range': (22, 30)},
            'dry': {'cloud_cover': (0.1, 0.3), 'temp_range': (25, 33)}
        },
        'base_temperature': {
            'mean': 25,
            'daily_variation': 5,
            'peak_hour': 14
        },
        'season_temp_offset': {
            'long_rains': -2,
            'short_rains': 0,
            'dry': 2
        },
        'seed': 42,
        'base_temperature': 25,
        'temperature_amplitude': 5,
        'temperature_peak_hour': 14,
        'temperature_noise_std': 0.5,
    }
}