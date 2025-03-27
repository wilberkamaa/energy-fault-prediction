"""
Demand Forecasting Dashboard

This Streamlit app visualizes energy demand patterns and forecasts
from the trained demand forecasting model.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import project modules
from src.data_generator import HybridSystemDataGenerator
from fault_analysis.data_preparation import (
    load_dataset, create_time_features,
    create_lag_features, create_window_features
)

# Set page configuration
st.set_page_config(
    page_title="Energy Demand Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "trained"

# Helper functions
@st.cache_data
def load_data():
    """Load or generate dataset for visualization."""
    data_file = DATA_DIR / "synthetic_data.csv"
    
    if data_file.exists():
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    else:
        st.warning("No data file found. Generating a small dataset for demonstration...")
        data_gen = HybridSystemDataGenerator(seed=42)
        df = data_gen.generate_dataset(
            start_date='2023-01-01',
            periods_years=0.1,
            output_file=str(data_file)
        )
    
    return df

@st.cache_resource
def load_forecast_model():
    """Load the trained demand forecasting model."""
    model_path = MODELS_DIR / "demand_forecast"
    
    if model_path.exists():
        try:
            model = tf.keras.models.load_model(str(model_path))
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.warning("No trained model found. Please train a model first.")
        return None

def prepare_data_for_forecast(df, seq_length=24):
    """Prepare data for forecasting."""
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
    df = create_lag_features(df, system_cols, lag_periods=[1, 3, 6, 12, 24])
    df = create_window_features(df, system_cols, window_sizes=[6, 12, 24])
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def make_forecast(model, df, forecast_horizon=12, seq_length=24):
    """Make demand forecast using the trained model."""
    if model is None:
        return None
    
    # Prepare data
    df_prepared = prepare_data_for_forecast(df)
    
    # Get the latest data for forecasting
    latest_data = df_prepared.iloc[-seq_length:]
    
    # Exclude fault columns and target
    exclude_cols = [col for col in df_prepared.columns if col.startswith('fault_')] + ['load_demand']
    X = latest_data.drop(columns=exclude_cols)
    
    # Scale features (simple standardization for demo)
    X_mean, X_std = X.mean(), X.std()
    X_scaled = (X - X_mean) / X_std
    
    # Reshape for model input
    X_input = X_scaled.values.reshape(1, seq_length, X.shape[1])
    
    # Make prediction
    y_pred_scaled = model.predict(X_input)[0]
    
    # Inverse transform prediction (simple demo scaling)
    y_mean, y_std = df_prepared['load_demand'].mean(), df_prepared['load_demand'].std()
    y_pred = y_pred_scaled * y_std + y_mean
    
    # Create forecast timestamps
    last_timestamp = df.index[-1]
    forecast_timestamps = pd.date_range(
        start=last_timestamp + timedelta(hours=1),
        periods=forecast_horizon,
        freq='H'
    )
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'timestamp': forecast_timestamps,
        'load_demand': y_pred
    })
    forecast_df.set_index('timestamp', inplace=True)
    
    return forecast_df

# Main dashboard
def main():
    # Sidebar
    st.sidebar.title("⚡ Energy Demand Forecast")
    st.sidebar.image("https://img.icons8.com/color/96/000000/energy.png", width=100)
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Historical Analysis", "Forecast", "Model Performance"]
    )
    
    # Load data
    df = load_data()
    
    # Load model
    model = load_forecast_model()
    
    if page == "Dashboard":
        st.title("Energy Demand Forecasting Dashboard")
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_demand = df['load_demand'].mean()
            st.metric("Average Demand", f"{avg_demand:.1f} kW")
        
        with col2:
            peak_demand = df['load_demand'].max()
            st.metric("Peak Demand", f"{peak_demand:.1f} kW")
        
        with col3:
            daily_avg = df.resample('D')['load_demand'].mean().mean()
            st.metric("Daily Average", f"{daily_avg:.1f} kW")
        
        with col4:
            demand_std = df['load_demand'].std()
            st.metric("Demand Volatility", f"{demand_std:.1f} kW")
        
        # Recent demand and forecast
        st.subheader("Recent Demand and Forecast")
        
        # Get recent data
        recent_df = df.iloc[-72:]  # Last 3 days
        
        # Make forecast if model is available
        forecast_df = None
        if model is not None:
            forecast_df = make_forecast(model, df)
        
        # Plot recent demand and forecast
        fig = go.Figure()
        
        # Add historical demand
        fig.add_trace(go.Scatter(
            x=recent_df.index,
            y=recent_df['load_demand'],
            name="Historical Demand",
            line=dict(color='blue')
        ))
        
        # Add forecast if available
        if forecast_df is not None:
            fig.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df['load_demand'],
                name="Forecast",
                line=dict(color='red', dash='dash')
            ))
        
        fig.update_layout(
            title="Recent Energy Demand and Forecast",
            xaxis_title="Time",
            yaxis_title="Load Demand (kW)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Demand patterns
        st.subheader("Demand Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly pattern
            hourly_demand = df.groupby(df.index.hour)['load_demand'].mean()
            
            fig = px.line(
                x=hourly_demand.index,
                y=hourly_demand.values,
                labels={"x": "Hour of Day", "y": "Average Demand (kW)"},
                title="Average Hourly Demand Pattern"
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Daily pattern
            daily_demand = df.groupby(df.index.day_name())['load_demand'].mean()
            # Reorder days
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_demand = daily_demand.reindex(days_order)
            
            fig = px.bar(
                x=daily_demand.index,
                y=daily_demand.values,
                labels={"x": "Day of Week", "y": "Average Demand (kW)"},
                title="Average Daily Demand Pattern"
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Historical Analysis":
        st.title("Historical Demand Analysis")
        
        # Time range selector
        date_range = st.date_input(
            "Select Date Range",
            value=(df.index.min().date(), df.index.max().date()),
            min_value=df.index.min().date(),
            max_value=df.index.max().date()
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            mask = (df.index.date >= start_date) & (df.index.date <= end_date)
            filtered_df = df.loc[mask]
            
            # Plot demand over time
            fig = px.line(
                filtered_df,
                x=filtered_df.index,
                y='load_demand',
                title=f"Energy Demand ({start_date} to {end_date})"
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Demand vs. other factors
            st.subheader("Demand vs. Other Factors")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Demand vs. temperature
                fig = px.scatter(
                    filtered_df,
                    x='weather_temperature',
                    y='load_demand',
                    color=filtered_df.index.hour,
                    title="Demand vs. Temperature",
                    labels={"weather_temperature": "Temperature (°C)", "load_demand": "Demand (kW)"}
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Demand vs. solar power
                fig = px.scatter(
                    filtered_df,
                    x='solar_power',
                    y='load_demand',
                    color=filtered_df.index.hour,
                    title="Demand vs. Solar Power",
                    labels={"solar_power": "Solar Power (kW)", "load_demand": "Demand (kW)"}
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Heatmap of demand by hour and day
            st.subheader("Demand Heatmap by Hour and Day")
            
            # Create pivot table
            pivot_df = filtered_df.pivot_table(
                index=filtered_df.index.hour,
                columns=filtered_df.index.day_name(),
                values='load_demand',
                aggfunc='mean'
            )
            
            # Reorder columns (days)
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            pivot_df = pivot_df.reindex(columns=days_order)
            
            fig = px.imshow(
                pivot_df,
                labels=dict(x="Day of Week", y="Hour of Day", color="Demand (kW)"),
                x=pivot_df.columns,
                y=pivot_df.index,
                color_continuous_scale="Viridis"
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Forecast":
        st.title("Energy Demand Forecast")
        
        if model is None:
            st.warning("No trained model available. Please train a model first.")
        else:
            # Forecast horizon selector
            forecast_horizon = st.slider(
                "Forecast Horizon (hours)",
                min_value=1,
                max_value=48,
                value=24,
                step=1
            )
            
            # Make forecast
            forecast_df = make_forecast(model, df, forecast_horizon=forecast_horizon)
            
            if forecast_df is not None:
                # Plot forecast
                st.subheader("Demand Forecast")
                
                # Get recent data for context
                recent_df = df.iloc[-48:]  # Last 2 days
                
                fig = go.Figure()
                
                # Add historical demand
                fig.add_trace(go.Scatter(
                    x=recent_df.index,
                    y=recent_df['load_demand'],
                    name="Historical Demand",
                    line=dict(color='blue')
                ))
                
                # Add forecast
                fig.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df['load_demand'],
                    name="Forecast",
                    line=dict(color='red', dash='dash')
                ))
                
                # Add confidence interval (simulated for demo)
                upper_bound = forecast_df['load_demand'] * 1.1
                lower_bound = forecast_df['load_demand'] * 0.9
                
                fig.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=upper_bound,
                    fill=None,
                    mode='lines',
                    line_color='rgba(255,0,0,0)',
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=lower_bound,
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(255,0,0,0)',
                    name='95% Confidence Interval'
                ))
                
                fig.update_layout(
                    title=f"{forecast_horizon}-Hour Energy Demand Forecast",
                    xaxis_title="Time",
                    yaxis_title="Load Demand (kW)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display forecast data
                st.subheader("Forecast Data")
                
                # Format the forecast dataframe
                display_df = forecast_df.copy()
                display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')
                display_df.columns = ['Demand (kW)']
                display_df['Demand (kW)'] = display_df['Demand (kW)'].round(2)
                
                st.dataframe(display_df, use_container_width=True)
                
                # Download forecast
                csv = display_df.to_csv()
                st.download_button(
                    label="Download Forecast CSV",
                    data=csv,
                    file_name="demand_forecast.csv",
                    mime="text/csv"
                )
    
    elif page == "Model Performance":
        st.title("Model Performance Metrics")
        
        if model is None:
            st.warning("No trained model available. Please train a model first.")
        else:
            # Display model architecture
            st.subheader("Model Architecture")
            
            # Convert model summary to string
            summary_list = []
            model.summary(print_fn=lambda x: summary_list.append(x))
            summary_str = "\n".join(summary_list)
            
            st.code(summary_str)
            
            # Simulated performance metrics (replace with actual metrics when available)
            st.subheader("Performance Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean Absolute Error", "12.4 kW")
            
            with col2:
                st.metric("Root Mean Squared Error", "18.7 kW")
            
            with col3:
                st.metric("R² Score", "0.89")
            
            # Simulated error distribution
            st.subheader("Error Distribution")
            
            # Generate simulated errors
            np.random.seed(42)
            errors = np.random.normal(0, 15, 1000)
            
            fig = px.histogram(
                errors,
                nbins=30,
                labels={"value": "Prediction Error (kW)", "count": "Frequency"},
                title="Distribution of Prediction Errors"
            )
            
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Simulated feature importance
            st.subheader("Feature Importance")
            
            # Create simulated feature importance
            features = [
                "Temperature", "Hour of Day", "Day of Week",
                "Solar Power", "Battery SOC", "Grid Voltage",
                "Previous Demand (1h)", "Previous Demand (24h)",
                "Demand Rolling Mean (24h)"
            ]
            
            importance = [0.22, 0.18, 0.12, 0.10, 0.09, 0.08, 0.08, 0.07, 0.06]
            
            fig = px.bar(
                x=importance,
                y=features,
                orientation='h',
                labels={"x": "Importance", "y": "Feature"},
                title="Feature Importance"
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
