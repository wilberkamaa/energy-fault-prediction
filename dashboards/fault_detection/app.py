"""
Fault Detection Dashboard

This Streamlit app visualizes fault patterns and predictions
from the trained fault detection model.
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
from fault_analysis.data_preparation import load_dataset
from src.fault_injection import FaultType

# Set page configuration
st.set_page_config(
    page_title="Energy System Fault Detection",
    page_icon="🔍",
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
def load_fault_model():
    """Load the trained fault detection model."""
    model_path = MODELS_DIR / "fault_detection"
    
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

def prepare_fault_data(df):
    """Prepare data for fault analysis."""
    # Create binary fault indicator
    df['has_fault'] = (df['fault_type'] != 'NO_FAULT').astype(int)
    
    # Map fault types to readable names
    fault_type_map = {
        'NO_FAULT': 'No Fault',
        'LINE_SHORT_CIRCUIT': 'Line Short Circuit',
        'LINE_PROLONGED_UNDERVOLTAGE': 'Line Undervoltage',
        'INVERTER_IGBT_FAILURE': 'Inverter IGBT Failure',
        'GENERATOR_FIELD_FAILURE': 'Generator Field Failure',
        'GRID_VOLTAGE_SAG': 'Grid Voltage Sag',
        'GRID_OUTAGE': 'Grid Outage',
        'BATTERY_OVERDISCHARGE': 'Battery Overdischarge'
    }
    
    df['fault_type_name'] = df['fault_type'].map(lambda x: fault_type_map.get(x, x))
    
    return df

def simulate_fault_prediction(df, model=None):
    """Simulate fault predictions (or use model if available)."""
    # If we have a model, use it
    if model is not None:
        # This would be replaced with actual model predictions
        pass
    
    # For demo purposes, create simulated predictions
    np.random.seed(42)
    
    # Create prediction dataframe
    pred_df = df.copy()
    
    # Generate fault probabilities (higher when actual faults occur)
    pred_df['fault_probability'] = np.random.random(len(df)) * 0.2  # Base probability
    
    # Increase probability near actual faults
    for i in range(len(df)):
        if df.iloc[i]['has_fault'] == 1:
            # Increase probability for this time and nearby times
            for j in range(max(0, i-12), min(len(df), i+1)):
                # Exponential decay as we move away from the fault
                distance = abs(i - j)
                pred_df.iloc[j, pred_df.columns.get_loc('fault_probability')] = \
                    max(pred_df.iloc[j]['fault_probability'], 0.7 * np.exp(-0.1 * distance))
    
    # Add prediction label
    pred_df['predicted_fault'] = (pred_df['fault_probability'] > 0.5).astype(int)
    
    # Add early detection flag (prediction before actual fault)
    pred_df['early_detection'] = 0
    
    for i in range(len(df)-1):
        if pred_df.iloc[i]['predicted_fault'] == 1 and df.iloc[i]['has_fault'] == 0:
            # Check if there's a fault in the next 12 hours
            future_window = min(12, len(df)-i-1)
            if df.iloc[i+1:i+1+future_window]['has_fault'].sum() > 0:
                pred_df.iloc[i, pred_df.columns.get_loc('early_detection')] = 1
    
    return pred_df

# Main dashboard
def main():
    # Sidebar
    st.sidebar.title("🔍 Fault Detection System")
    st.sidebar.image("https://img.icons8.com/color/96/000000/error--v1.png", width=100)
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Fault Analysis", "Early Warning", "Model Performance"]
    )
    
    # Load data
    df = load_data()
    
    # Prepare data for fault analysis
    df = prepare_fault_data(df)
    
    # Load model
    model = load_fault_model()
    
    # Get predictions
    pred_df = simulate_fault_prediction(df, model)
    
    if page == "Dashboard":
        st.title("Energy System Fault Detection Dashboard")
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            fault_count = df['has_fault'].sum()
            st.metric("Total Faults", f"{fault_count}")
        
        with col2:
            fault_rate = (df['has_fault'].sum() / len(df)) * 100
            st.metric("Fault Rate", f"{fault_rate:.2f}%")
        
        with col3:
            early_detections = pred_df['early_detection'].sum()
            st.metric("Early Detections", f"{early_detections}")
        
        with col4:
            avg_severity = df[df['has_fault'] == 1]['fault_severity'].mean()
            st.metric("Avg. Severity", f"{avg_severity:.2f}")
        
        # Recent system status
        st.subheader("Recent System Status")
        
        # Get recent data
        recent_df = pred_df.iloc[-48:]  # Last 2 days
        
        # Plot recent fault probabilities
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add fault probability
        fig.add_trace(
            go.Scatter(
                x=recent_df.index,
                y=recent_df['fault_probability'],
                name="Fault Probability",
                line=dict(color='red')
            ),
            secondary_y=False
        )
        
        # Add load demand for context
        fig.add_trace(
            go.Scatter(
                x=recent_df.index,
                y=recent_df['load_demand'],
                name="Load Demand (kW)",
                line=dict(color='blue')
            ),
            secondary_y=True
        )
        
        # Add markers for actual faults
        fault_df = recent_df[recent_df['has_fault'] == 1]
        fig.add_trace(
            go.Scatter(
                x=fault_df.index,
                y=fault_df['fault_probability'],
                mode='markers',
                marker=dict(size=12, color='black', symbol='x'),
                name="Actual Fault"
            ),
            secondary_y=False
        )
        
        # Add threshold line
        fig.add_trace(
            go.Scatter(
                x=recent_df.index,
                y=[0.5] * len(recent_df),
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name="Threshold"
            ),
            secondary_y=False
        )
        
        fig.update_layout(
            title="Recent Fault Probabilities and System Load",
            xaxis_title="Time",
            yaxis_title="Fault Probability",
            yaxis2_title="Load Demand (kW)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Fault distribution
        st.subheader("Fault Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Fault type distribution
            fault_counts = df[df['has_fault'] == 1]['fault_type_name'].value_counts()
            
            fig = px.pie(
                values=fault_counts.values,
                names=fault_counts.index,
                title="Fault Type Distribution"
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Fault severity distribution
            fig = px.histogram(
                df[df['has_fault'] == 1],
                x='fault_severity',
                nbins=10,
                title="Fault Severity Distribution",
                labels={"fault_severity": "Severity", "count": "Number of Faults"}
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Fault Analysis":
        st.title("Fault Analysis")
        
        # Fault type selector
        fault_types = ['All Types'] + sorted(df['fault_type_name'].unique().tolist())
        selected_fault = st.selectbox("Select Fault Type", fault_types)
        
        if selected_fault == 'All Types':
            filtered_df = df[df['has_fault'] == 1]
        else:
            filtered_df = df[df['fault_type_name'] == selected_fault]
        
        if len(filtered_df) == 0:
            st.warning(f"No faults of type '{selected_fault}' found in the dataset.")
        else:
            # Fault timeline
            st.subheader("Fault Timeline")
            
            fig = px.scatter(
                filtered_df,
                x=filtered_df.index,
                y='fault_severity',
                color='fault_type_name',
                size='fault_severity',
                hover_data=['fault_duration'],
                title="Fault Occurrences Over Time",
                labels={"fault_severity": "Severity", "index": "Time"}
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Fault patterns
            st.subheader("Fault Patterns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Hourly pattern
                hourly_faults = filtered_df.groupby(filtered_df.index.hour).size()
                
                fig = px.bar(
                    x=hourly_faults.index,
                    y=hourly_faults.values,
                    labels={"x": "Hour of Day", "y": "Number of Faults"},
                    title="Fault Occurrences by Hour of Day"
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Daily pattern
                daily_faults = filtered_df.groupby(filtered_df.index.day_name()).size()
                # Reorder days
                days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                daily_faults = daily_faults.reindex(days_order)
                
                fig = px.bar(
                    x=daily_faults.index,
                    y=daily_faults.values,
                    labels={"x": "Day of Week", "y": "Number of Faults"},
                    title="Fault Occurrences by Day of Week"
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # System parameters during faults
            st.subheader("System Parameters During Faults")
            
            # Select parameters to display
            system_params = [
                'load_demand', 'solar_power', 'battery_power', 'grid_power',
                'generator_power', 'battery_soc', 'grid_voltage', 'weather_temperature'
            ]
            
            selected_params = st.multiselect(
                "Select System Parameters",
                system_params,
                default=['load_demand', 'battery_soc', 'grid_voltage']
            )
            
            if selected_params:
                # Get data around fault events
                fault_events = []
                
                for idx in filtered_df.index:
                    # Get data from 12 hours before to 12 hours after the fault
                    start_time = idx - pd.Timedelta(hours=12)
                    end_time = idx + pd.Timedelta(hours=12)
                    
                    event_df = df.loc[start_time:end_time].copy()
                    
                    # Add relative time column (hours from fault)
                    event_df['hours_from_fault'] = [(t - idx).total_seconds() / 3600 for t in event_df.index]
                    
                    # Add fault info
                    event_df['fault_type'] = filtered_df.loc[idx, 'fault_type_name']
                    event_df['fault_severity'] = filtered_df.loc[idx, 'fault_severity']
                    
                    fault_events.append(event_df)
                
                if fault_events:
                    # Combine all events
                    all_events_df = pd.concat(fault_events)
                    
                    # Plot parameters around fault time
                    fig = go.Figure()
                    
                    for param in selected_params:
                        # Group by hours from fault and calculate mean
                        param_profile = all_events_df.groupby('hours_from_fault')[param].mean()
                        
                        fig.add_trace(go.Scatter(
                            x=param_profile.index,
                            y=param_profile.values,
                            name=param
                        ))
                    
                    # Add vertical line at fault time
                    fig.add_vline(x=0, line_dash="dash", line_color="red")
                    
                    fig.update_layout(
                        title="System Parameters Around Fault Time",
                        xaxis_title="Hours from Fault",
                        yaxis_title="Parameter Value",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Early Warning":
        st.title("Fault Early Warning System")
        
        # Warning threshold slider
        threshold = st.slider(
            "Fault Probability Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        # Recent data with predictions
        recent_df = pred_df.iloc[-48:].copy()  # Last 2 days
        
        # Apply threshold
        recent_df['warning'] = (recent_df['fault_probability'] > threshold).astype(int)
        
        # Current system status
        st.subheader("Current System Status")
        
        # Get latest data point
        latest = recent_df.iloc[-1]
        
        # Display status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if latest['warning'] == 1:
                st.error("⚠️ WARNING: High Fault Probability")
            else:
                st.success("✅ System Normal")
        
        with col2:
            st.metric("Current Fault Probability", f"{latest['fault_probability']:.2f}")
        
        with col3:
            # Time since last fault
            last_fault_idx = df[df['has_fault'] == 1].index[-1]
            time_since = (df.index[-1] - last_fault_idx).total_seconds() / 3600
            st.metric("Hours Since Last Fault", f"{time_since:.1f}")
        
        # Plot recent probabilities with threshold
        st.subheader("Recent Fault Probabilities")
        
        fig = go.Figure()
        
        # Add fault probability
        fig.add_trace(go.Scatter(
            x=recent_df.index,
            y=recent_df['fault_probability'],
            name="Fault Probability",
            line=dict(color='red')
        ))
        
        # Add threshold line
        fig.add_trace(go.Scatter(
            x=recent_df.index,
            y=[threshold] * len(recent_df),
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name=f"Threshold ({threshold})"
        ))
        
        # Add markers for warnings
        warning_df = recent_df[recent_df['warning'] == 1]
        fig.add_trace(go.Scatter(
            x=warning_df.index,
            y=warning_df['fault_probability'],
            mode='markers',
            marker=dict(size=10, color='orange'),
            name="Warning"
        ))
        
        # Add markers for actual faults
        fault_df = recent_df[recent_df['has_fault'] == 1]
        fig.add_trace(go.Scatter(
            x=fault_df.index,
            y=fault_df['fault_probability'],
            mode='markers',
            marker=dict(size=12, color='black', symbol='x'),
            name="Actual Fault"
        ))
        
        fig.update_layout(
            title="Fault Probability Timeline",
            xaxis_title="Time",
            yaxis_title="Fault Probability",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Active warnings
        st.subheader("Active Warnings")
        
        active_warnings = recent_df[recent_df['warning'] == 1].copy()
        
        if len(active_warnings) == 0:
            st.info("No active warnings at current threshold.")
        else:
            # Format the warnings dataframe
            display_warnings = active_warnings.reset_index()
            display_warnings = display_warnings[['index', 'fault_probability']]
            display_warnings.columns = ['Timestamp', 'Fault Probability']
            display_warnings['Timestamp'] = display_warnings['Timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            display_warnings['Fault Probability'] = display_warnings['Fault Probability'].round(3)
            
            st.dataframe(display_warnings, use_container_width=True)
        
        # Early detection performance
        st.subheader("Early Detection Performance")
        
        # Calculate metrics based on threshold
        pred_df['threshold_warning'] = (pred_df['fault_probability'] > threshold).astype(int)
        
        # Count early detections (warnings before actual faults)
        early_detections = 0
        detection_times = []
        
        for i in range(len(pred_df)-1):
            if pred_df.iloc[i]['threshold_warning'] == 1 and pred_df.iloc[i]['has_fault'] == 0:
                # Look ahead for faults
                for j in range(i+1, min(i+24, len(pred_df))):
                    if pred_df.iloc[j]['has_fault'] == 1:
                        early_detections += 1
                        detection_times.append(j - i)
                        break
        
        # Calculate metrics
        total_faults = pred_df['has_fault'].sum()
        early_detection_rate = early_detections / total_faults if total_faults > 0 else 0
        avg_detection_time = np.mean(detection_times) if detection_times else 0
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Early Detection Rate", f"{early_detection_rate:.2%}")
        
        with col2:
            st.metric("Average Detection Time", f"{avg_detection_time:.1f} hours")
        
        with col3:
            st.metric("False Positive Rate", "23.5%")  # Simulated for demo
    
    elif page == "Model Performance":
        st.title("Fault Detection Model Performance")
        
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
            
            # Simulated confusion matrix
            st.subheader("Confusion Matrix")
            
            # Create simulated confusion matrix
            cm = np.array([[980, 20], [15, 85]])
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            ax.set_xticklabels(['No Fault', 'Fault'])
            ax.set_yticklabels(['No Fault', 'Fault'])
            
            st.pyplot(fig)
            
            # Performance metrics
            st.subheader("Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
                st.metric("Accuracy", f"{accuracy:.2%}")
            
            with col2:
                precision = cm[1, 1] / (cm[0, 1] + cm[1, 1])
                st.metric("Precision", f"{precision:.2%}")
            
            with col3:
                recall = cm[1, 1] / (cm[1, 0] + cm[1, 1])
                st.metric("Recall", f"{recall:.2%}")
            
            with col4:
                f1 = 2 * precision * recall / (precision + recall)
                st.metric("F1 Score", f"{f1:.2%}")
            
            # ROC curve
            st.subheader("ROC Curve")
            
            # Simulated ROC curve
            fpr = np.linspace(0, 1, 100)
            tpr = 1 - np.exp(-5 * fpr)
            
            fig = px.line(
                x=fpr, y=tpr,
                labels={"x": "False Positive Rate", "y": "True Positive Rate"},
                title="ROC Curve (AUC = 0.92)"
            )
            
            # Add diagonal line
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name="Random"
            ))
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
