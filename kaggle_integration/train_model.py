"""
Script to create and push Kaggle notebooks for model training.

This script:
1. Creates Jupyter notebooks for model training
2. Pushes them to Kaggle
3. Optionally triggers training runs
"""
import os
import sys
import json
import kaggle
from pathlib import Path
import nbformat as nbf

def create_training_notebook(notebook_path, dataset_name, model_type="fault_detection"):
    """Create a Jupyter notebook for model training on Kaggle."""
    print(f"Creating {model_type} training notebook...")
    
    # Create a new notebook
    nb = nbf.v4.new_notebook()
    
    # Add cells based on model type
    if model_type == "fault_detection":
        cells = [
            nbf.v4.new_markdown_cell("# Fault Detection Model Training\n\nThis notebook trains a deep learning model for fault detection in hybrid energy systems."),
            nbf.v4.new_code_cell("# Import libraries\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.metrics import classification_report, confusion_matrix\nimport tensorflow as tf\nfrom tensorflow.keras.models import Sequential\nfrom tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional"),
            nbf.v4.new_code_cell(f"# Load dataset\ndf_train = pd.read_csv('../input/{dataset_name}/train.csv', index_col=0, parse_dates=True)\ndf_val = pd.read_csv('../input/{dataset_name}/validation.csv', index_col=0, parse_dates=True)\ndf_test = pd.read_csv('../input/{dataset_name}/test.csv', index_col=0, parse_dates=True)\n\nprint(f'Training data shape: {df_train.shape}')\nprint(f'Validation data shape: {df_val.shape}')\nprint(f'Test data shape: {df_test.shape}')"),
            nbf.v4.new_code_cell("# Prepare features and target\ndef prepare_data(df):\n    # Create binary target (fault/no-fault)\n    y = (df['fault_type'] != 'NO_FAULT').astype(int)\n    \n    # Select features (exclude fault columns)\n    fault_cols = [col for col in df.columns if col.startswith('fault_')]\n    X = df.drop(columns=fault_cols)\n    \n    # Scale features\n    scaler = StandardScaler()\n    X_scaled = pd.DataFrame(\n        scaler.fit_transform(X),\n        columns=X.columns,\n        index=X.index\n    )\n    \n    return X_scaled, y\n\nX_train, y_train = prepare_data(df_train)\nX_val, y_val = prepare_data(df_val)\nX_test, y_test = prepare_data(df_test)"),
            nbf.v4.new_code_cell("# Create sequences for LSTM\ndef create_sequences(X, y, seq_length=24):\n    X_seq, y_seq = [], []\n    for i in range(len(X) - seq_length):\n        X_seq.append(X.iloc[i:i+seq_length].values)\n        y_seq.append(y.iloc[i+seq_length])\n    return np.array(X_seq), np.array(y_seq)\n\nseq_length = 24  # 24 hours\nX_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)\nX_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)\nX_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)\n\nprint(f'Training sequences shape: {X_train_seq.shape}')\nprint(f'Validation sequences shape: {X_val_seq.shape}')\nprint(f'Test sequences shape: {X_test_seq.shape}')"),
            nbf.v4.new_code_cell("# Build LSTM model\ndef build_model(input_shape):\n    model = Sequential([\n        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),\n        Dropout(0.2),\n        Bidirectional(LSTM(32)),\n        Dropout(0.2),\n        Dense(16, activation='relu'),\n        Dense(1, activation='sigmoid')\n    ])\n    \n    model.compile(\n        optimizer='adam',\n        loss='binary_crossentropy',\n        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]\n    )\n    \n    return model\n\nmodel = build_model((X_train_seq.shape[1], X_train_seq.shape[2]))\nmodel.summary()"),
            nbf.v4.new_code_cell("# Train model\nearly_stopping = tf.keras.callbacks.EarlyStopping(\n    monitor='val_loss',\n    patience=10,\n    restore_best_weights=True\n)\n\nhistory = model.fit(\n    X_train_seq, y_train_seq,\n    epochs=50,\n    batch_size=32,\n    validation_data=(X_val_seq, y_val_seq),\n    callbacks=[early_stopping]\n)"),
            nbf.v4.new_code_cell("# Plot training history\nplt.figure(figsize=(12, 4))\nplt.subplot(1, 2, 1)\nplt.plot(history.history['loss'], label='Train')\nplt.plot(history.history['val_loss'], label='Validation')\nplt.title('Loss')\nplt.legend()\n\nplt.subplot(1, 2, 2)\nplt.plot(history.history['accuracy'], label='Train')\nplt.plot(history.history['val_accuracy'], label='Validation')\nplt.title('Accuracy')\nplt.legend()\nplt.tight_layout()\nplt.show()"),
            nbf.v4.new_code_cell("# Evaluate model\ny_pred = (model.predict(X_test_seq) > 0.5).astype(int).flatten()\n\nprint('Classification Report:')\nprint(classification_report(y_test_seq, y_pred))\n\n# Plot confusion matrix\ncm = confusion_matrix(y_test_seq, y_pred)\nplt.figure(figsize=(8, 6))\nsns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\nplt.xlabel('Predicted')\nplt.ylabel('Actual')\nplt.title('Confusion Matrix')\nplt.show()"),
            nbf.v4.new_code_cell("# Save model\nmodel.save('fault_detection_model.h5')\nprint('Model saved as fault_detection_model.h5')\n\n# Also save as TensorFlow SavedModel format\nmodel.save('fault_detection_model')\nprint('Model saved in SavedModel format')")
        ]
    elif model_type == "demand_forecast":
        cells = [
            nbf.v4.new_markdown_cell("# Demand Forecasting Model Training\n\nThis notebook trains a time series forecasting model for energy demand prediction."),
            nbf.v4.new_code_cell("# Import libraries\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\nimport tensorflow as tf\nfrom tensorflow.keras.models import Sequential\nfrom tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten"),
            nbf.v4.new_code_cell(f"# Load dataset\ndf_train = pd.read_csv('../input/{dataset_name}/train.csv', index_col=0, parse_dates=True)\ndf_val = pd.read_csv('../input/{dataset_name}/validation.csv', index_col=0, parse_dates=True)\ndf_test = pd.read_csv('../input/{dataset_name}/test.csv', index_col=0, parse_dates=True)\n\nprint(f'Training data shape: {df_train.shape}')\nprint(f'Validation data shape: {df_val.shape}')\nprint(f'Test data shape: {df_test.shape}')"),
            nbf.v4.new_code_cell("# Prepare features and target for demand forecasting\ndef prepare_data(df):\n    # Target is load_demand\n    y = df['load_demand']\n    \n    # Select features (exclude fault columns and target)\n    exclude_cols = [col for col in df.columns if col.startswith('fault_')] + ['load_demand']\n    X = df.drop(columns=exclude_cols)\n    \n    # Scale features\n    scaler = StandardScaler()\n    X_scaled = pd.DataFrame(\n        scaler.fit_transform(X),\n        columns=X.columns,\n        index=X.index\n    )\n    \n    # Scale target\n    target_scaler = StandardScaler()\n    y_scaled = pd.Series(\n        target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten(),\n        index=y.index\n    )\n    \n    return X_scaled, y_scaled, target_scaler\n\nX_train, y_train, target_scaler = prepare_data(df_train)\nX_val, y_val, _ = prepare_data(df_val)\nX_test, y_test, _ = prepare_data(df_test)"),
            nbf.v4.new_code_cell("# Create sequences for time series forecasting\ndef create_sequences(X, y, seq_length=24, forecast_horizon=12):\n    X_seq, y_seq = [], []\n    for i in range(len(X) - seq_length - forecast_horizon + 1):\n        X_seq.append(X.iloc[i:i+seq_length].values)\n        y_seq.append(y.iloc[i+seq_length:i+seq_length+forecast_horizon].values)\n    return np.array(X_seq), np.array(y_seq)\n\nseq_length = 24  # 24 hours of history\nforecast_horizon = 12  # Predict next 12 hours\n\nX_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length, forecast_horizon)\nX_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length, forecast_horizon)\nX_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length, forecast_horizon)\n\nprint(f'Training sequences shape: {X_train_seq.shape}')\nprint(f'Training targets shape: {y_train_seq.shape}')"),
            nbf.v4.new_code_cell("# Build CNN-LSTM model for demand forecasting\ndef build_model(input_shape, output_length):\n    model = Sequential([\n        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),\n        MaxPooling1D(pool_size=2),\n        Conv1D(filters=32, kernel_size=3, activation='relu'),\n        LSTM(64, return_sequences=True),\n        Dropout(0.2),\n        LSTM(32),\n        Dropout(0.2),\n        Dense(32, activation='relu'),\n        Dense(output_length)\n    ])\n    \n    model.compile(\n        optimizer='adam',\n        loss='mse',\n        metrics=['mae']\n    )\n    \n    return model\n\nmodel = build_model((X_train_seq.shape[1], X_train_seq.shape[2]), forecast_horizon)\nmodel.summary()"),
            nbf.v4.new_code_cell("# Train model\nearly_stopping = tf.keras.callbacks.EarlyStopping(\n    monitor='val_loss',\n    patience=10,\n    restore_best_weights=True\n)\n\nhistory = model.fit(\n    X_train_seq, y_train_seq,\n    epochs=50,\n    batch_size=32,\n    validation_data=(X_val_seq, y_val_seq),\n    callbacks=[early_stopping]\n)"),
            nbf.v4.new_code_cell("# Plot training history\nplt.figure(figsize=(12, 4))\nplt.subplot(1, 2, 1)\nplt.plot(history.history['loss'], label='Train')\nplt.plot(history.history['val_loss'], label='Validation')\nplt.title('Loss (MSE)')\nplt.legend()\n\nplt.subplot(1, 2, 2)\nplt.plot(history.history['mae'], label='Train')\nplt.plot(history.history['val_mae'], label='Validation')\nplt.title('Mean Absolute Error')\nplt.legend()\nplt.tight_layout()\nplt.show()"),
            nbf.v4.new_code_cell("# Evaluate model\ny_pred = model.predict(X_test_seq)\n\n# Inverse transform predictions and actual values\ny_test_inv = target_scaler.inverse_transform(y_test_seq)\ny_pred_inv = target_scaler.inverse_transform(y_pred)\n\n# Calculate metrics\nmae = mean_absolute_error(y_test_inv.flatten(), y_pred_inv.flatten())\nrmse = np.sqrt(mean_squared_error(y_test_inv.flatten(), y_pred_inv.flatten()))\nr2 = r2_score(y_test_inv.flatten(), y_pred_inv.flatten())\n\nprint(f'Mean Absolute Error: {mae:.2f} kW')\nprint(f'Root Mean Squared Error: {rmse:.2f} kW')\nprint(f'R² Score: {r2:.4f}')"),
            nbf.v4.new_code_cell("# Plot sample predictions\ndef plot_prediction(idx):\n    plt.figure(figsize=(12, 6))\n    \n    # Plot history\n    history_data = X_test_seq[idx, :, X_test.columns.get_loc('load_demand')]\n    history_dates = pd.date_range(end=df_test.index[seq_length + idx], periods=seq_length, freq='H')\n    plt.plot(history_dates, target_scaler.inverse_transform(history_data.reshape(-1, 1)), 'b-', label='Historical Load')\n    \n    # Plot forecast\n    forecast_dates = pd.date_range(start=df_test.index[seq_length + idx], periods=forecast_horizon, freq='H')\n    plt.plot(forecast_dates, y_test_inv[idx], 'g-', label='Actual Load')\n    plt.plot(forecast_dates, y_pred_inv[idx], 'r--', label='Predicted Load')\n    \n    plt.title(f'Load Demand Forecast Starting at {forecast_dates[0]}')\n    plt.xlabel('Time')\n    plt.ylabel('Load Demand (kW)')\n    plt.legend()\n    plt.grid(True, alpha=0.3)\n    plt.tight_layout()\n    \nplot_prediction(0)  # First test sample\nplot_prediction(24)  # Sample from next day"),
            nbf.v4.new_code_cell("# Save model\nmodel.save('demand_forecast_model.h5')\nprint('Model saved as demand_forecast_model.h5')\n\n# Also save as TensorFlow SavedModel format\nmodel.save('demand_forecast_model')\nprint('Model saved in SavedModel format')")
        ]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Add cells to notebook
    nb['cells'] = cells
    
    # Write notebook to file
    os.makedirs(os.path.dirname(notebook_path), exist_ok=True)
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)
    
    print(f"Notebook created at {notebook_path}")
    return notebook_path

def push_notebook_to_kaggle(notebook_path, kernel_metadata):
    """Push a notebook to Kaggle as a kernel."""
    try:
        # Write kernel metadata
        metadata_path = os.path.join(os.path.dirname(notebook_path), 'kernel-metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(kernel_metadata, f, indent=2)
        
        # Push to Kaggle
        kaggle.api.kernels_push_cli(os.path.dirname(notebook_path))
        
        print(f"Successfully pushed notebook to Kaggle as '{kernel_metadata['id']}'")
        return True
    except Exception as e:
        print(f"Error pushing notebook to Kaggle: {e}")
        return False

if __name__ == "__main__":
    # Define paths and settings
    notebooks_dir = "kaggle_integration/notebooks"
    dataset_name = "yourusername/energy-fault-prediction-data"  # Replace with your actual dataset name
    
    # Create fault detection notebook
    fault_notebook_path = os.path.join(notebooks_dir, "fault_detection", "fault_detection_training.ipynb")
    create_training_notebook(fault_notebook_path, dataset_name, "fault_detection")
    
    # Create demand forecasting notebook
    demand_notebook_path = os.path.join(notebooks_dir, "demand_forecast", "demand_forecast_training.ipynb")
    create_training_notebook(demand_notebook_path, dataset_name, "demand_forecast")
    
    # Push notebooks to Kaggle (uncomment when ready)
    """
    # Push fault detection notebook
    fault_metadata = {
        "id": "yourusername/fault-detection-training",
        "title": "Fault Detection Model Training",
        "code_file": "fault_detection_training.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [dataset_name],
        "competition_sources": [],
        "kernel_sources": []
    }
    push_notebook_to_kaggle(fault_notebook_path, fault_metadata)
    
    # Push demand forecasting notebook
    demand_metadata = {
        "id": "yourusername/demand-forecast-training",
        "title": "Demand Forecasting Model Training",
        "code_file": "demand_forecast_training.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [dataset_name],
        "competition_sources": [],
        "kernel_sources": []
    }
    push_notebook_to_kaggle(demand_notebook_path, demand_metadata)
    """
    
    print("Notebook creation complete! Uncomment the push code when ready to upload to Kaggle.")
