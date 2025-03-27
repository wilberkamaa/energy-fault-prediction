"""
Import trained models from Kaggle to the local environment.

This script:
1. Downloads model files from Kaggle kernels
2. Converts them to the appropriate format if needed
3. Saves them to the local models directory
"""
import os
import sys
import kaggle
import tensorflow as tf
from pathlib import Path
import shutil

def download_model_from_kaggle(kernel_name, output_dir):
    """Download model files from a Kaggle kernel."""
    print(f"Downloading model from Kaggle kernel: {kernel_name}")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Download kernel output
        kaggle.api.kernels_output(kernel_name, path=output_dir)
        
        print(f"Successfully downloaded model files to {output_dir}")
        return True
    except Exception as e:
        print(f"Error downloading from Kaggle: {e}")
        return False

def import_tensorflow_model(model_path, output_path):
    """Import a TensorFlow model and save it to the local models directory."""
    try:
        # Load the model
        if model_path.endswith('.h5'):
            model = tf.keras.models.load_model(model_path)
            print(f"Loaded H5 model from {model_path}")
        else:
            model = tf.keras.models.load_model(model_path)
            print(f"Loaded SavedModel from {model_path}")
        
        # Save the model in both formats
        h5_path = f"{output_path}.h5"
        savedmodel_path = output_path
        
        model.save(h5_path)
        model.save(savedmodel_path)
        
        print(f"Model saved as H5: {h5_path}")
        print(f"Model saved as SavedModel: {savedmodel_path}")
        
        return model
    except Exception as e:
        print(f"Error importing model: {e}")
        return None

def main():
    # Define paths
    models_dir = Path(__file__).parent.parent / "models" / "trained"
    kaggle_output_dir = Path(__file__).parent / "kaggle_output"
    
    # Create directories if they don't exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(kaggle_output_dir, exist_ok=True)
    
    # Define Kaggle kernels to download models from
    kernels = [
        {
            "name": "yourusername/fault-detection-training",
            "model_files": ["fault_detection_model.h5", "fault_detection_model"],
            "output_name": "fault_detection"
        },
        {
            "name": "yourusername/demand-forecast-training",
            "model_files": ["demand_forecast_model.h5", "demand_forecast_model"],
            "output_name": "demand_forecast"
        }
    ]
    
    for kernel in kernels:
        # Download model files from Kaggle
        kernel_output_dir = kaggle_output_dir / kernel["name"].split("/")[1]
        success = download_model_from_kaggle(kernel["name"], kernel_output_dir)
        
        if success:
            # Import and save models
            for model_file in kernel["model_files"]:
                model_path = kernel_output_dir / model_file
                
                if os.path.exists(model_path):
                    if model_file.endswith('.h5'):
                        output_path = models_dir / kernel["output_name"]
                        model = import_tensorflow_model(str(model_path), str(output_path))
                    elif os.path.isdir(model_path):
                        # For SavedModel format (directory)
                        output_path = models_dir / kernel["output_name"]
                        if os.path.exists(output_path):
                            shutil.rmtree(output_path)
                        shutil.copytree(model_path, output_path)
                        print(f"Copied SavedModel directory from {model_path} to {output_path}")
                else:
                    print(f"Model file not found: {model_path}")
    
    print("\nModel import complete!")
    print(f"Trained models are available in: {models_dir}")

if __name__ == "__main__":
    main()
