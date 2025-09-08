# GitHub Repository Setup Guide

This guide provides step-by-step instructions for setting up your GitHub repository for the Energy Fault Prediction final year project.

## Step 1: Initialize and Configure Local Repository

Your local repository is already initialized with Git. The next steps will help you prepare it for GitHub.

```bash
# Navigate to your project directory
cd /home/wilberkamau/CascadeProjects/energy-fault-prediction

# Stage all files for commit
git add .

# Commit changes with an initial commit message
git commit -m "Initial commit: Energy fault prediction project structure"
```

## Step 2: Create a New Repository on GitHub

1. Go to [GitHub](https://github.com/) and sign in to your account
2. Click the "+" icon in the top right corner and select "New repository"
3. Name your repository "energy-fault-prediction"
4. Add a description: "Energy demand forecasting and fault detection system for hybrid energy systems"
5. Keep the repository public (or private if you prefer)
6. Do NOT initialize with README, .gitignore, or license (we already have these)
7. Click "Create repository"

## Step 3: Connect Local Repository to GitHub

After creating the repository, GitHub will show instructions. Follow these commands:

```bash
# Add the remote repository URL
git remote add origin https://github.com/YOUR_USERNAME/energy-fault-prediction.git

# Push your local repository to GitHub
git push -u origin main
```

## Step 4: Organize Your Repository Structure

Your repository is already well-organized with the following structure:

```
energy-fault-prediction/
├── data/                  # Synthetic datasets
├── dashboards/            # Streamlit dashboards
│   ├── demand_forecast/   # Demand forecasting dashboard
│   └── fault_detection/   # Fault detection dashboard
├── docs/                  # Documentation
├── fault_analysis/        # Fault analysis modules
├── kaggle_integration/    # Scripts for Kaggle model training
├── models/                # Model definitions and saved models
│   └── trained/           # Pre-trained model files from Kaggle
├── notebooks/             # Jupyter notebooks
├── output/                # Generated visualizations
├── src/                   # Source code
└── tests/                 # Unit tests
```

## Step 5: Set Up GitHub Actions for CI/CD (Optional)

Create a GitHub Actions workflow to automate testing:

1. Create a directory: `.github/workflows`
2. Create a file: `.github/workflows/python-tests.yml` with basic test configuration

## Step 6: Kaggle Integration Setup

1. Generate a Kaggle API token:
   - Go to your Kaggle account settings
   - Scroll to "API" section and click "Create New API Token"
   - This downloads a `kaggle.json` file

2. Set up your Kaggle credentials locally:
   ```bash
   # Create Kaggle directory
   mkdir -p ~/.kaggle
   
   # Move your kaggle.json file (after downloading)
   mv ~/Downloads/kaggle.json ~/.kaggle/
   
   # Set proper permissions
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. Test the Kaggle integration:
   ```bash
   # Test that Kaggle API works
   python -c "import kaggle; print('Kaggle API is working!')"
   ```

## Step 7: Managing Large Model Files with Git LFS (Optional)

For trained models that are too large for regular Git:

1. Install Git LFS:
   ```bash
   sudo apt-get install git-lfs
   ```

2. Initialize Git LFS:
   ```bash
   cd /home/wilberkamau/CascadeProjects/energy-fault-prediction
   git lfs install
   ```

3. Track large model files:
   ```bash
   git lfs track "models/trained/**/*.h5"
   git lfs track "models/trained/**/saved_model.pb"
   ```

4. Add the `.gitattributes` file:
   ```bash
   git add .gitattributes
   git commit -m "Configure Git LFS for model files"
   ```

## Step 8: Workflow for Training Models on Kaggle

1. Generate synthetic data locally
2. Use `kaggle_integration/prepare_dataset.py` to prepare and upload data to Kaggle
3. Use `kaggle_integration/train_model.py` to create and push training notebooks to Kaggle
4. Run the notebooks on Kaggle with GPU acceleration
5. Use `kaggle_integration/import_model.py` to download trained models

## Step 9: Running the Streamlit Dashboards

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the demand forecasting dashboard:
   ```bash
   streamlit run dashboards/demand_forecast/app.py
   ```

3. Run the fault detection dashboard:
   ```bash
   streamlit run dashboards/fault_detection/app.py
   ```

## Step 10: Regular Updates and Maintenance

1. Pull the latest changes before starting work:
   ```bash
   git pull origin main
   ```

2. Commit changes regularly:
   ```bash
   git add .
   git commit -m "Descriptive message about your changes"
   ```

3. Push changes to GitHub:
   ```bash
   git push origin main
   ```

## Additional Resources

- [GitHub Documentation](https://docs.github.com/)
- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Git LFS Documentation](https://git-lfs.github.com/)
