# -*- coding: utf-8 -*-
"""
UFLUX-emsemble model training and products generating program
Author: Songyan Zhu
Email: Songyan.Zhu@soton.ac.uk
"""

import numpy as np # Fundamental package for numerical computing
import pandas as pd # Library for data manipulation and analysis
import xarray as xr # Library for labelled multi-dimensional arrays (great for climate/remote sensing data)
import pickle # Used for serializing and deserializing Python objects (saving/loading models)
from pathlib import Path # Object-oriented filesystem paths
from deepforest import CascadeForestRegressor # Deep Forest implementation
from xgboost import XGBRegressor # Implementation of the gradient boosting framework
from tqdm import tqdm # Library to display progress bars for loops
from sciml import pipelines # Scientific Machine Learning (SciML) utility pipelines
from scigeo.meteo import saturation_vapor_pressure # Meteorological calculations (e.g., for VPD)
from scieco import photosynthesis # Ecosystem calculations (e.g., for iWUE)
from sklearn.model_selection import GridSearchCV # Utility for exhaustive search over specified parameter values for an estimator

# ------------------------------------------------------------------------------

# Define the root directory for the project data
# Assuming 'root' is defined elsewhere, likely a Path object pointing to the project root.
project_directory = Path('<YOUR PROJECT DIRECTORY>')

# ------------------------------------------------------------------------------

# Load the training dataset from a parquet file
# This dataset contains flux tower measurements and corresponding input features.
df_training = pd.read_parquet(project_directory.joinpath('0_input/0_towerlevel_data/UFLUX-ensemble_training_dataset.parquet'))

# Define lists of common and domain-specific feature names (independent variables X)

# Common features available across different data sources
X_names_common = ['IGBP', 'DoY_sin', 'DoY_cos', 'Latitude'] # IGBP land cover type, Day of Year (sin/cos transformed), and Latitude

# ERA5-specific climate reanalysis features
X_names_ERA5 = ['ERA5_SWIN', 'ERA5_LWIN', 'ERA5_TA', 'ERA5_TS', 'ERA5_U', 'ERA5_V', 'ERA5_P', 'ERA5_PA', 'ERA5_VPD', 'ERA5_iWUE'] # Shortwave/Longwave In radiation, Air/Surface Temp, Wind components, Precipitation, Surface Pressure, VPD, and Intrinsic Water Use Efficiency

# CFSV2-specific climate reanalysis features
X_names_CFSV2 = ['CFSV2_TA', 'CFSV2_SWIN', 'CFSV2_SMC', 'CFSV2_U', 'CFSV2_V', 'CFSV2_P', 'CFSV2_VPD', 'CFSV2_iWUE'] # Air Temp, Shortwave In, Soil Moisture Content, Wind components, Precipitation, VPD, and Intrinsic Water Use Efficiency

# Dictionary to define different feature ensembles (combinations of remote sensing and climate data)
# Key is the ensemble name, value is the list of features (X_names)
# You can add more combinations if you like
X_names_dict = {
    # MODIS Normalized Difference Vegetation Index (NIRv) + Common + ERA5 features
    'MODIS-NIRv-ERA5': ['MODIS_NIRv'] + X_names_common + X_names_ERA5,
    # MODIS NIRv + ERA5 features (No common features like IGBP, Lat, DoY)
    'MODIS-NIRv-ERA5-NT': ['MODIS_NIRv'] + X_names_ERA5,
    # MODIS NIRv + Year + ERA5 features
    'MODIS-NIRv-ERA5-WY': ['MODIS_NIRv'] + ['Year'] + X_names_ERA5,
    # MODIS Enhanced Vegetation Index (EVI) + Common + ERA5 features
    'MODIS-EVI2-ERA5': ['MODIS_EVI'] + X_names_common + X_names_ERA5,
    # MODIS Normalized Difference Vegetation Index (NDVI) + Common + ERA5 features
    'MODIS-NDVI-ERA5': ['MODIS_NDVI'] + X_names_common + X_names_ERA5,
    # AVHRR EVI + Common + ERA5 features
    'AVHRR-EVI2-ERA5': ['AVHRR_EVI'] + X_names_common + X_names_ERA5,
    # AVHRR NDVI + Common + ERA5 features
    'AVHRR-NDVI-ERA5': ['AVHRR_NDVI'] + X_names_common + X_names_ERA5,
    # AVHRR NIRv + Common + ERA5 features
    'AVHRR-NIRv-ERA5': ['AVHRR_NIRv'] + X_names_common + X_names_ERA5,
    # MODIS NIRv + Common + CFSV2 features
    'MODIS-NIRv-CFSV2': ['MODIS_NIRv'] + X_names_common + X_names_CFSV2,
    # Solar-Induced Fluorescence (SIF) from GOME-2 + Common + ERA5 features
    'GOME-2-SIF-ERA5': ['GOME-2-SIF'] + X_names_common + X_names_ERA5,
    # SIF from GOSAT (755 nm band) + Common + ERA5 features
    'GOSAT-755-SIF-ERA5': ['GOSAT-755-SIF'] + X_names_common + X_names_ERA5,
    # SIF from GOSAT (772 nm band) + Common + ERA5 features
    'GOSAT-772-SIF-ERA5': ['GOSAT-772-SIF'] + X_names_common + X_names_ERA5,
    # Column-integrated SIF (CSIF) from OCO-2 + Common + ERA5 features
    'OCO-2-CSIF-ERA5': ['OCO-2-CSIF'] + X_names_common + X_names_ERA5,
}


# Select the specific ensemble to be used for training
ensemble_name = 'MODIS-NIRv-ERA5'
X_names = X_names_dict[ensemble_name] # Get the list of feature names for the selected ensemble


# Select the target variable (dependent variable Y) for prediction
# Options are ['GPP', 'RECO', 'NEE', 'H', 'LE'] (Gross Primary Production, Ecosystem Respiration, Net Ecosystem Exchange, Sensible Heat, Latent Heat)
y_name = ['GPP']

# --- Leave-One-Out Cross-Validation (LOOCV) ---
# Iterate through each unique flux tower site ('ID') to use it as the test set once.
df_validation_metrics = [] # List to store performance metrics for each LOOCV fold
for test_id in tqdm(df_training.index.get_level_values('ID').drop_duplicates()):
    
    # Identify all unique site IDs
    training_ids = list(df_training.index.get_level_values('ID').drop_duplicates())
    # Remove the current test site ID from the list to get the training sites
    training_ids.pop(training_ids.index(test_id))

    # Get the training data (X and y) by selecting all data *except* the test_id site
    X_training = df_training.loc[df_training.index.get_level_values('ID').isin(training_ids), X_names]
    y_training = df_training.loc[df_training.index.get_level_values('ID').isin(training_ids), y_name]

    # Get the test data (X and y) by selecting only the data for the current test_id site
    X_test = df_training.loc[df_training.index.get_level_values('ID') == test_id, X_names]
    y_test = df_training.loc[df_training.index.get_level_values('ID') == test_id, y_name].copy() # Use .copy() to avoid SettingWithCopyWarning

    # --- Model Training Configuration ---
    # Select the Machine Learning model
    model_name = 'DF21' # 'DF21' for Deep Forest 21, 'XGB' for XGBoost
    # Choose whether to use GPU acceleration
    GPU_mode = 0 # 1 for using GPU, 0 for not
    
    # Hyperparameter settings (None means default or auto-tuning will be used in pipelines.train_ml)
    xgb_params_user = None # User-specified parameters for XGBoost optimization
    df21_params_user = None  # User-specified parameters for Deep Forest 21 optimization

    # Section for optional automatic hyperparameter fine-tuning using GridSearchCV
    # Comment out this section if you'd like to use the default/user-specified parameters
    '''
    # Example of user parameters for fine tuning
    param_grid = {
        'n_estimators': [50, 100],  # Number of boosting rounds (trees)
        'max_depth': [3, 5],        # Maximum depth of a tree
        'learning_rate': [0.01, 0.1], # Step size shrinkage used to prevent overfitting
        'subsample': [0.8, 1.0],      # Fraction of samples used for training each tree
        'colsample_bytree': [0.8, 1.0] # Fraction of features used for training each tree
    }

    # Set up and run GridSearchCV
    grid_search = GridSearchCV(
        estimator=regr, # Need a base estimator here, e.g., XGBRegressor()
        param_grid=param_grid,
        scoring='neg_mean_squared_error', # GridSearchCV maximizes the score, so we use 'neg_mean_squared_error' for minimizing MSE
        cv=3,                             # 3-fold cross-validation
        verbose=2,                        # Prints progress
        n_jobs=-1                         # Use all available cores for parallel processing
    )

    # Starting GridSearchCV
    grid_search.fit(X_training, y_training)


    # Get the Best Hyperparameters found on the training set:
    best_params = grid_search.best_params_
    '''

    # Train the machine learning model using the defined pipeline
    regr = pipelines.train_ml(X_training, y_training, model_name = model_name, gpu = GPU_mode, xgb_params_user = xgb_params_user, df21_params_user = df21_params_user)
    
    # --- Prediction and Validation ---
    # Rename the actual (truth) column for clarity
    y_test.columns = ['Truth']
    # Generate predictions on the test set using the trained model
    y_test['Prediction'] = np.array(regr.predict(X_test))

    # Get validation metrics: r2, RMSE, MBE, etc.
    validation_metrics = pipelines.get_metrics(y_test, truth = 'Truth', pred = 'Prediction', return_dict = True)
    # Calculate and add the mean of the actual values for context
    validation_metrics['Mean'] = y_test['Truth'].mean()
    # Set the site ID as the index for the metrics
    validation_metrics.index = [test_id]
    # Store the metrics for this LOOCV fold
    df_validation_metrics.append(validation_metrics)
    
# Combine all validation metrics into a single DataFrame
df_validation_metrics = pd.concat(df_validation_metrics)

# --- Final Model Training ---
# Train the final model using the entire training dataset (all sites)
# This model will be used for generating the gridded prediction product.
regr = pipelines.train_ml(df_training[X_names], df_training[y_name], model_name = model_name, gpu = GPU_mode, xgb_params_user = xgb_params_user, df21_params_user = df21_params_user)

# Save the trained final model using pickle
model_savefile = project_directory.joinpath(f'1_models/{ensemble_name}/{y_name[0]}.pth')
# Ensure the directory exists before saving
model_savefile.parent.mkdir(parents=True, exist_ok=True)
if not model_savefile.exists():
    try:
        with open(model_savefile, 'wb') as file:
            pickle.dump(regr, file)
        print(f"Object successfully saved to: {model_savefile}")
    except Exception as e:
        print(f"An error occurred during pickling: {e}")


## Generate Ensemble Products (Gridded Prediction)

# --- Model Loading ---
# Redefine model and flux name in case this section is run independently
ensemble_name = 'MODIS-NIRv-ERA5'
flux_name = ['GPP']

# Load the previously trained final model from the saved pickle file
try:
    with open(project_directory.joinpath(f'1_models/{ensemble_name}/{flux_name[0]}.pth'), 'rb') as file:
        regr = pickle.load(file)
except FileNotFoundError:
    print(f"Error: Model file not found at {project_directory.joinpath(f'1_models/{ensemble_name}/{flux_name[0]}.pth')}. Ensure the training section was run successfully.")


# ------------------------------------------------------------------------------

## Load Gridded Data for Prediction

year = 2020
# Load remote sensing data (e.g., MODIS, AVHRR, GOSAT, or OCO-2)
satellite = xr.open_dataset(project_directory.joinpath(f'0_input/1_griddata/MODIS/monthly-MODIS-{year}-025deg.nc'))
# Load climate reanalysis data (e.g., ERA5 or CFSV2)
climate = xr.open_dataset(project_directory.joinpath(f'0_input/1_griddata/ERA5/monthly-ERA5-{year}.nc'))
# Load static land cover data
IGBP = xr.open_dataset(project_directory.joinpath('0_input/1_griddata/IGBP/MODIS-IGBP-01deg.nc'))
# Load atmospheric CO2 data
CO2 = xr.open_dataset(project_directory.joinpath(f'0_input/1_griddata/CO2/CO2-{year}.nc'))

# Example of necessary data pre-processing (unit conversions, variable calculation)
# Do remember processing satellite and climate data to match the units/definitions used in training
'''
climate['PA'] = climate['PA'] / 100 # Convert pressure from hPa to Pa
climate['T2m'] = climate['T2m'] - 273.15 # Convert 2m Temperature from Kelvin to degC
climate['D2m'] = climate['D2m'] - 273.15 # Convert 2m Dewpoint Temperature from Kelvin to degC
climate['Tsoil1'] = climate['Tsoil1'] - 273.15 # Convert Soil Temperature from Kelvin to degC

# calc VPD (Vapor Pressure Deficit)
# VPD = saturation_vapor_pressure(Air Temp) - saturation_vapor_pressure(Dewpoint Temp)
climate['VPD'] = saturation_vapor_pressure(climate['T2m']) - saturation_vapor_pressure(climate['D2m'])

# calc EVI2, NDVI, NIRv etc.
# Example: EVI2 calculation (satellite is assumed to be the variable holding the EVI2 data after calculation)
# satellite = 2.5 * (satellite['B2'] - satellite['B1']) / (satellite['B2'] + 2.4 * satellite['B1'] + 1)
'''

# Regrid all gridded data to a common grid (using the IGBP grid as the reference) and merge them
# This ensures all variables have the same 'latitude' and 'longitude' coordinates
nc = xr.merge([
    # Interpolate satellite and climate data to the IGBP grid
    satellite.interp(latitude = IGBP.latitude, longitude = IGBP.longitude), 
    climate.interp(latitude = IGBP.latitude, longitude = IGBP.longitude), 
    # IGBP and CO2 data (assuming CO2 is already on the IGBP grid or interpolation is not needed for it)
    IGBP.interp(latitude = IGBP.latitude, longitude = IGBP.longitude), 
    CO2.interp(latitude = IGBP.latitude, longitude = IGBP.longitude)
])

# --- Calculate Derived Features ---

# Calculate the partial pressure of CO2 (Ca)
ca = photosynthesis.calc_co2_to_ca(nc['CO2'], nc['PA']) # Requires CO2 concentration and atmospheric pressure (PA)

# Calculate Intrinsic Water Use Efficiency (iWUE)
# The 'wang17' limitation factor method is used.
_, iwue = photosynthesis.calc_light_water_use_efficiency(nc['TA'], nc['PA'], ca, nc['VPD'], True, False, limitation_factors = 'wang17')
nc['iWUE'] = iwue

# Calculate Day of Year (DoY) and expand dimensions to match grid
nc['doy'] = nc.time.dt.dayofyear.expand_dims({'latitude': nc.latitude, 'longitude': nc.longitude})

# Calculate DoY sine and cosine components to capture seasonality while maintaining continuity
days_in_year = 365.25 # Use 365.25 to account for leap years over time
nc['DoY_sin'] = np.sin(2 * np.pi * nc['doy'] / days_in_year)
nc['DoY_cos'] = np.cos(2 * np.pi * nc['doy'] / days_in_year)

# --- Apply the Model to the Gridded Data ---

# Convert the xarray dataset to a pandas DataFrame, select only the required features (X_names), 
# predict using the trained model, and convert the result back to an xarray DataArray.
UFLUX_prediction = regr.predict(nc.to_dataframe()[X_names]).reset_index().set_index(['latitude', 'longitude', 'time']).to_xarray()

# Save the final gridded prediction product to a NetCDF file
savefile = project_directory.joinpath(f'3_output_products/{ensemble_name}/monthly-{flux_name[0]}-{year}.nc')
# Ensure the output directory exists
savefile.parent.mkdir(parents=True, exist_ok=True)
UFLUX_prediction.to_netcdf(savefile)
