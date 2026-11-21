"""
ML Comparative Analysis - Regression Module: Data Preprocessing
Handles Data Cleaning, Imputation, Feature Engineering, Outlier Management,
Strict 80:20 Data Splitting, and Leakage-Free StandardScaler Pipeline.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATASET_PATH = os.path.join(PROJECT_ROOT, 'datasets', 'meteo', 'open-meteo-18.52N73.87E561m.csv')
RESULTS_DIR = os.path.join(CURRENT_DIR, 'regression_results')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def preprocess_meteo_data(sample_size=None, test_size=0.20, random_state=42):
    print("=" * 80)
    print("REGRESSION WORKFLOW: DATA PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # 1. Load Data
    print(f"Loading raw data from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH, skiprows=3, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    initial_rows = len(df)
    
    # 2. Duplicate Removal
    df = df.drop_duplicates(subset=['time']).reset_index(drop=True)
    print(f"Duplicates removed: {initial_rows - len(df)} rows. Current count: {len(df):,}")
    
    # 3. Time Feature Extraction & Cyclical Encoding
    df['datetime'] = pd.to_datetime(df['time'])
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['day_of_year'] = df['datetime'].dt.dayofyear
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    target_col = 'temperature_2m (°C)'
    
    # 4. Feature Selection: Retain core meteorological columns with high information density
    selected_features = [
        'apparent_temperature (°C)',
        'relative_humidity_2m (%)',
        'dew_point_2m (°C)',
        'surface_pressure (hPa)',
        'cloud_cover (%)',
        'cloud_cover_low (%)',
        'cloud_cover_mid (%)',
        'cloud_cover_high (%)',
        'wind_speed_10m (km/h)',
        'wind_direction_10m (°)',
        'wind_gusts_10m (km/h)',
        'et0_fao_evapotranspiration (mm)',
        'vapour_pressure_deficit (kPa)',
        'soil_temperature_0_to_7cm (°C)',
        'soil_moisture_0_to_7cm (m³/m³)',
        'is_day ()',
        'hour_sin', 'hour_cos',
        'month_sin', 'month_cos'
    ]
    
    # Verify presence
    selected_features = [f for f in selected_features if f in df.columns]
    print(f"Number of selected predictive features: {len(selected_features)}")
    
    # 5. Missing Value Imputation (Linear Interpolation + Ffill/Bfill for continuous weather time series)
    df[selected_features] = df[selected_features].interpolate(method='linear').ffill().bfill()
    df[target_col] = df[target_col].interpolate(method='linear').ffill().bfill()
    
    # 6. Outlier Handling: Capping physically impossible extremes
    # Ambient temperature in Pune historically stays within [5°C, 48°C]
    df[target_col] = df[target_col].clip(5.0, 50.0)
    df['relative_humidity_2m (%)'] = df['relative_humidity_2m (%)'].clip(0.0, 100.0)
    
    # 7. Optional sampling for swift model tuning/cross-validation if specified
    if sample_size and sample_size < len(df):
        print(f"Using representative sample of {sample_size:,} records for high-speed modeling...")
        df_sample = df.sample(n=sample_size, random_state=random_state).sort_values('datetime').reset_index(drop=True)
    else:
        df_sample = df.copy()
        
    X = df_sample[selected_features].copy()
    y = df_sample[target_col].copy()
    
    # 8. Train-Test Split (80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    print(f"Train Set Shape: {X_train.shape[0]:,} samples")
    print(f"Test Set Shape:  {X_test.shape[0]:,} samples")
    
    # 9. Feature Scaling (Fit strictly on training set to prevent data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 10. Save Preprocessing Artifacts
    preprocessor_bundle = {
        'scaler': scaler,
        'feature_names': selected_features,
        'target_name': target_col,
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'mean_target': float(y_train.mean()),
        'std_target': float(y_train.std())
    }
    joblib.dump(preprocessor_bundle, os.path.join(MODELS_DIR, 'preprocessor.joblib'))
    
    # Save arrays for training scripts
    np.save(os.path.join(RESULTS_DIR, 'X_train_scaled.npy'), X_train_scaled)
    np.save(os.path.join(RESULTS_DIR, 'X_test_scaled.npy'), X_test_scaled)
    np.save(os.path.join(RESULTS_DIR, 'y_train.npy'), y_train.values)
    np.save(os.path.join(RESULTS_DIR, 'y_test.npy'), y_test.values)
    
    print(f"Preprocessing bundle saved to: {os.path.join(MODELS_DIR, 'preprocessor.joblib')}")
    print("=" * 80)
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, selected_features, target_col

if __name__ == '__main__':
    preprocess_meteo_data()
