"""
ML Comparative Analysis - Regression Module: Data Exploration (EDA)
Dataset: Open-Meteo Weather Dataset (Pune, India - 18.52N, 73.87E, 561m)
Application: Real-Time Weather Temperature Forecasting
Author: AI/ML Engineering Team
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATASET_PATH = os.path.join(PROJECT_ROOT, 'datasets', 'meteo', 'open-meteo-18.52N73.87E561m.csv')
VIZ_DIR = os.path.join(PROJECT_ROOT, 'visualizations', 'regression')
os.makedirs(VIZ_DIR, exist_ok=True)

def load_and_explore():
    print("=" * 80)
    print("REGRESSION WORKFLOW: OPEN-METEO WEATHER DATASET EXPLORATION")
    print("=" * 80)
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
        
    # Read CSV skipping initial 3 lines of API metadata
    df = pd.read_csv(DATASET_PATH, skiprows=3, low_memory=False)
    
    # Clean column names (strip whitespace and brackets for clean scripting)
    df.columns = [c.strip() for c in df.columns]
    
    print(f"\n1. DATASET SHAPE: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print("\n2. FIRST 5 RECORDS:")
    print(df.head())
    print("\n3. LAST 5 RECORDS:")
    print(df.tail())
    
    print("\n4. COLUMN NAMES & DATA TYPES:")
    print(df.dtypes.to_string())
    
    print("\n5. STATISTICAL SUMMARY (Key Numerical Features):")
    core_cols = [
        'temperature_2m (°C)', 'relative_humidity_2m (%)', 'dew_point_2m (°C)',
        'apparent_temperature (°C)', 'precipitation (mm)', 'surface_pressure (hPa)',
        'cloud_cover (%)', 'wind_speed_10m (km/h)', 'vapour_pressure_deficit (kPa)',
        'soil_temperature_0_to_7cm (°C)'
    ]
    avail_cols = [c for c in core_cols if c in df.columns]
    print(df[avail_cols].describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']])
    
    print("\n6. TARGET VARIABLE SELECTION & JUSTIFICATION:")
    target_var = 'temperature_2m (°C)'
    print(f"   Selected Target Variable: '{target_var}'")
    print("   Justification:")
    print("   - Ambient 2-meter air temperature is the fundamental thermodynamic state variable")
    print("     governing tropospheric dynamics, heatwave warnings, agricultural planning, and")
    print("     urban energy demand.")
    print("   - It is a strictly continuous, real-valued variable exhibiting rich diurnal,")
    print("     synoptic, and seasonal variations, making it ideal for benchmarking linear,")
    print("     kernel, tree-ensemble, and neural regression algorithms.")

    # -------------------------------------------------------------
    # EDA VISUALIZATIONS
    # -------------------------------------------------------------
    print("\n7. GENERATING EDA VISUALIZATIONS...")
    
    # Visualization 1: Target Variable Distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(df[target_var].dropna(), kde=True, color='#1f77b4', bins=40)
    plt.title('Target Variable Distribution: 2m Temperature (°C) - Pune, India', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Temperature (°C)', fontsize=11)
    plt.ylabel('Hourly Observations', fontsize=11)
    plt.savefig(os.path.join(VIZ_DIR, 'target_distribution.png'), dpi=300)
    plt.close()
    
    # Visualization 2: Key Features Histograms
    plt.figure(figsize=(14, 10))
    df[avail_cols[:6]].hist(bins=30, figsize=(14, 10), color='#2ca02c', edgecolor='black', alpha=0.8)
    plt.suptitle('Histograms of Core Meteorological Predictors', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'histograms.png'), dpi=300)
    plt.close()
    
    # Visualization 3: Boxplots (Outlier Inspection)
    plt.figure(figsize=(14, 6))
    scaled_box = (df[avail_cols[:6]] - df[avail_cols[:6]].mean()) / df[avail_cols[:6]].std()
    sns.boxplot(data=scaled_box, palette='Set2')
    plt.title('Standardized Boxplots of Core Meteorological Features (Outlier Inspection)', fontsize=13, fontweight='bold', pad=12)
    plt.ylabel('Z-Score Standardized Values', fontsize=11)
    plt.xticks(rotation=20)
    plt.savefig(os.path.join(VIZ_DIR, 'boxplots.png'), dpi=300)
    plt.close()
    
    # Visualization 4: Correlation Matrix Heatmap
    plt.figure(figsize=(11, 9))
    corr_matrix = df[avail_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={'shrink': 0.8})
    plt.title('Correlation Heatmap of Meteorological Features with Temperature', fontsize=14, fontweight='bold', pad=12)
    plt.savefig(os.path.join(VIZ_DIR, 'correlation_heatmap.png'), dpi=300)
    plt.close()
    
    # Visualization 5: Scatter Plots (Predictor vs Target)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sample_df = df.sample(min(3000, len(df)), random_state=42)
    
    sns.scatterplot(data=sample_df, x='dew_point_2m (°C)', y=target_var, ax=axes[0], color='#ff7f0e', alpha=0.5)
    axes[0].set_title('Dew Point vs. Temperature', fontweight='bold')
    
    sns.scatterplot(data=sample_df, x='relative_humidity_2m (%)', y=target_var, ax=axes[1], color='#1f77b4', alpha=0.5)
    axes[1].set_title('Relative Humidity vs. Temperature', fontweight='bold')
    
    sns.scatterplot(data=sample_df, x='vapour_pressure_deficit (kPa)', y=target_var, ax=axes[2], color='#d62728', alpha=0.5)
    axes[2].set_title('Vapour Pressure Deficit vs. Temperature', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'scatter_plots.png'), dpi=300)
    plt.close()
    
    print(f"   Visualizations saved successfully in: {VIZ_DIR}")
    print("=" * 80)
    return df

if __name__ == '__main__':
    load_and_explore()
