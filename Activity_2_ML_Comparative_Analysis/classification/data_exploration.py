"""
ML Comparative Analysis - Classification Module: Data Exploration (EDA)
Dataset: Global Disaster Classification Dataset (14,644 worldwide disaster records)
Application: Real-World Multi-Class Disaster Subgroup & Climate Hazard Prediction
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
DATASET_PATH = os.path.join(PROJECT_ROOT, 'datasets', 'global', 'global_classification_dataset.csv')
VIZ_DIR = os.path.join(PROJECT_ROOT, 'visualizations', 'classification')
os.makedirs(VIZ_DIR, exist_ok=True)

def explore_classification_dataset():
    print("=" * 80)
    print("CLASSIFICATION WORKFLOW: GLOBAL DISASTER DATASET EXPLORATION")
    print("=" * 80)
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
        
    df = pd.read_csv(DATASET_PATH)
    
    print(f"\n1. DATASET SHAPE: {df.shape[0]:,} records, {df.shape[1]} columns")
    print("\n2. FIRST 5 RECORDS:")
    print(df.head())
    print("\n3. LAST 5 RECORDS:")
    print(df.tail())
    
    print("\n4. COLUMN NAMES & DATA TYPES:")
    print(df.dtypes.to_string())
    
    print("\n5. STATISTICAL SUMMARY (Numerical Columns):")
    num_cols = df.select_dtypes(include=[np.number]).columns
    print(df[num_cols].describe().T[['count', 'mean', 'std', 'min', '50%', 'max']])
    
    print("\n6. TARGET VARIABLE IDENTIFICATION & COMPARATIVE CHOICES:")
    print("   Candidate Targets in the Global Dataset:")
    print("   1. 'Disaster_Subgroup_Cleaned' (Primary): Multi-class scientific environmental category")
    print("      Classes: Hydrological, Meteorological, Biological, Geophysical, Climatological")
    print("   2. 'Disaster_Severity_Class': 4-tier impact class (Minor, Moderate, Severe, Catastrophic)")
    print("   3. 'High_Severity_Flag': Binary target (0 = Minor/Moderate, 1 = Severe/Catastrophic)")
    print("\n   Primary Selected Target: 'Disaster_Subgroup_Cleaned'")
    print("   Justification: Predicting the environmental mechanism of an impending hazard from")
    print("   spatio-temporal onset features, continent, seasonality, and early duration is the")
    print("   core multi-class classification challenge in real-world disaster management.")
    
    # Check Class Distribution
    target_var = 'Disaster_Subgroup_Cleaned'
    class_dist = df[target_var].value_counts()
    print(f"\n7. CLASS DISTRIBUTION ({target_var}):")
    for cls, count in class_dist.items():
        print(f"   - {cls:20s}: {count:5,d} records ({count/len(df)*100:5.2f}%)")
        
    # -------------------------------------------------------------
    # EDA VISUALIZATIONS
    # -------------------------------------------------------------
    print("\n8. GENERATING EDA VISUALIZATIONS...")
    
    # Visualization 1: Target Class Distribution Chart
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=class_dist.index, y=class_dist.values, hue=class_dist.index, palette='viridis', legend=False)
    plt.title('Multi-Class Target Distribution: Disaster Subgroup (Global 1970 - 2021)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Disaster Subgroup', fontsize=11)
    plt.ylabel('Recorded Disaster Occurrences', fontsize=11)
    for i, v in enumerate(class_dist.values):
        ax.text(i, v + 100, f"{v:,}", ha='center', fontweight='bold', fontsize=10)
    plt.savefig(os.path.join(VIZ_DIR, 'target_class_distribution.png'), dpi=300)
    plt.close()
    
    # Visualization 2: Feature Histograms
    plt.figure(figsize=(12, 5))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.histplot(df['Disaster_Year'], bins=25, ax=axes[0], color='#1f77b4', kde=True)
    axes[0].set_title('Temporal Distribution (1970 - 2021)', fontweight='bold')
    
    sns.histplot(df['Disaster_Month'], bins=12, ax=axes[1], color='#2ca02c')
    axes[1].set_title('Monthly Seasonal Onset', fontweight='bold')
    
    sns.histplot(np.log1p(df['Disaster_Duration_Days']), bins=30, ax=axes[2], color='#d62728', kde=True)
    axes[2].set_title('Log(Disaster Duration Days + 1)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'feature_histograms.png'), dpi=300)
    plt.close()
    
    # Visualization 3: Correlation Matrix Heatmap
    plt.figure(figsize=(9, 7))
    num_corr_cols = ['Disaster_Year', 'Disaster_Month', 'Disaster_Duration_Days', 
                     'Country_Historical_Disaster_Count', 'Climate_Related_Flag', 
                     'Compound_Disaster_Flag', 'International_Assistance_Flag', 'High_Severity_Flag']
    corr_mat = df[num_corr_cols].corr()
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    sns.heatmap(corr_mat, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={'shrink': 0.8})
    plt.title('Correlation Matrix of Quantitative Disaster Predictors', fontsize=13, fontweight='bold', pad=12)
    plt.savefig(os.path.join(VIZ_DIR, 'correlation_heatmap.png'), dpi=300)
    plt.close()
    
    # Visualization 4: Subgroup vs Continent Crosstab Heatmap
    plt.figure(figsize=(10, 6))
    ct = pd.crosstab(df['Continent_Cleaned'], df[target_var], normalize='index') * 100
    sns.heatmap(ct, annot=True, fmt='.1f', cmap='YlGnBu', cbar_kws={'label': '% of Continental Disasters'})
    plt.title('Disaster Subgroup Occurrence Proportions Across Continents (%)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Disaster Subgroup', fontsize=11)
    plt.ylabel('Continent', fontsize=11)
    plt.savefig(os.path.join(VIZ_DIR, 'subgroup_by_continent.png'), dpi=300)
    plt.close()
    
    print(f"   Visualizations saved successfully in: {VIZ_DIR}")
    print("=" * 80)
    return df

if __name__ == '__main__':
    explore_classification_dataset()
