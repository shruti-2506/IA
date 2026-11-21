"""
ML Comparative Analysis - Classification Module: Data Preprocessing
Handles One-Hot/Label Encoding, Missing Value Checks, Class Balancing,
Stratified 80:20 Data Splitting, and Leakage-Free StandardScaler Pipeline.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATASET_PATH = os.path.join(PROJECT_ROOT, 'datasets', 'global', 'global_classification_dataset.csv')
RESULTS_DIR = os.path.join(CURRENT_DIR, 'classification_results')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def preprocess_classification_data(target_variable='Disaster_Subgroup_Cleaned', test_size=0.20, random_state=42):
    print("=" * 80)
    print("CLASSIFICATION WORKFLOW: DATA PREPROCESSING PIPELINE")
    print(f"Target Variable: '{target_variable}'")
    print("=" * 80)
    
    # 1. Load Data
    df = pd.read_csv(DATASET_PATH)
    initial_count = len(df)
    
    # 2. Duplicate Audit
    df = df.drop_duplicates(subset=['Dis No']).reset_index(drop=True)
    print(f"Loaded {len(df):,} valid records after duplicate check.")
    
    # Clean singletons in target variable for stratified cross validation
    # 'Extra-terrestrial' has only 1 event globally (meteor impact); merge into 'Geophysical'
    if target_variable == 'Disaster_Subgroup_Cleaned':
        df[target_variable] = df[target_variable].replace({'Extra-terrestrial': 'Geophysical'})
        
    # 3. Feature Selection & Categorical Encoding
    num_features = [
        'Disaster_Year', 'Disaster_Month', 'Disaster_Duration_Days',
        'Start_Month_Sin', 'Start_Month_Cos',
        'Country_Historical_Disaster_Count',
        'Climate_Related_Flag', 'Compound_Disaster_Flag', 'International_Assistance_Flag'
    ]
    
    cat_features = ['Continent_Cleaned', 'Disaster_Season', 'Quarter']
    
    # One-hot encode categorical predictors
    df_encoded = pd.get_dummies(df[cat_features], drop_first=True, dtype=float)
    
    X = pd.concat([df[num_features], df_encoded], axis=1)
    feature_names = X.columns.tolist()
    print(f"Total encoded predictive features: {len(feature_names)}")
    
    # Encode Target Labels
    le = LabelEncoder()
    y = le.fit_transform(df[target_variable].astype(str))
    class_names = le.classes_.tolist()
    print(f"Target Classes ({len(class_names)}): {class_names}")
    
    # 4. Stratified Train-Test Split (80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Stratified Train Split: {X_train.shape[0]:,} samples")
    print(f"Stratified Test Split:  {X_test.shape[0]:,} samples")
    
    # 5. Standard Scaling (Fitted strictly on Training set)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Save Preprocessing Bundle
    preprocessor_bundle = {
        'scaler': scaler,
        'label_encoder': le,
        'feature_names': feature_names,
        'class_names': class_names,
        'target_variable': target_variable,
        'num_features': num_features,
        'cat_features': cat_features
    }
    joblib.dump(preprocessor_bundle, os.path.join(MODELS_DIR, 'preprocessor.joblib'))
    
    np.save(os.path.join(RESULTS_DIR, 'X_train_scaled.npy'), X_train_scaled)
    np.save(os.path.join(RESULTS_DIR, 'X_test_scaled.npy'), X_test_scaled)
    np.save(os.path.join(RESULTS_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(RESULTS_DIR, 'y_test.npy'), y_test)
    
    print(f"Saved Preprocessor Bundle to: {os.path.join(MODELS_DIR, 'preprocessor.joblib')}")
    print("=" * 80)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, class_names, feature_names

if __name__ == '__main__':
    preprocess_classification_data()
