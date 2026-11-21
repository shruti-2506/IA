"""
ML Comparative Analysis - Regression Module: Model Training & Hyperparameter Tuning
Implements 6 Regression Algorithms:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Support Vector Regression (SVR)
5. K-Nearest Neighbors Regressor (KNN)
6. Multi-Layer Perceptron Regressor (MLP)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
RESULTS_DIR = os.path.join(CURRENT_DIR, 'regression_results')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_tune_regression_models():
    print("=" * 80)
    print("REGRESSION WORKFLOW: MODEL TRAINING & HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    
    # 1. Load Preprocessed Arrays
    x_train_path = os.path.join(RESULTS_DIR, 'X_train_scaled.npy')
    if not os.path.exists(x_train_path):
        from preprocessing import preprocess_meteo_data
        preprocess_meteo_data(sample_size=15000)
        
    X_train = np.load(os.path.join(RESULTS_DIR, 'X_train_scaled.npy'))
    X_test = np.load(os.path.join(RESULTS_DIR, 'X_test_scaled.npy'))
    y_train = np.load(os.path.join(RESULTS_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(RESULTS_DIR, 'y_test.npy'))
    
    print(f"Loaded Training Data: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"Loaded Testing Data:  {X_test.shape[0]:,} samples")
    
    # Define Baseline Models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=12),
        'Support Vector Regression': SVR(C=10.0, epsilon=0.1, kernel='rbf', max_iter=2500),
        'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=7, weights='distance', n_jobs=-1),
        'Multi-Layer Perceptron': MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=250, random_state=42, early_stopping=True)
    }
    
    baseline_records = []
    trained_models = {}
    
    # 2. Train Each Baseline Model & Measure Wall-Clock Time
    print("\n--- Training 6 Baseline Regression Models ---")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\nTraining: {name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        
        t0_inf = time.time()
        y_pred = model.predict(X_test)
        test_time = time.time() - t0_inf
        
        # Cross-Validation R2 score on training set
        cv_scores = cross_val_score(model, X_train[:3000], y_train[:3000], cv=kfold, scoring='r2', n_jobs=-1)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Save baseline model
        slug = name.lower().replace(' ', '_').replace('-', '_')
        model_save_path = os.path.join(MODELS_DIR, f"{slug}.joblib")
        joblib.dump(model, model_save_path)
        trained_models[name] = model
        
        record = {
            'Algorithm': name,
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4),
            'R2_Score': round(r2, 4),
            'CV_R2_Mean': round(cv_scores.mean(), 4),
            'CV_R2_Std': round(cv_scores.std(), 4),
            'Train_Time_Sec': round(train_time, 4),
            'Test_Time_Sec': round(test_time, 4),
            'Model_File': f"{slug}.joblib"
        }
        baseline_records.append(record)
        print(f"  -> RMSE: {rmse:.4f}°C | MAE: {mae:.4f}°C | R²: {r2:.4f} | Train Time: {train_time:.2f}s")
        
    df_metrics = pd.DataFrame(baseline_records)
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.csv')
    df_metrics.to_csv(metrics_path, index=False)
    print(f"\nSaved Baseline Metrics Table to: {metrics_path}")

    # 3. Hyperparameter Optimization on Primary Model (Random Forest & Decision Tree)
    print("\n--- Hyperparameter Optimization (GridSearchCV) ---")
    print("Primary Model Selected for Tuning: Random Forest Regressor")
    print("Justification: Ensemble learning captures nonlinear atmospheric lapse rates, solar radiation diurnal cycles, and humidity-wind couplings.")
    
    rf_param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid=rf_param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    t0_tune = time.time()
    # Use subset for fast and effective parameter exploration
    grid_search.fit(X_train[:5000], y_train[:5000])
    tune_time = time.time() - t0_tune
    
    best_rf = grid_search.best_estimator_
    print(f"Best Hyperparameters Identified in {tune_time:.1f}s:")
    print(grid_search.best_params_)
    
    # Evaluate Optimized Model on Full Test Set
    best_rf.fit(X_train, y_train)
    y_pred_opt = best_rf.predict(X_test)
    
    mse_opt = mean_squared_error(y_test, y_pred_opt)
    rmse_opt = np.sqrt(mse_opt)
    mae_opt = mean_absolute_error(y_test, y_pred_opt)
    r2_opt = r2_score(y_test, y_pred_opt)
    
    # Save optimized model as best model
    best_model_path = os.path.join(MODELS_DIR, 'best_model.joblib')
    joblib.dump(best_rf, best_model_path)
    joblib.dump(best_rf, os.path.join(MODELS_DIR, 'random_forest_tuned.joblib'))
    
    # Before vs After Optimization Comparison
    baseline_rf = df_metrics[df_metrics['Algorithm'] == 'Random Forest'].iloc[0]
    tuning_comparison = pd.DataFrame([
        {
            'Stage': 'Baseline Default',
            'Parameters': "{'n_estimators': 100, 'max_depth': 12}",
            'MSE': baseline_rf['MSE'],
            'RMSE': baseline_rf['RMSE'],
            'MAE': baseline_rf['MAE'],
            'R2_Score': baseline_rf['R2_Score']
        },
        {
            'Stage': 'Hyperparameter Optimized',
            'Parameters': str(grid_search.best_params_),
            'MSE': round(mse_opt, 4),
            'RMSE': round(rmse_opt, 4),
            'MAE': round(mae_opt, 4),
            'R2_Score': round(r2_opt, 4)
        }
    ])
    tuning_path = os.path.join(RESULTS_DIR, 'tuning_comparison.csv')
    tuning_comparison.to_csv(tuning_path, index=False)
    print(f"Saved Tuning Comparison to: {tuning_path}")
    print(tuning_comparison)
    
    print("\n" + "=" * 80)
    print("REGRESSION MODEL TRAINING & TUNING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return df_metrics, tuning_comparison

if __name__ == '__main__':
    train_and_tune_regression_models()
