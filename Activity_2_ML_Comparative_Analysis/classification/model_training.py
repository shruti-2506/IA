"""
ML Comparative Analysis - Classification Module: Model Training & Tuning
Implements 6 Classification Algorithms:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors Classifier (KNN)
6. Multi-Layer Perceptron Classifier (MLP)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
RESULTS_DIR = os.path.join(CURRENT_DIR, 'classification_results')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_tune_classification_models():
    print("=" * 80)
    print("CLASSIFICATION WORKFLOW: MODEL TRAINING & HYPERPARAMETER TUNING")
    print("=" * 80)
    
    # 1. Load Preprocessed Data
    x_train_path = os.path.join(RESULTS_DIR, 'X_train_scaled.npy')
    if not os.path.exists(x_train_path):
        from preprocessing import preprocess_classification_data
        preprocess_classification_data()
        
    X_train = np.load(os.path.join(RESULTS_DIR, 'X_train_scaled.npy'))
    X_test = np.load(os.path.join(RESULTS_DIR, 'X_test_scaled.npy'))
    y_train = np.load(os.path.join(RESULTS_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(RESULTS_DIR, 'y_test.npy'))
    
    prep_bundle = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.joblib'))
    class_names = prep_bundle['class_names']
    
    print(f"Loaded Training Data: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"Loaded Testing Data:  {X_test.shape[0]:,} samples")
    print(f"Target Classes ({len(class_names)}): {class_names}")
    
    # Define Baseline Models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=12, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15, class_weight='balanced'),
        'Support Vector Machine': SVC(C=5.0, kernel='rbf', probability=True, random_state=42, max_iter=3000),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1),
        'Multi-Layer Perceptron': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=250, random_state=42, early_stopping=True)
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    baseline_records = []
    trained_models = {}
    
    print("\n--- Training 6 Baseline Classification Models ---")
    for name, model in models.items():
        print(f"\nTraining: {name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        
        t0_inf = time.time()
        y_pred = model.predict(X_test)
        test_time = time.time() - t0_inf
        
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        prec_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # 5-Fold Stratified CV Accuracy
        cv_acc = cross_val_score(model, X_train[:4000], y_train[:4000], cv=skf, scoring='accuracy', n_jobs=-1)
        
        slug = name.lower().replace(' ', '_')
        model_save_path = os.path.join(MODELS_DIR, f"{slug}.joblib")
        joblib.dump(model, model_save_path)
        trained_models[name] = model
        
        record = {
            'Algorithm': name,
            'Accuracy': round(acc, 4),
            'Precision_Macro': round(prec_macro, 4),
            'Recall_Macro': round(rec_macro, 4),
            'F1_Macro': round(f1_macro, 4),
            'Precision_Weighted': round(prec_weighted, 4),
            'Recall_Weighted': round(rec_weighted, 4),
            'F1_Weighted': round(f1_weighted, 4),
            'CV_Accuracy_Mean': round(cv_acc.mean(), 4),
            'CV_Accuracy_Std': round(cv_acc.std(), 4),
            'Train_Time_Sec': round(train_time, 4),
            'Test_Time_Sec': round(test_time, 4),
            'Model_File': f"{slug}.joblib"
        }
        baseline_records.append(record)
        print(f"  -> Accuracy: {acc*100:.2f}% | F1 (Macro): {f1_macro:.4f} | F1 (Weighted): {f1_weighted:.4f} | Time: {train_time:.2f}s")
        
    df_metrics = pd.DataFrame(baseline_records)
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.csv')
    df_metrics.to_csv(metrics_path, index=False)
    print(f"\nSaved Classification Baseline Metrics Table to: {metrics_path}")

    # 3. Hyperparameter Optimization on Primary Model (Random Forest)
    print("\n--- Hyperparameter Optimization (GridSearchCV) ---")
    print("Primary Model Selected for Tuning: Random Forest Classifier")
    print("Justification: Robust against class imbalance, handles mixed feature interactions, and minimizes out-of-fold generalization variance.")
    
    rf_param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [12, 16, 20],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=rf_param_grid,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    t0_tune = time.time()
    grid_search.fit(X_train[:6000], y_train[:6000])
    tune_time = time.time() - t0_tune
    
    best_rf = grid_search.best_estimator_
    print(f"Best Hyperparameters Identified in {tune_time:.1f}s:")
    print(grid_search.best_params_)
    
    # Train Best Estimator on Full Training Set
    best_rf.fit(X_train, y_train)
    y_pred_opt = best_rf.predict(X_test)
    
    acc_opt = accuracy_score(y_test, y_pred_opt)
    prec_opt = precision_score(y_test, y_pred_opt, average='weighted', zero_division=0)
    rec_opt = recall_score(y_test, y_pred_opt, average='weighted', zero_division=0)
    f1_opt = f1_score(y_test, y_pred_opt, average='weighted', zero_division=0)
    
    best_model_path = os.path.join(MODELS_DIR, 'best_model.joblib')
    joblib.dump(best_rf, best_model_path)
    joblib.dump(best_rf, os.path.join(MODELS_DIR, 'random_forest_tuned.joblib'))
    
    # Before vs After Comparison
    baseline_rf = df_metrics[df_metrics['Algorithm'] == 'Random Forest'].iloc[0]
    tuning_comparison = pd.DataFrame([
        {
            'Stage': 'Baseline Default',
            'Parameters': "{'n_estimators': 100, 'max_depth': 15, 'class_weight': 'balanced'}",
            'Accuracy': baseline_rf['Accuracy'],
            'Precision_Weighted': baseline_rf['Precision_Weighted'],
            'Recall_Weighted': baseline_rf['Recall_Weighted'],
            'F1_Weighted': baseline_rf['F1_Weighted']
        },
        {
            'Stage': 'Hyperparameter Optimized',
            'Parameters': str(grid_search.best_params_),
            'Accuracy': round(acc_opt, 4),
            'Precision_Weighted': round(prec_opt, 4),
            'Recall_Weighted': round(rec_opt, 4),
            'F1_Weighted': round(f1_opt, 4)
        }
    ])
    tuning_path = os.path.join(RESULTS_DIR, 'tuning_comparison.csv')
    tuning_comparison.to_csv(tuning_path, index=False)
    print(f"Saved Tuning Comparison to: {tuning_path}")
    print(tuning_comparison)
    
    print("\n" + "=" * 80)
    print("CLASSIFICATION MODEL TRAINING & TUNING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return df_metrics, tuning_comparison

if __name__ == '__main__':
    train_and_tune_classification_models()
