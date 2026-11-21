"""
ML Comparative Analysis - Regression Module: Evaluation & Visualization
Computes Metrics, Residual Distributions, Actual vs. Predicted Visualizations,
Model Performance Rankings, and Comparative Charts.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
RESULTS_DIR = os.path.join(CURRENT_DIR, 'regression_results')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
VIZ_DIR = os.path.join(PROJECT_ROOT, 'visualizations', 'regression')
os.makedirs(VIZ_DIR, exist_ok=True)

def evaluate_regression_models():
    print("=" * 80)
    print("REGRESSION WORKFLOW: PERFORMANCE EVALUATION & VISUALIZATION")
    print("=" * 80)
    
    # 1. Load Metrics and Test Data
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.csv')
    if not os.path.exists(metrics_path):
        from model_training import train_and_tune_regression_models
        train_and_tune_regression_models()
        
    df_metrics = pd.read_csv(metrics_path)
    X_test = np.load(os.path.join(RESULTS_DIR, 'X_test_scaled.npy'))
    y_test = np.load(os.path.join(RESULTS_DIR, 'y_test.npy'))
    
    best_model_path = os.path.join(MODELS_DIR, 'best_model.joblib')
    best_model = joblib.load(best_model_path)
    
    y_pred_best = best_model.predict(X_test)
    residuals = y_test - y_pred_best
    
    print("\n1. PERFORMANCE SUMMARY TABLE:")
    print(df_metrics[['Algorithm', 'MSE', 'RMSE', 'MAE', 'R2_Score', 'Train_Time_Sec']].sort_values(by='R2_Score', ascending=False).to_string(index=False))
    
    # 2. Performance Ranking
    df_ranked = df_metrics.sort_values(by='R2_Score', ascending=False).reset_index(drop=True)
    df_ranked['Rank'] = range(1, len(df_ranked) + 1)
    df_ranked.to_csv(os.path.join(RESULTS_DIR, 'model_ranking.csv'), index=False)
    
    top_model = df_ranked.iloc[0]
    print(f"\n2. TOP PERFORMING REGRESSION ALGORITHM: {top_model['Algorithm']}")
    print(f"   R² Score:  {top_model['R2_Score']:.4f}")
    print(f"   RMSE:      {top_model['RMSE']:.4f}°C")
    print(f"   MAE:       {top_model['MAE']:.4f}°C")
    
    # -------------------------------------------------------------
    # GENERATE VISUALIZATIONS
    # -------------------------------------------------------------
    print("\n3. GENERATING COMPARATIVE VISUALIZATIONS...")
    
    # Plot 1: R2 Score Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    colors = ['#2ca02c' if x == top_model['Algorithm'] else '#1f77b4' for x in df_ranked['Algorithm']]
    ax = sns.barplot(data=df_ranked, x='R2_Score', y='Algorithm', palette=colors, hue='Algorithm', legend=False)
    plt.title('Regression Algorithm Comparison: Coefficient of Determination (R² Score)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('R² Score (Higher is Better, Max = 1.0)', fontsize=11)
    plt.xlim(0, 1.05)
    for i, v in enumerate(df_ranked['R2_Score']):
        ax.text(v + 0.01, i, f"{v:.4f}", va='center', fontweight='bold', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'model_r2_comparison.png'), dpi=300)
    plt.close()
    
    # Plot 2: RMSE and MAE Comparison Grouped Bar Chart
    plt.figure(figsize=(11, 5))
    df_melt = pd.melt(df_ranked, id_vars=['Algorithm'], value_vars=['RMSE', 'MAE'], var_name='Metric', value_name='Error (°C)')
    sns.barplot(data=df_melt, x='Algorithm', y='Error (°C)', hue='Metric', palette=['#d62728', '#ff7f0e'])
    plt.title('Regression Error Comparison: RMSE vs. MAE (°C)', fontsize=13, fontweight='bold', pad=12)
    plt.ylabel('Prediction Error in °C (Lower is Better)', fontsize=11)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'model_rmse_mae_comparison.png'), dpi=300)
    plt.close()
    
    # Plot 3: Actual vs. Predicted Scatter (Best Model)
    plt.figure(figsize=(7, 7))
    sample_indices = np.random.choice(len(y_test), size=min(1500, len(y_test)), replace=False)
    plt.scatter(y_test[sample_indices], y_pred_best[sample_indices], alpha=0.4, color='#1f77b4', edgecolors='none', s=25)
    min_val, max_val = min(y_test.min(), y_pred_best.min()), max(y_test.max(), y_pred_best.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit Line (y = x)')
    plt.title(f'Actual vs. Predicted Temperature: {top_model["Algorithm"]}', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Actual Temperature (°C)', fontsize=11)
    plt.ylabel('Predicted Temperature (°C)', fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'scatter_actual_vs_predicted.png'), dpi=300)
    plt.close()
    
    # Plot 4: Residuals Distribution Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(residuals, kde=True, ax=axes[0], color='#9467bd', bins=40)
    axes[0].set_title('Residuals Error Distribution (Normal Error Check)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Residual Error = (Actual - Predicted) [°C]', fontsize=10)
    axes[0].axvline(0, color='red', linestyle='--')
    
    axes[1].scatter(y_pred_best[sample_indices], residuals[sample_indices], alpha=0.4, color='#8c564b', s=20)
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].set_title('Residuals vs. Predicted Values (Homoscedasticity)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Temperature (°C)', fontsize=10)
    axes[1].set_ylabel('Residual Error (°C)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'residuals_plot.png'), dpi=300)
    plt.close()
    
    # Plot 5: Training Time Comparison
    plt.figure(figsize=(9, 4.5))
    sns.barplot(data=df_ranked.sort_values(by='Train_Time_Sec'), x='Train_Time_Sec', y='Algorithm', palette='viridis', hue='Algorithm', legend=False)
    plt.title('Computational Training Time Comparison (Seconds)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Training Time (Seconds)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'training_time_comparison.png'), dpi=300)
    plt.close()
    
    print(f"   Visualizations saved to: {VIZ_DIR}")
    print("=" * 80)
    return df_ranked

if __name__ == '__main__':
    evaluate_regression_models()
