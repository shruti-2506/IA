"""
ML Comparative Analysis - Classification Module: Evaluation & Visualization
Computes Confusion Matrices, Per-Class Classification Reports, Multiclass ROC Curves,
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

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
RESULTS_DIR = os.path.join(CURRENT_DIR, 'classification_results')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
VIZ_DIR = os.path.join(PROJECT_ROOT, 'visualizations', 'classification')
os.makedirs(VIZ_DIR, exist_ok=True)

def evaluate_classification_models():
    print("=" * 80)
    print("CLASSIFICATION WORKFLOW: PERFORMANCE EVALUATION & VISUALIZATION")
    print("=" * 80)
    
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.csv')
    if not os.path.exists(metrics_path):
        from model_training import train_and_tune_classification_models
        train_and_tune_classification_models()
        
    df_metrics = pd.read_csv(metrics_path)
    X_test = np.load(os.path.join(RESULTS_DIR, 'X_test_scaled.npy'))
    y_test = np.load(os.path.join(RESULTS_DIR, 'y_test.npy'))
    
    prep_bundle = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.joblib'))
    class_names = prep_bundle['class_names']
    
    best_model_path = os.path.join(MODELS_DIR, 'best_model.joblib')
    best_model = joblib.load(best_model_path)
    
    y_pred_best = best_model.predict(X_test)
    y_prob_best = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
    
    # 1. Performance Ranking
    df_ranked = df_metrics.sort_values(by='F1_Weighted', ascending=False).reset_index(drop=True)
    df_ranked['Rank'] = range(1, len(df_ranked) + 1)
    df_ranked.to_csv(os.path.join(RESULTS_DIR, 'model_ranking.csv'), index=False)
    
    print("\n1. PERFORMANCE SUMMARY TABLE:")
    print(df_ranked[['Algorithm', 'Accuracy', 'Precision_Weighted', 'Recall_Weighted', 'F1_Weighted', 'Train_Time_Sec']].to_string(index=False))
    
    top_model = df_ranked.iloc[0]
    print(f"\n2. TOP PERFORMING CLASSIFICATION ALGORITHM: {top_model['Algorithm']}")
    print(f"   Accuracy:    {top_model['Accuracy']*100:.2f}%")
    print(f"   F1 (Weighted): {top_model['F1_Weighted']:.4f}")
    
    # 2. Detailed Classification Report
    clf_rep = classification_report(y_test, y_pred_best, target_names=class_names, zero_division=0)
    print("\n3. DETAILED CLASSIFICATION REPORT (BEST MODEL):")
    print(clf_rep)
    with open(os.path.join(RESULTS_DIR, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Top Model: {top_model['Algorithm']}\n")
        f.write("="*60 + "\n")
        f.write(clf_rep)
        
    # -------------------------------------------------------------
    # GENERATE VISUALIZATIONS
    # -------------------------------------------------------------
    print("\n4. GENERATING COMPARATIVE VISUALIZATIONS...")
    
    # Plot 1: Model Accuracy Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    colors = ['#2ca02c' if x == top_model['Algorithm'] else '#1f77b4' for x in df_ranked['Algorithm']]
    ax = sns.barplot(data=df_ranked, x='Accuracy', y='Algorithm', palette=colors, hue='Algorithm', legend=False)
    plt.title('Classification Algorithm Comparison: Accuracy Score (%)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Test Accuracy Score (Higher is Better)', fontsize=11)
    plt.xlim(0, 1.05)
    for i, v in enumerate(df_ranked['Accuracy']):
        ax.text(v + 0.01, i, f"{v*100:.2f}%", va='center', fontweight='bold', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'model_accuracy_comparison.png'), dpi=300)
    plt.close()
    
    # Plot 2: Precision, Recall & F1 Comparison
    plt.figure(figsize=(12, 5))
    df_melt = pd.melt(df_ranked, id_vars=['Algorithm'], 
                      value_vars=['Precision_Weighted', 'Recall_Weighted', 'F1_Weighted'],
                      var_name='Metric', value_name='Score')
    sns.barplot(data=df_melt, x='Algorithm', y='Score', hue='Metric', palette=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title('Classification Metrics Comparison (Weighted Precision, Recall & F1)', fontsize=13, fontweight='bold', pad=12)
    plt.ylabel('Score (0.0 to 1.0)', fontsize=11)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=20)
    plt.legend(title='Metric', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'precision_recall_f1_comparison.png'), dpi=300)
    plt.close()
    
    # Plot 3: Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred_best)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Normalized Proportion'})
    plt.title(f'Normalized Confusion Matrix: {top_model["Algorithm"]}', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Predicted Class', fontsize=11)
    plt.ylabel('True Class', fontsize=11)
    plt.xticks(rotation=25, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'confusion_matrices.png'), dpi=300)
    plt.close()
    
    # Plot 4: Multiclass ROC Curves (One-vs-Rest)
    if y_prob_best is not None and len(class_names) > 1:
        plt.figure(figsize=(9, 7))
        y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob_best[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
            
        plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance (AUC = 0.50)')
        plt.title(f'Multiclass ROC Curves (One-vs-Rest): {top_model["Algorithm"]}', fontsize=13, fontweight='bold', pad=12)
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
        plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
        plt.legend(loc='lower right', frameon=True, fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, 'roc_curves.png'), dpi=300)
        plt.close()
        
    # Plot 5: Training Time Comparison
    plt.figure(figsize=(9, 4.5))
    sns.barplot(data=df_ranked.sort_values(by='Train_Time_Sec'), x='Train_Time_Sec', y='Algorithm', palette='mako', hue='Algorithm', legend=False)
    plt.title('Computational Training Time Comparison (Seconds)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Training Time (Seconds)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'training_time_comparison.png'), dpi=300)
    plt.close()
    
    print(f"   Visualizations saved to: {VIZ_DIR}")
    print("=" * 80)
    return df_ranked

if __name__ == '__main__':
    evaluate_classification_models()
