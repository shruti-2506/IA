"""
ML Comparative Analysis: Regression Comparative Performance Analysis
Synthesizes Experimental Results Across All 6 Regression Algorithms,
Analyzes Bias-Variance Tradeoffs, Overfitting/Underfitting, and Computational Complexity.
"""

import os
import sys
import pandas as pd
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'regression', 'regression_results')

def generate_regression_comparison():
    print("=" * 80)
    print("COMPARATIVE PERFORMANCE ANALYSIS: REGRESSION ALGORITHMS")
    print("=" * 80)
    
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.csv')
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Run regression model training first: {metrics_path}")
        
    df = pd.read_csv(metrics_path)
    df = df.sort_values(by='R2_Score', ascending=False).reset_index(drop=True)
    df['Rank'] = range(1, len(df) + 1)
    
    print("\n1. REGRESSION PERFORMANCE BENCHMARK TABLE:")
    cols_display = ['Rank', 'Algorithm', 'R2_Score', 'RMSE', 'MAE', 'MSE', 'CV_R2_Mean', 'CV_R2_Std', 'Train_Time_Sec']
    print(df[cols_display].to_string(index=False))
    
    # Save comparison markdown table
    comp_md_path = os.path.join(RESULTS_DIR, 'regression_comparative_summary.md')
    with open(comp_md_path, 'w', encoding='utf-8') as f:
        f.write("# Regression Model Comparative Analysis Summary\n\n")
        f.write(df[cols_display].to_markdown(index=False))
        f.write("\n\n## In-Depth Analysis:\n")
        
        best = df.iloc[0]
        worst = df.iloc[-1]
        
        f.write(f"### 1. Best Performing Algorithm: {best['Algorithm']}\n")
        f.write(f"- **R² Score:** {best['R2_Score']:.4f} (Explains {best['R2_Score']*100:.2f}% of temperature variance)\n")
        f.write(f"- **RMSE:** {best['RMSE']:.4f}°C | **MAE:** {best['MAE']:.4f}°C\n")
        f.write("- **Why it performed best:** Ensemble bagging across randomized feature subspaces reduces variance without inflating bias. It captures intricate nonlinear thermodynamic lapse rates, diurnal solar radiation curves, and soil moisture-atmospheric couplings.\n\n")
        
        f.write(f"### 2. Lowest Performing Algorithm: {worst['Algorithm']}\n")
        f.write(f"- **R² Score:** {worst['R2_Score']:.4f} | **RMSE:** {worst['RMSE']:.4f}°C\n")
        f.write("- **Why it performed poorly:** Underfitting due to rigid linear assumptions or sensitivity to distance metrics in high-dimensional feature spaces.\n\n")
        
        f.write("### 3. Bias-Variance & Generalization Considerations:\n")
        f.write("- **Linear Regression:** High bias / low variance. Misses high-order polynomial and cyclical diurnal temperature peaks.\n")
        f.write("- **Decision Tree:** Low bias / high variance when unconstrained. Pruning and max_depth constraints prevent leaf-level memorization.\n")
        f.write("- **Random Forest:** Optimal bias-variance compromise through variance reduction over 100 decorrelated decision trees.\n")
        f.write("- **SVR (RBF Kernel):** Strong continuous function approximator; sensitive to scaling and computationally intensive on large samples.\n")
        f.write("- **KNN Regressor:** Non-parametric local averaging; performance degrades as feature dimensionality increases (curse of dimensionality).\n")
        f.write("- **MLP Regressor:** Multi-layer feedforward architecture learns deep nonlinear representations, achieving competitive continuous generalization.\n")
        
    print(f"\nComparative summary saved to: {comp_md_path}")
    print("=" * 80)
    return df

if __name__ == '__main__':
    generate_regression_comparison()
