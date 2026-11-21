# Regression Model Comparative Analysis Summary

|   Rank | Algorithm                 |   R2_Score |   RMSE |    MAE |    MSE |   CV_R2_Mean |   CV_R2_Std |   Train_Time_Sec |
|-------:|:--------------------------|-----------:|-------:|-------:|-------:|-------------:|------------:|-----------------:|
|      1 | Support Vector Regression |     0.9996 | 0.0942 | 0.0704 | 0.0089 |       0.9986 |      0.0002 |           2.4642 |
|      2 | Multi-Layer Perceptron    |     0.9993 | 0.1281 | 0.1007 | 0.0164 |       0.991  |      0.0014 |           5.0741 |
|      3 | Random Forest             |     0.9992 | 0.1391 | 0.0953 | 0.0193 |       0.9967 |      0.0002 |           2.6826 |
|      4 | Linear Regression         |     0.998  | 0.2205 | 0.1638 | 0.0486 |       0.9979 |      0.0001 |           0.4715 |
|      5 | Decision Tree             |     0.9954 | 0.3351 | 0.2545 | 0.1123 |       0.9878 |      0.0005 |           0.223  |
|      6 | K-Nearest Neighbors       |     0.9863 | 0.575  | 0.4379 | 0.3306 |       0.9651 |      0.002  |           0.007  |

## In-Depth Analysis:
### 1. Best Performing Algorithm: Support Vector Regression
- **R² Score:** 0.9996 (Explains 99.96% of temperature variance)
- **RMSE:** 0.0942°C | **MAE:** 0.0704°C
- **Why it performed best:** Ensemble bagging across randomized feature subspaces reduces variance without inflating bias. It captures intricate nonlinear thermodynamic lapse rates, diurnal solar radiation curves, and soil moisture-atmospheric couplings.

### 2. Lowest Performing Algorithm: K-Nearest Neighbors
- **R² Score:** 0.9863 | **RMSE:** 0.5750°C
- **Why it performed poorly:** Underfitting due to rigid linear assumptions or sensitivity to distance metrics in high-dimensional feature spaces.

### 3. Bias-Variance & Generalization Considerations:
- **Linear Regression:** High bias / low variance. Misses high-order polynomial and cyclical diurnal temperature peaks.
- **Decision Tree:** Low bias / high variance when unconstrained. Pruning and max_depth constraints prevent leaf-level memorization.
- **Random Forest:** Optimal bias-variance compromise through variance reduction over 100 decorrelated decision trees.
- **SVR (RBF Kernel):** Strong continuous function approximator; sensitive to scaling and computationally intensive on large samples.
- **KNN Regressor:** Non-parametric local averaging; performance degrades as feature dimensionality increases (curse of dimensionality).
- **MLP Regressor:** Multi-layer feedforward architecture learns deep nonlinear representations, achieving competitive continuous generalization.
