# Classification Model Comparative Analysis Summary

|   Rank | Algorithm              |   Accuracy |   Precision_Weighted |   Recall_Weighted |   F1_Weighted |   F1_Macro |   Train_Time_Sec |
|-------:|:-----------------------|-----------:|---------------------:|------------------:|--------------:|-----------:|-----------------:|
|      1 | Random Forest          |     0.733  |               0.7418 |            0.733  |        0.736  |     0.7417 |           0.5367 |
|      2 | Multi-Layer Perceptron |     0.6897 |               0.6954 |            0.6897 |        0.6826 |     0.6862 |           2.0801 |
|      3 | Support Vector Machine |     0.6869 |               0.6974 |            0.6869 |        0.6809 |     0.6862 |          16.2607 |
|      4 | Decision Tree          |     0.674  |               0.6968 |            0.674  |        0.6806 |     0.698  |           0.0867 |
|      5 | K-Nearest Neighbors    |     0.6818 |               0.6789 |            0.6818 |        0.6782 |     0.6727 |           0.002  |
|      6 | Logistic Regression    |     0.5951 |               0.6253 |            0.5951 |        0.6014 |     0.6341 |           0.329  |

## In-Depth Analysis:
### 1. Best Performing Algorithm: Random Forest
- **Accuracy:** 73.30%
- **Weighted F1-Score:** 0.7360 | **Macro F1:** 0.7417
- **Why it performed best:** Random Forest combines decision trees trained on bootstrap samples with random feature selection. It handles non-linear boundaries, resists overfitting to majority classes via balanced weighting, and isolates subtle geographic/seasonal patterns distinguishing hydrological hazards from meteorological ones.

### 2. Lowest Performing Algorithm: Logistic Regression
- **Accuracy:** 59.51% | **Weighted F1:** 0.6014
- **Why it performed poorly:** Linear hyperplanes (Logistic Regression) or distance-based metrics (KNN) struggle with complex multiclass non-linear boundaries and high-cardinality geographic one-hot encodings.

### 3. Bias-Variance & Generalization Considerations:
- **Logistic Regression:** Underfits nonlinear hazard interactions due to linear decision boundaries.
- **Decision Tree:** Captures hierarchical rules but prone to high variance on noisy disaster records without pruning.
- **Random Forest:** Maximizes generalization accuracy by aggregating decorrelated trees, achieving the highest F1-Score.
- **Support Vector Machine (SVM):** Strong kernel-based decision margins; balanced class handling prevents boundary collapse.
- **KNN Classifier:** Effective for spatially localized clusters, but sensitive to feature scale and computationally slow at test time.
- **Multi-Layer Perceptron (MLP):** Neural hidden layers capture abstract feature interactions, yielding solid multiclass discriminability.
