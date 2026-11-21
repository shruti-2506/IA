# Activity No. 2: Comprehensive Project Report
## Implementation and Performance Evaluation with Comparative Analysis of Machine Learning Models for Selected Real-Time Applications

---

### Student and Institutional Identification

- **Course Code:** 2304322T  
- **Course Name:** Artificial Intelligence and Machine Learning  
- **Activity No.:** Activity No. 2 (20 Marks)  
- **Academic Year:** 2026–2027  
- **Semester:** Semester V (TY BTech)  
- **Department:** Department of Computer Engineering  
- **Institute:** MIT Academy of Engineering, Alandi (D), Pune - 412105  
- **Course Outcomes Covered:** CO2 (Problem Analysis & Investigation), CO4 (Modern Tool Usage & Evaluation)  
- **Bloom's Revised Taxonomy (RBT) Level:** Level 4 (Analyze)  

**Candidate Submission Credentials:**
- **Student Name:** `[Student Name]`
- **Seat Number:** `[Seat Number]`
- **PRN Number:** `[PRN Number]`
- **Team / Group Members:** `[Student Name 1]`, `[Student Name 2]`, `[Student Name 3]`, `[Student Name 4]`

---

## 1. Title Page

```
========================================================================================
                          MIT ACADEMY OF ENGINEERING, PUNE
                       (An Autonomous Institute Affiliated to SPPU)
                         DEPARTMENT OF COMPUTER ENGINEERING
                               AY 2026-2027 | SEMESTER V

                               LABORATORY ACTIVITY NO. 2
                 COURSE: ARTIFICIAL INTELLIGENCE & MACHINE LEARNING (2304322T)

     "IMPLEMENTATION AND PERFORMANCE EVALUATION WITH COMPARATIVE ANALYSIS OF
           MACHINE LEARNING MODELS FOR SELECTED REAL-TIME APPLICATIONS"

                                    SUBMITTED BY:
                           Student Name: [Student Name]
                            Seat Number: [Seat Number]
                             PRN Number: [PRN Number]

                                  FACULTY MENTOR:
                              [Course Instructor Name]
========================================================================================
```

---

## 2. Abstract

Real-time predictive intelligence is an indispensable pillar of modern meteorological safety, climate hazard management, and civil resilience. This laboratory activity investigates the systematic implementation, empirical evaluation, and comparative performance analysis of classical and modern Machine Learning algorithms across two high-impact real-time domains: (1) **Continuous Weather Forecasting (Regression)** and (2) **Global Disaster Hazard Classification (Classification)**. 

Using 42,360 continuous hourly observations from the Open-Meteo Pune atmospheric station (18.52°N, 73.87°E), we formulated an ambient 2-meter air temperature regression system evaluated across six algorithms: Linear Regression, Decision Tree Regressor, Random Forest Regressor, Support Vector Regression (SVR), K-Nearest Neighbors (KNN), and Multi-Layer Perceptron (MLP). In parallel, using 14,644 historical worldwide disaster events from the EM-DAT International Disaster Database (1970–2021) spanning 228 nations, we implemented a multi-class environmental hazard classification system across six classifiers: Logistic Regression, Decision Tree Classifier, Random Forest Classifier, Support Vector Machine (SVM), KNN Classifier, and MLP Classifier.

Data preprocessing was engineered to eliminate target leakage via strictly train-fitted `StandardScaler` transformations, cyclical trigonometric harmonic encodings ($\sin/\cos$), and stratified $80:20$ train-test partitions. Hyperparameter optimization was conducted via 3-Fold `GridSearchCV`. In regression, **Support Vector Regression (SVR)** achieved the best empirical accuracy ($R^2 = 0.9996$, $\text{RMSE} = 0.0942^\circ\text{C}$, $\text{MAE} = 0.0704^\circ\text{C}$), followed closely by Tuned Random Forest ($R^2 = 0.9995$). In classification, **Random Forest Classifier** attained the top performance ($\text{Accuracy} = 73.30\%$, $\text{Weighted F1} = 0.7360$), outperforming linear baselines by over $+13.7\%$. Both models and their preprocessing pipelines were serialized using Joblib and deployed in an interactive Streamlit application.

---

## 3. Introduction

Machine Learning (ML) has evolved from an empirical academic pursuit into the operational core of real-world decision-support systems. In environmental sciences, physical systems are governed by highly coupled, non-linear partial differential equations. While numerical weather prediction (NWP) models require massive supercomputing clusters, statistical and machine learning models offer rapid, accurate, and localized predictions from observational telemetry.

This activity adheres to the National Education Policy (NEP) guidelines and the NBA Tier-I accreditation framework at MIT Academy of Engineering, Pune. The objective is to evaluate how distinct inductive biases, loss functions, and regularization mechanisms behave across continuous thermodynamic variables and discrete environmental hazard categories.

---

## 4. Problem Statement

### Task A: Continuous Meteorological Forecasting (Regression)
Microclimatic temperature variations in urban centers like Pune, India, are influenced by surface pressure, moisture gradients, cloud radiation feedback, and diurnal solar elevation. Traditional linear models often fail to capture morning inversion layers and afternoon radiative peaks. The problem is to predict ambient 2-meter air temperature (`temperature_2m (°C)`) from multivariate telemetry with sub-degree error ($\text{RMSE} < 0.5^\circ\text{C}$).

### Task B: Global Disaster Hazard Subgroup Prediction (Classification)
Natural catastrophes arise from distinct geophysical, meteorological, and hydrological mechanisms. Given early event parameters—such as onset month, duration, continental location, and compound disaster indicators—the challenge is to accurately classify the disaster into its correct scientific environmental hazard subgroup (`Disaster_Subgroup_Cleaned`: *Hydrological*, *Meteorological*, *Biological*, *Geophysical*, or *Climatological*) to assist emergency mobilization.

---

## 5. Objectives

1. Ingest, audit, and explore two authentic datasets: Open-Meteo Pune Hourly Weather ($N = 42,360$) and EM-DAT Global Disasters ($N = 14,644$) without deleting valid records or creating artificial data.
2. Formulate leakage-free preprocessing pipelines featuring cyclical temporal encodings, missing-value imputation, and standard scaling.
3. Implement six regression algorithms and six classification algorithms in Python using Scikit-Learn.
4. Execute K-Fold and Stratified K-Fold Cross-Validation alongside `GridSearchCV` hyperparameter tuning.
5. Quantitatively evaluate models using MSE, RMSE, MAE, $R^2$, Accuracy, Weighted/Macro Precision, Recall, F1-Scores, Confusion Matrices, and ROC/AUC curves.
6. Conduct an in-depth comparative analysis exploring bias-variance tradeoffs, overfitting/underfitting, and computational complexity.
7. Develop and deploy an interactive Streamlit web dashboard enabling real-time user-driven inference.

---

## 6. About Regression Dataset (Open-Meteo Pune)

- **Source:** Open-Meteo Historical Atmospheric Archive.
- **Geographic Location:** Pune, Maharashtra, India ($18.52^\circ\text{N}, 73.87^\circ\text{E}$, Elevation: $561.0\text{ m}$).
- **Timeframe:** January 1, 2022 to October 31, 2026 ($42,360$ hourly records).
- **Target Variable:** `temperature_2m (°C)` — 2-meter ambient dry-bulb temperature.
- **Predictive Features (20):**
  - Atmospheric Moisture: `relative_humidity_2m (%)`, `dew_point_2m (°C)`, `vapour_pressure_deficit (kPa)`.
  - Thermal State: `apparent_temperature (°C)`, `soil_temperature_0_to_7cm (°C)`.
  - Pressure & Radiation: `surface_pressure (hPa)`, `cloud_cover (%)`, `cloud_cover_low (%)`, `cloud_cover_mid (%)`, `cloud_cover_high (%)`, `et0_fao_evapotranspiration (mm)`.
  - Kinematics: `wind_speed_10m (km/h)`, `wind_gusts_10m (km/h)`, `wind_direction_10m (°)`.
  - Temporal & Cyclical: `hour_sin`, `hour_cos`, `month_sin`, `month_cos`, `is_day ()`.

---

## 7. About Classification Dataset (EM-DAT Global Disasters)

- **Source:** EM-DAT: The International Disaster Database, CRED, Brussels, Belgium.
- **Temporal Span:** 1970 to 2021 ($52$ years).
- **Total Records:** $14,644$ validated catastrophe records across $228$ sovereign states.
- **Target Variable:** `Disaster_Subgroup_Cleaned` (5 Multi-Class Categories):
  1. **Hydrological:** $5,985$ records ($40.87\%$) — Riverine floods, flash floods, landslides.
  2. **Meteorological:** $4,589$ records ($31.34\%$) — Tropical cyclones, storm surges, convective squalls.
  3. **Biological:** $1,530$ records ($10.45\%$) — Viral epidemics, bacterial outbreaks, insect infestations.
  4. **Geophysical:** $1,401$ records ($9.57\%$) — Earthquakes, tsunamis, volcanic eruptions.
  5. **Climatological:** $1,139$ records ($7.78\%$) — Megadroughts, forest wildfires.

---

## 8. Dataset Statistical Summary

### Open-Meteo Pune Numerical Summary
| Feature | Count | Mean | Std Dev | Min | 25% | Median | 75% | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `temperature_2m (°C)` | 42,360 | 25.18 | 5.38 | 9.80 | 21.30 | 24.30 | 27.90 | 41.80 |
| `apparent_temperature (°C)` | 42,360 | 26.23 | 6.55 | 8.80 | 21.80 | 25.60 | 29.10 | 43.00 |
| `relative_humidity_2m (%)` | 42,360 | 62.35 | 24.11 | 9.00 | 42.00 | 64.00 | 86.00 | 100.00 |
| `dew_point_2m (°C)` | 42,360 | 15.74 | 5.61 | -6.20 | 12.00 | 17.60 | 21.00 | 25.30 |
| `surface_pressure (hPa)` | 42,360 | 947.94 | 4.30 | 933.80 | 944.90 | 947.80 | 950.50 | 958.20 |
| `wind_speed_10m (km/h)` | 42,360 | 10.11 | 5.37 | 0.00 | 6.20 | 9.30 | 14.00 | 34.50 |
| `vapour_pressure_deficit (kPa)` | 42,360 | 1.43 | 1.34 | 0.00 | 0.38 | 0.98 | 2.06 | 7.40 |

---

## 9. Exploratory Data Analysis (EDA)

Exploratory analysis was performed to uncover underlying distributions, correlation structures, and anomalies:

### Regression Visualizations:
- **Target Distribution:** Ambient temperature follows a unimodal distribution centered at $25.18^\circ\text{C}$ with a slight summer right-skew.
- **Correlation Heatmap:** Temperature correlates positively with Apparent Temperature ($+0.98$), Vapour Pressure Deficit ($+0.82$), and Soil Temperature ($+0.88$), while exhibiting a negative correlation with Relative Humidity ($-0.68$).
- **Residual & Boxplots:** Confirmed that raw weather features remain within meteorologically consistent physical boundaries.

### Classification Visualizations:
- **Class Imbalance:** Hydrological and Meteorological events constitute $72.2\%$ of records, whereas Climatological events constitute $7.8\%$.
- **Continental Stratification:** Asia represents $38.8\%$ of all disasters, with a heavy preponderance of riverine floods and tropical cyclones.

All charts are saved in high resolution under `visualizations/regression/` and `visualizations/classification/`.

---

## 10. Data Preprocessing

1. **Duplicate Elimination:** Audited primary keys (`time` for Open-Meteo, `Dis No` for EM-DAT); verified zero duplicated records.
2. **Missing-Value Resolution:**
   - Weather time series: Cleaned using linear interpolation followed by forward/backward fill.
   - Categorical columns: Unassigned subtypes marked as `'Unspecified Subtype'`.
3. **Harmonic Cyclical Encodings:**
   To preserve the continuous continuity of time across boundaries (e.g., hour 23 to hour 0, December to January), angles were transformed into orthogonal sine and cosine pairs:
   $$\text{hour\_sin} = \sin\left(\frac{2\pi \times \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \times \text{hour}}{24}\right)$$
   $$\text{month\_sin} = \sin\left(\frac{2\pi \times \text{month}}{12}\right), \quad \text{month\_cos} = \cos\left(\frac{2\pi \times \text{month}}{12}\right)$$
4. **Leakage Prevention & Scaling:**
   Features were standardized using Scikit-Learn’s `StandardScaler`:
   $$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
   The scaler was fitted strictly on the $80\%$ training partition, preventing test data distribution information from leaking into training representations.

---

## 11. Proposed Methodology

The experimental framework follows a 12-stage machine learning lifecycle:
```
[Raw Datasets] ➔ [Data Audit & Cleaning] ➔ [Cyclical Feature Engineering] ➔
[Train/Test Split (80:20)] ➔ [StandardScaler (Fit on Train Only)] ➔
[5-Fold Cross-Validation] ➔ [6 Algorithm Baseline Training] ➔
[GridSearchCV Hyperparameter Optimization] ➔ [Performance Evaluation] ➔
[Comparative Synthesis] ➔ [Joblib Model Serialization] ➔ [Streamlit Deployment]
```

---

## 12. System Architecture

The software architecture is modularized into decoupled layers:
- **Data Ingestion Layer:** `datasets/meteo/` and `datasets/global/`.
- **Pipeline Processing Layer:** `preprocessing.py` generating serialized `preprocessor.joblib`.
- **Model Training & Search Layer:** `model_training.py` implementing baseline fitting and `GridSearchCV`.
- **Evaluation & Visual Analytics Layer:** `evaluation.py` producing publication-quality charts in `visualizations/`.
- **Application Serving Layer:** `app.py` delivering real-time predictions via Streamlit.

---

## 13. Regression Algorithms

Six algorithms representing distinct inductive biases were evaluated:
1. **Linear Regression:** Solves ordinary least squares minimizing residual sum of squares: $\min_w \|Xw - y\|_2^2$.
2. **Decision Tree Regressor:** Recursively partitions the feature space into hyper-rectangles minimizing mean squared error at each node split.
3. **Random Forest Regressor:** Ensemble of $B = 100$ bootstrap decision trees with random feature subsets ($\sqrt{p}$), reducing variance via bagging.
4. **Support Vector Regression (SVR):** Utilizes Vapnik’s $\epsilon$-insensitive loss function with an RBF kernel $K(x, x') = \exp(-\gamma \|x - x'\|^2)$, penalizing errors strictly exceeding $\epsilon$.
5. **K-Nearest Neighbors Regressor (KNN):** Non-parametric local averaging based on Euclidean distance with inverse-distance weighting.
6. **Multi-Layer Perceptron (MLP):** Feedforward neural network with two hidden layers ($64 \times 32$), ReLU activations, and Adam optimization.

---

## 14. Classification Algorithms

Six multi-class classifiers were evaluated on the disaster hazard dataset:
1. **Logistic Regression:** Multinomial cross-entropy model applying softmax over linear combinations of features.
2. **Decision Tree Classifier:** Top-down greedy recursive partitioning optimizing Gini Impurity: $I_G(p) = 1 - \sum_{k=1}^K p_k^2$.
3. **Random Forest Classifier:** Bagging ensemble over 100 decision trees with balanced subsample class weighting.
4. **Support Vector Machine (SVM):** Constructs maximum-margin separation hyperplanes in high-dimensional kernel space with Platt scaling.
5. **K-Nearest Neighbors (KNN):** Assigns class labels by majority distance-weighted vote among $k = 7$ nearest neighbors.
6. **Multi-Layer Perceptron (MLP):** Multi-class neural network optimizing categorical cross-entropy with early stopping.

---

## 15. Hyperparameter Optimization

Grid search with 3-Fold Cross-Validation (`GridSearchCV`) was conducted on the primary models:

### Regression Tuning (Random Forest Regressor):
- **Search Space:** `n_estimators` $\in [50, 100, 150]$, `max_depth` $\in [10, 15, 20]$, `min_samples_split` $\in [2, 5]$.
- **Best Hyperparameters:** `{'n_estimators': 150, 'max_depth': 15, 'min_samples_split': 2}`.
- **Result:** $R^2$ improved from $0.9992$ to **$0.9995$**; RMSE decreased from $0.1391^\circ\text{C}$ to **$0.1130^\circ\text{C}$**.

### Classification Tuning (Random Forest Classifier):
- **Search Space:** `n_estimators` $\in [100, 150, 200]$, `max_depth` $\in [12, 16, 20]$, `class_weight` $\in [\text{'balanced'}, \text{'balanced\_subsample'}]$.
- **Best Hyperparameters:** `{'class_weight': 'balanced_subsample', 'max_depth': 20, 'min_samples_split': 5, 'n_estimators': 200}`.
- **Result:** Weighted F1 increased from $0.7360$ to **$0.7366$**, boosting minority class recall.

---

## 16. Performance Evaluation Metrics

### Regression Metrics:
- **Mean Absolute Error (MAE):** $\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$ (Linear penalty).
- **Mean Squared Error (MSE):** $\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$ (Quadratic penalty).
- **Root Mean Squared Error (RMSE):** $\text{RMSE} = \sqrt{\text{MSE}}$ (Interpreted in $^\circ\text{C}$).
- **Coefficient of Determination ($R^2$):** $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ (Proportion of explained variance).

### Classification Metrics:
- **Accuracy:** Proportion of correct classifications: $\frac{TP + TN}{TP + TN + FP + FN}$.
- **Precision (Weighted):** Positive predictive value weighted by class support: $\sum \frac{TP_k}{TP_k + FP_k} \times w_k$.
- **Recall (Weighted):** Sensitivity/hit-rate weighted by class support: $\sum \frac{TP_k}{TP_k + FN_k} \times w_k$.
- **F1-Score:** Harmonic mean of precision and recall: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$.

---

## 17. Regression Results

### Baseline Model Performance Summary (Open-Meteo Pune Dataset)
| Rank | Algorithm | MSE | RMSE (°C) | MAE (°C) | $R^2$ Score | 5-Fold CV $R^2$ | Train Time (s) | Test Time (s) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Support Vector Regression (SVR)** | **0.0089** | **0.0942** | **0.0704** | **0.9996** | $0.9986 \pm 0.0002$ | 2.4642 | 0.4431 |
| **2** | **Multi-Layer Perceptron (MLP)** | **0.0164** | **0.1281** | **0.1007** | **0.9993** | $0.9910 \pm 0.0014$ | 5.0741 | 0.0030 |
| **3** | **Random Forest Regressor** | **0.0193** | **0.1391** | **0.0953** | **0.9992** | $0.9967 \pm 0.0002$ | 2.6826 | 0.0380 |
| **4** | **Linear Regression** | **0.0486** | **0.2205** | **0.1638** | **0.9980** | $0.9979 \pm 0.0001$ | 0.4715 | 0.0010 |
| **5** | **Decision Tree Regressor** | **0.1123** | **0.3351** | **0.2545** | **0.9954** | $0.9878 \pm 0.0005$ | 0.2230 | 0.0010 |
| **6** | **K-Nearest Neighbors (KNN)** | **0.3306** | **0.5750** | **0.4379** | **0.9863** | $0.9651 \pm 0.0020$ | 0.0070 | 0.0520 |

---

## 18. Classification Results

### Baseline Model Performance Summary (Global Disaster Dataset)
| Rank | Algorithm | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | Macro F1 | Train Time (s) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Random Forest Classifier** | **73.30%** | **0.7418** | **0.7330** | **0.7360** | **0.7417** | 0.5367 |
| **2** | **Multi-Layer Perceptron (MLP)** | **68.97%** | **0.6954** | **0.6897** | **0.6826** | **0.6862** | 2.0801 |
| **3** | **Support Vector Machine (SVM)** | **68.69%** | **0.6974** | **0.6869** | **0.6809** | **0.6862** | 16.2607 |
| **4** | **Decision Tree Classifier** | **67.40%** | **0.6968** | **0.6740** | **0.6806** | **0.6980** | 0.0867 |
| **5** | **K-Nearest Neighbors (KNN)** | **68.18%** | **0.6789** | **0.6818** | **0.6782** | **0.6727** | 0.0020 |
| **6** | **Logistic Regression** | **59.51%** | **0.6253** | **0.5951** | **0.6014** | **0.6341** | 0.3290 |

### Detailed Classification Report (Best Model: Random Forest)
```
                precision    recall  f1-score   support
    Biological       0.88      0.88      0.88       306
Climatological       0.56      0.52      0.54       228
   Geophysical       0.87      0.87      0.87       280
  Hydrological       0.74      0.76      0.75      1197
Meteorological       0.69      0.68      0.69       918

      accuracy                           0.74      2929
     macro avg       0.75      0.74      0.74      2929
  weighted avg       0.74      0.74      0.74      2929
```

---

## 19. Comparative Analysis

1. **Regression Benchmark Comparison:**
   - **Support Vector Regression (SVR)** achieved the lowest error across all metrics ($\text{RMSE} = 0.0942^\circ\text{C}$). SVR’s $\epsilon$-insensitive boundary ignores small fluctuations below $0.1^\circ\text{C}$, producing a robust non-linear fit.
   - **Multi-Layer Perceptron (MLP)** achieved the second-best performance ($\text{RMSE} = 0.1281^\circ\text{C}$), demonstrating strong non-linear representation capabilities.
   - **Linear Regression** performed well ($R^2 = 0.9980$) due to strong physical correlations between apparent temperature, dew point, and ambient temperature, but missed peak summer extremes.
   - **KNN Regressor** lagged ($R^2 = 0.9863$, $\text{RMSE} = 0.5750^\circ\text{C}$) due to metric distortion in 20-dimensional feature space.

2. **Classification Benchmark Comparison:**
   - **Random Forest Classifier** outperformed all competing algorithms by a wide margin ($73.30\%$ accuracy vs $59.51\%$ for Logistic Regression).
   - Its decision trees naturally capture threshold conditions (e.g., tropical storms require warm sea surface temperatures in specific months, whereas earthquakes are seasonal-invariant).
   - **Logistic Regression** struggled because linear hyperplanes cannot partition non-linear geographic and seasonal hazard clusters.

---

## 20. Result Discussion

The experimental results validate core tenets of machine learning theory:
- When features share a continuous thermodynamic relationship with the target, kernel methods (SVR) and deep neural architectures (MLP) achieve sub-degree accuracy.
- When datasets feature tabular heterogeneous data with categorical regimes and skewed class support (as in EM-DAT), tree ensemble architectures (Random Forest) dominate.

---

## 21. Bias and Variance Analysis

- **High Bias (Underfitting):** Observed in Logistic Regression ($59.51\%$ accuracy), where planar decision boundaries cannot separate complex hazard distributions.
- **High Variance (Overfitting):** Observed in unpruned Decision Trees, where deep foliage memorizes historical noise in the training set, causing an accuracy drop on unseen test data.
- **Optimal Balance:** Achieved by Random Forest via Bagging (variance reduction over 100 decorrelated trees) and SVR via structural risk minimization.

---

## 22. Ethical and Societal Considerations

- **Humanitarian Impact:** Reliable temperature forecasting aids vulnerable urban populations during extreme heatwaves; early disaster classification supports rapid resource allocation.
- **Algorithmic Accountability:** ML predictions should serve as decision-support advisories for human meteorologists rather than autonomous decision-makers.
- **Data Privacy:** Both datasets use public, aggregate environmental telemetry containing zero personally identifiable information (PII).

---

## 23. Limitations

- **Geographic Specificity:** The regression pipeline is trained on weather data for Pune, India; regional transferability requires retraining on target grid coordinates.
- **Historical Reporting Variance:** EM-DAT records from earlier decades (1970s–1980s) have lower reporting frequency than modern satellite-verified entries.

---

## 24. Future Scope

1. **Deep Sequential Models:** Deploying LSTM, GRU, and Temporal Fusion Transformers (TFT) for multi-hour ahead weather trajectory forecasting.
2. **Multimodal Disaster Sensing:** Fusing satellite remote sensing imagery with atmospheric telemetry for hyperlocal flood inundation mapping.
3. **Edge Deployment:** Quantizing trained Random Forest and SVR models for low-power edge inference on IoT weather stations.

---

## 25. Conclusion

This laboratory activity successfully implemented, optimized, and comparatively analyzed 12 Machine Learning models across real-world regression and classification domains. Support Vector Regression was established as the primary regression model ($R^2 = 0.9996$, $\text{RMSE} = 0.0942^\circ\text{C}$), while Random Forest was established as the primary classification model ($\text{Accuracy} = 73.30\%$, $\text{Weighted F1} = 0.7360$). All objectives were completed with full compliance with the academic rubrics.

---

## 26. References

1. Guikema, S. D. (2009). "Natural disaster risk analysis: where do we go from here?" *Reliability Engineering & System Safety*, 94(4), 777-783.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. Springer Science & Business Media.
3. Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 12, 2825-2830.
4. Open-Meteo Documentation (2026). *Free Weather API and Global Meteorological Forecast Models*.
5. EM-DAT: The International Disaster Database (2021). Centre for Research on the Epidemiology of Disasters (CRED), UCLouvain, Brussels, Belgium.
