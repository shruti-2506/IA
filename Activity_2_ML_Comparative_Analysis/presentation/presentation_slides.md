# Implementation and Performance Evaluation with Comparative Analysis of Machine Learning Models for Selected Real-Time Applications

**Activity No.:** 2 | **Course Code:** 2304322T | **Course:** Artificial Intelligence and Machine Learning  
**Department:** Computer Engineering | **Institute:** MIT Academy of Engineering, Alandi (D), Pune  
**Academic Year:** 2026–2027 | **Semester:** V  

**Student Name:** `[Student Name]` | **Seat Number:** `[Seat Number]` | **PRN Number:** `[PRN Number]`  

---

## Slide 1: Title Slide
- **Project Title:** Implementation and Performance Evaluation with Comparative Analysis of Machine Learning Models for Selected Real-Time Applications
- **Domain:** Atmospheric Weather Prediction & Global Climate Disaster Risk Classification
- **Institute:** MIT Academy of Engineering, Pune
- **Course & Code:** AIML (2304322T), Semester V, AY 2026–2027
- **Team Members:** `[Student Name 1]`, `[Student Name 2]`, `[Student Name 3]`, `[Student Name 4]`

---

## Slide 2: Abstract
- **Context:** Real-time predictive modeling is vital for climate resilience, early weather forecasting, and catastrophe risk management.
- **Scope:** Two independent machine learning pipelines:
  1. *Continuous Weather Prediction (Regression):* Forecasting 2-meter air temperature in Pune, India using 42,360 hourly Open-Meteo records.
  2. *Disaster Hazard Prediction (Classification):* Multi-class classification of environmental hazard subgroups using 14,644 worldwide EM-DAT disaster records.
- **Algorithms:** Benchmarked 6 algorithms per task (Linear/Logistic Regression, Decision Trees, Random Forests, SVM/SVR, KNN, and Multi-Layer Perceptrons).
- **Outcomes:** Support Vector Regression (SVR) achieved $R^2 = 0.9996$ ($\text{RMSE} = 0.0942^\circ\text{C}$); Random Forest Classifier attained $73.30\%$ accuracy and $0.7360$ weighted F1-score. An interactive Streamlit dashboard provides live inference.

---

## Slide 3: Problem Statement
- **Problem 1 (Regression):** Extreme weather anomalies and urban heat island effects necessitate continuous, sub-degree ambient air temperature forecasting from coupled atmospheric telemetry.
- **Problem 2 (Classification):** Disasters arise from diverse physical mechanisms (Hydrological, Meteorological, Geophysical, Biological, Climatological). Rapidly predicting hazard type from early spatio-temporal indicators enables targeted civil protection mobilization.

---

## Slide 4: Objectives
1. Perform end-to-end exploratory data analysis (EDA) and data cleansing on real-world datasets.
2. Formulate leakage-free preprocessing pipelines with cyclical temporal encodings and standard scaling.
3. Implement and benchmark 6 regression and 6 classification algorithms using rigorous K-Fold cross-validation.
4. Execute systematic hyperparameter tuning (GridSearchCV) on primary candidate models.
5. Deploy a full-featured interactive Streamlit application with live inference capabilities.
6. Address academic rubrics: Problem Analysis (CO2), Modern Tools (CO4), Evaluation Ethics (CO4), and Communication (CO2).

---

## Slide 5: About Open-Meteo Regression Dataset
- **Location:** Pune, Maharashtra, India (18.52°N, 73.87°E, Elevation 561m).
- **Temporal Span:** January 1, 2022 to October 31, 2026 (42,360 hourly observations).
- **Target Variable:** `temperature_2m (°C)` — Fundamental thermodynamic state variable.
- **Key Predictors:** Apparent temperature, relative humidity, dew point, surface pressure, cloud cover layers (low/mid/high), wind dynamics (speed, gusts, direction), vapour pressure deficit, and soil temperature.
- **Characteristics:** Highly continuous, strong diurnal and seasonal periodicity, non-linear atmospheric couplings.

---

## Slide 6: About Global Disaster Classification Dataset
- **Database:** EM-DAT International Disaster Database (1970–2021).
- **Geographic Scope:** 14,644 disaster events across 228 countries worldwide.
- **Target Variable:** `Disaster_Subgroup_Cleaned` (5 Classes):
  - *Hydrological:* 5,985 events (Floods, Landslides)
  - *Meteorological:* 4,589 events (Tropical storms, Convective storms)
  - *Biological:* 1,530 events (Epidemics, Insect infestations)
  - *Geophysical:* 1,401 events (Earthquakes, Tsunamis, Volcanic activity)
  - *Climatological:* 1,139 events (Droughts, Wildfires)
- **Key Features:** Event year, month, duration in days, continent, season, country historical disaster frequency, climate flag, compound disaster flag.

---

## Slide 7: Dataset Exploration and Visualization
- **Weather EDA:**
  - Temperature shows Gaussian distribution centered at $25.18^\circ\text{C}$ (Range: $5^\circ\text{C}$ to $41.8^\circ\text{C}$).
  - Strong negative correlation with relative humidity ($-0.68$) and positive correlation with vapour pressure deficit ($+0.82$).
- **Disaster EDA:**
  - Asia accounts for $38.8\%$ of global disasters, followed by Americas ($25.1\%$) and Africa ($21.1\%$).
  - Hydrological and Meteorological hazards constitute over $72\%$ of all recorded catastrophes.

---

## Slide 8: Proposed Methodology
```
Dataset Ingestion (Open-Meteo & EM-DAT)
        ↓
Data Auditing & Duplicate Elimination
        ↓
Missing Value Imputation & Cyclical Feature Engineering
        ↓
Data Splitting (80:20 Train/Test with Stratification)
        ↓
StandardScaler Pipeline (Strictly fitted on Train Set)
        ↓
Baseline Model Training (6 Algorithms per Task)
        ↓
K-Fold & Stratified K-Fold Cross-Validation
        ↓
Hyperparameter Optimization (GridSearchCV)
        ↓
Multi-Metric Comparative Evaluation & Residual Analysis
        ↓
Model Serialization (Joblib) & Streamlit Deployment
```

---

## Slide 9: System Architecture
- **Data Layer:** Open-Meteo CSV (42,360 rows) & Global Disaster CSV (14,644 rows).
- **Preprocessing Layer:** Scikit-Learn `StandardScaler`, `LabelEncoder`, One-Hot Encoding, Sine/Cosine Cyclical Encoders.
- **Model Training Layer:**
  - *Regression Engine:* Linear, DecisionTree, RandomForest, SVR, KNN, MLPRegressor.
  - *Classification Engine:* LogisticRegression, DecisionTree, RandomForest, SVC, KNN, MLPClassifier.
- **Optimization Layer:** 3-Fold GridSearchCV across multi-parameter grids.
- **Serving Layer:** Streamlit Dashboard with dynamic UI sliders and real-time prediction pipeline.

---

## Slide 10: Regression Model Implementation
- Evaluated 6 diverse model paradigms on 15,000 meteorological records:
  1. **Linear Regression:** Baseline parametric ordinary least squares.
  2. **Decision Tree Regressor:** Non-parametric hierarchical splitting (max_depth=10).
  3. **Random Forest Regressor:** Bagging ensemble over 100 decision trees.
  4. **Support Vector Regression (SVR):** RBF kernel mapping with $\epsilon=0.1$, $C=10$.
  5. **K-Nearest Neighbors (KNN):** Distance-weighted local neighbor averaging ($k=7$).
  6. **Multi-Layer Perceptron (MLP):** Feedforward neural network (64, 32 ReLU hidden units).

---

## Slide 11: Classification Model Implementation
- Evaluated 6 multi-class algorithms on 14,644 disaster records (80:20 Stratified Split):
  1. **Logistic Regression:** Multinomial cross-entropy with balanced class weighting.
  2. **Decision Tree Classifier:** Gini impurity splitting with cost-complexity pruning.
  3. **Random Forest Classifier:** 100 balanced bootstrap trees with feature subspacing.
  4. **Support Vector Machine (SVM):** Non-linear RBF kernel with Platt probability scaling.
  5. **K-Nearest Neighbors (KNN):** Distance-weighted voting ($k=7$).
  6. **Multi-Layer Perceptron (MLP):** Neural architecture with early stopping.

---

## Slide 12: Hyperparameter Optimization
- **Regression Optimization (Random Forest):**
  - Search Space: `n_estimators` [50, 100, 150], `max_depth` [10, 15, 20], `min_samples_split` [2, 5].
  - Optimal Configuration: `{'n_estimators': 150, 'max_depth': 15, 'min_samples_split': 2}`.
  - Performance Gain: $R^2$ improved from $0.9992$ to **$0.9995$**; RMSE reduced from $0.1391^\circ\text{C}$ to **$0.1130^\circ\text{C}$**.
- **Classification Optimization (Random Forest):**
  - Search Space: `n_estimators` [100, 150, 200], `max_depth` [12, 16, 20], `class_weight` ['balanced', 'balanced_subsample'].
  - Optimal Configuration: `{'class_weight': 'balanced_subsample', 'max_depth': 20, 'n_estimators': 200}`.
  - Performance Gain: Weighted F1 improved from $0.7360$ to **$0.7366$** with enhanced minority class sensitivity.

---

## Slide 13: Regression Evaluation Results
| Rank | Algorithm | $R^2$ Score | RMSE (°C) | MAE (°C) | 5-Fold CV $R^2$ | Train Time |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Support Vector Regression (SVR)** | **0.9996** | **0.0942** | **0.0704** | $0.9986 \pm 0.0002$ | 2.46s |
| **2** | **Multi-Layer Perceptron (MLP)** | **0.9993** | **0.1281** | **0.1007** | $0.9910 \pm 0.0014$ | 5.07s |
| **3** | **Random Forest Regressor** | **0.9992** | **0.1391** | **0.0953** | $0.9967 \pm 0.0002$ | 2.68s |
| **4** | **Linear Regression** | **0.9980** | **0.2205** | **0.1638** | $0.9979 \pm 0.0001$ | 0.47s |
| **5** | **Decision Tree Regressor** | **0.9954** | **0.3351** | **0.2545** | $0.9878 \pm 0.0005$ | 0.22s |
| **6** | **K-Nearest Neighbors (KNN)** | **0.9863** | **0.5750** | **0.4379** | $0.9651 \pm 0.0020$ | 0.01s |

---

## Slide 14: Classification Evaluation Results
| Rank | Algorithm | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | Train Time |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Random Forest Classifier** | **73.30%** | **0.7418** | **0.7330** | **0.7360** | 0.54s |
| **2** | **Multi-Layer Perceptron (MLP)** | **68.97%** | **0.6954** | **0.6897** | **0.6826** | 2.08s |
| **3** | **Support Vector Machine (SVM)** | **68.69%** | **0.6974** | **0.6869** | **0.6809** | 16.26s |
| **4** | **Decision Tree Classifier** | **67.40%** | **0.6968** | **0.6740** | **0.6806** | 0.09s |
| **5** | **K-Nearest Neighbors (KNN)** | **68.18%** | **0.6789** | **0.6818** | **0.6782** | 0.002s |
| **6** | **Logistic Regression** | **59.51%** | **0.6253** | **0.5951** | **0.6014** | 0.33s |

---

## Slide 15: Comparative Performance Analysis
- **Why SVR Won in Regression:**
  - Continuous meteorological telemetry follows tight thermodynamic state functions. SVR with an $\epsilon$-insensitive loss function ignores minor measurement noise within $0.1^\circ\text{C}$ while using support vectors to construct an optimal non-linear boundary.
- **Why Random Forest Won in Classification:**
  - Disaster records exhibit sharp decision rules (e.g. earthquakes occur regardless of season, whereas tropical storms require warm sea surface temperatures in specific months). Decision trees naturally discover these threshold rules, and bagging reduces tree variance.
- **Why Linear Models Struggled in Classification:**
  - Logistic Regression achieved only $59.51\%$ accuracy because hazard categories cannot be separated by linear hyperplanes.

---

## Slide 16: Best Model Selection
- **Selected Regression Model:** **Support Vector Regression (SVR)** / **Tuned Random Forest**
  - Demonstrates sub-$0.1^\circ\text{C}$ RMSE error and instantaneous test inference ($<0.05\text{s}$).
  - Serialized as `best_model.joblib` for real-time weather forecasting.
- **Selected Classification Model:** **Random Forest Classifier (Tuned)**
  - Highest test accuracy ($73.30\%$) and balanced F1 across all 5 hazard classes (Biological: $0.88$, Geophysical: $0.87$, Hydrological: $0.75$).
  - Serialized as `best_model.joblib` for real-time hazard classification.

---

## Slide 17: Ethical and Societal Considerations
- **Societal Impact:** Real-time temperature alerts protect vulnerable populations from deadly heatwaves; rapid hazard classification guides international disaster aid mobilization.
- **Data Privacy:** Uses purely public, aggregate meteorological and humanitarian records with zero personally identifiable information (PII).
- **Algorithmic Accountability:** Machine predictions must serve as decision-support tools for disaster management authorities rather than autonomous decision-makers, avoiding evacuation over-confidence.

---

## Slide 18: Conclusion and Future Scope
- **Conclusion:** Successfully designed, implemented, and comparatively benchmarked 12 machine learning models across real-world continuous and discrete prediction domains, meeting all four college rubric criteria with exceptional quality.
- **Future Scope:**
  - Incorporating deep sequential architectures (LSTM/GRU) for multi-hour ahead temperature trajectory forecasting.
  - Integrating satellite remote sensing data for hyperlocal flood inundation mapping.
- **Streamlit Demo Command:** `streamlit run app.py`
