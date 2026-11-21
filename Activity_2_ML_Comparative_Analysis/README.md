# Machine Learning Comparative Analysis: Real-Time Weather Forecasting and Global Disaster Classification

**Course Code:** 2304322T | **Course:** Artificial Intelligence and Machine Learning  
**Activity No.:** Activity No. 2: Implementation and Performance Evaluation with Comparative Analysis of Machine Learning Models  
**Department:** Computer Engineering | **Institute:** MIT Academy of Engineering, Alandi (D), Pune - 412105  
**Academic Year:** 2026–2027 | **Semester:** V (TY BTech)  

**Candidate Information:**  
- **Student Name:** Shruti Siddheshwar Thorat  
- **PRN Number:** 202401040053  
- **Roll Number:** 16 (Batch: A1)  
- **Group Members:**  
  1. Rohit Gadade (PRN: 202401040037 / 38)  
  2. Hrishikesh Koshti (PRN: 202401040054)  
  3. Shruti Thorat (PRN: 202401040053)  
  4. Vaishnavi Suryawanshi (PRN: 202401040069)    

---

## 1. Project Overview

This project delivers an end-to-end Machine Learning system benchmarked across two real-time applications:
1. **Regression Application:** Ambient 2-meter air temperature forecasting using $42,360$ hourly Open-Meteo observations for Pune, India ($18.52^\circ\text{N}, 73.87^\circ\text{E}$).
2. **Classification Application:** Multi-class environmental hazard subgroup prediction using $14,644$ worldwide historical disaster records from the EM-DAT International Disaster Database ($1970–2021$).

Twelve distinct Machine Learning algorithms (6 regression + 6 classification) were trained, evaluated, cross-validated, tuned via `GridSearchCV`, and deployed within an interactive Streamlit web application.

---

## 2. Directory Structure

```
ML_Comparative_Analysis/
│
├── datasets/
│   ├── meteo/
│   │   └── open-meteo-18.52N73.87E561m.csv        (42,360 hourly records, Pune, India)
│   └── global/
│       └── global_classification_dataset.csv     (14,644 worldwide disaster events)
│
├── regression/
│   ├── data_exploration.py                       (EDA, histograms, boxplots, correlations)
│   ├── preprocessing.py                          (Cyclical encoding, scaling, train/test split)
│   ├── model_training.py                         (Trains 6 regression models & GridSearchCV tuning)
│   ├── evaluation.py                             (Computes MSE, RMSE, MAE, R2, residual plots)
│   └── regression_results/
│       ├── models/                               (Serialized Joblib models & preprocessor)
│       │   ├── best_model.joblib
│       │   ├── support_vector_regression.joblib
│       │   ├── random_forest.joblib
│       │   ├── random_forest_tuned.joblib
│       │   ├── linear_regression.joblib
│       │   ├── decision_tree.joblib
│       │   ├── k_nearest_neighbors.joblib
│       │   ├── multi_layer_perceptron.joblib
│       │   └── preprocessor.joblib
│       ├── metrics.csv                           (Comparative performance table)
│       ├── model_ranking.csv                     (R2 ranking)
│       └── tuning_comparison.csv                 (Before vs After hyperparameter tuning)
│
├── classification/
│   ├── data_exploration.py                       (Target distribution, feature histograms)
│   ├── preprocessing.py                          (One-hot encoding, stratified 80:20 split)
│   ├── model_training.py                         (Trains 6 classification models & GridSearchCV)
│   ├── evaluation.py                             (Accuracy, F1, confusion matrix, ROC curves)
│   └── classification_results/
│       ├── models/                               (Serialized Joblib models & preprocessor)
│       │   ├── best_model.joblib
│       │   ├── random_forest.joblib
│       │   ├── random_forest_tuned.joblib
│       │   ├── logistic_regression.joblib
│       │   ├── decision_tree.joblib
│       │   ├── support_vector_machine.joblib
│       │   ├── k-nearest_neighbors.joblib
│       │   ├── multi-layer_perceptron.joblib
│       │   └── preprocessor.joblib
│       ├── metrics.csv                           (Comparative performance table)
│       ├── model_ranking.csv                     (Accuracy / F1 ranking)
│       ├── classification_report.txt             (Per-class Precision, Recall, F1)
│       └── tuning_comparison.csv                 (Before vs After hyperparameter tuning)
│
├── comparison/
│   ├── regression_comparison.py                  (In-depth bias-variance & error analysis)
│   └── classification_comparison.py              (Multi-class synthesis & tradeoff analysis)
│
├── visualizations/
│   ├── regression/                               (High-resolution 300 DPI evaluation figures)
│   │   ├── target_distribution.png
│   │   ├── histograms.png
│   │   ├── boxplots.png
│   │   ├── correlation_heatmap.png
│   │   ├── scatter_plots.png
│   │   ├── model_r2_comparison.png
│   │   ├── model_rmse_mae_comparison.png
│   │   ├── scatter_actual_vs_predicted.png
│   │   ├── residuals_plot.png
│   │   └── training_time_comparison.png
│   └── classification/                           (High-resolution 300 DPI evaluation figures)
│       ├── target_class_distribution.png
│       ├── feature_histograms.png
│       ├── correlation_heatmap.png
│       ├── subgroup_by_continent.png
│       ├── model_accuracy_comparison.png
│       ├── precision_recall_f1_comparison.png
│       ├── confusion_matrices.png
│       ├── roc_curves.png
│       └── training_time_comparison.png
│
├── reports/
│   ├── Activity2_ML_Comprehensive_Report.md      (26-section detailed college report)
│   ├── peer_assessed_quiz.md                     (10 MCQs with solutions & rubrics)
│   └── team_contribution.md                      (Role allocations & rubric mappings)
│
├── presentation/
│   ├── presentation_slides.md                    (18 slides in markdown format)
│   └── presentation_slides.html                  (Interactive presentation deck)
│
├── app.py                                        (Interactive Streamlit web application)
├── requirements.txt                              (Python library dependencies)
└── README.md                                     (Project documentation guide)
```

---

## 3. Technologies Used

- **Python 3.12+**
- **Scikit-Learn:** Preprocessing, model training, cross-validation, hyperparameter tuning (`GridSearchCV`), and metrics.
- **Pandas & NumPy:** Data loading, manipulation, cyclical trigonometric transformation, and statistical calculations.
- **Matplotlib & Seaborn:** Publication-quality 300 DPI visual analytics.
- **Joblib:** Model persistence and serialization.
- **Streamlit:** Interactive web application for real-time model inference.

---

## 4. Installation & Execution Guide

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Regression Pipeline
```bash
# Run EDA
python regression/data_exploration.py

# Run Preprocessing & Splitting
python regression/preprocessing.py

# Train & Tune 6 Regression Models
python regression/model_training.py

# Evaluate & Generate Charts
python regression/evaluation.py

# Generate Comparative Synthesis
python comparison/regression_comparison.py
```

### Step 3: Run Classification Pipeline
```bash
# Run EDA
python classification/data_exploration.py

# Run Preprocessing & Splitting
python classification/preprocessing.py

# Train & Tune 6 Classification Models
python classification/model_training.py

# Evaluate & Generate Confusion Matrices/ROC
python classification/evaluation.py

# Generate Comparative Synthesis
python comparison/classification_comparison.py
```

### Step 4: Launch the Interactive Streamlit Web App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 5. Performance Summary

### Regression Benchmark (Open-Meteo Weather Dataset)
- **Primary Target:** `temperature_2m (°C)`
- **Top Model:** **Support Vector Regression (SVR)**
  - **$R^2$ Score:** **$0.9996$**
  - **RMSE:** **$0.0942^\circ\text{C}$**
  - **MAE:** **$0.0704^\circ\text{C}$**
  - **5-Fold CV $R^2$:** **$0.9986 \pm 0.0002$**
  - **Random Forest Tuned:** $R^2 = 0.9995$, $\text{RMSE} = 0.1130^\circ\text{C}$.

### Classification Benchmark (Global Disaster Dataset)
- **Primary Target:** `Disaster_Subgroup_Cleaned` (5 Classes)
- **Top Model:** **Random Forest Classifier**
  - **Test Accuracy:** **$73.30\%$**
  - **Weighted F1-Score:** **$0.7360$** (Tuned: **$0.7366$**)
  - **Macro F1-Score:** **$0.7417$**
  - **Per-Class F1:** Biological: $0.88$, Geophysical: $0.87$, Hydrological: $0.75$, Meteorological: $0.69$, Climatological: $0.54$.
