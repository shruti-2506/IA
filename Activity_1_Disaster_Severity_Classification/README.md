# Activity 1: Disaster Severity Classification Using Machine Learning Algorithms

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

> **Course Code:** 2304322T  
> **Course:** Artificial Intelligence and Machine Learning  
> **Activity:** Classification Assignment Using Machine Learning Algorithms  
> **Student:** Shruti Siddheshwar Thorat  
> **PRN:** 202401040053  
> **Roll No:** 16 | **Batch:** A1  

---

## 📌 Project Overview

This activity focuses on developing, benchmarking, and evaluating four supervised machine learning classification algorithms to predict the severity category of natural and technological disaster events across the globe.

Disasters are categorized into multi-class severity grades (`Minor`, `Moderate`, `Severe`, `Catastrophic`) based on multi-dimensional socio-economic and demographic damage indicators.

---

## 📊 Dataset Details

- **Dataset File:** `global_classification_dataset.csv`
- **Total Records:** 14,644 global disaster events (spanning 1970–2021)
- **Target Variable:** `Disaster_Severity_Class` (Multi-class: `Minor`, `Moderate`, `Severe`, `Catastrophic`)
- **Key Features:**
  - Geographic: `Country`, `Region`, `Continent`
  - Temporal: `Start Year`, `Start Month`, `Start Day`, `End Year`, `End Month`
  - Event Typology: `Disaster Group`, `Disaster Subgroup`, `Disaster Type`, `Disaster Subtype`
  - Impact Metrics: Total Deaths, Affected Population, Injured, Homeless, Total Damage (USD)

---

## ⚙️ Data Preprocessing & Pipeline

1. **Data Cleaning & Leakage Prevention:**
   - Excluded unique identifier `Dis No`.
   - Removed direct target leakage features (e.g., `High_Severity_Flag`).
2. **Missing Value Imputation:**
   - Median imputation for numerical features (robust against extreme outliers in casualty and loss metrics).
   - Most-frequent (mode) imputation for categorical attributes.
3. **Feature Transformation:**
   - Standardization via `StandardScaler` for all continuous numerical features.
   - One-Hot Encoding (`OneHotEncoder`) for high-cardinality nominal categorical features.
   - Dense array representation enabled for Gaussian probabilistic modeling.
4. **Data Splitting:**
   - Stratified 80:20 Train-Test Split ensuring identical class proportion distributions in training and validation sets.

---

## 🤖 Algorithms Implemented

| Algorithm | Implementation Details | Key Characteristics |
|---|---|---|
| **Support Vector Machine (SVM)** | `LinearSVC` wrapped in `CalibratedClassifierCV` | Finds optimal maximum-margin hyperplane; probability calibration enabled for multiclass log-loss evaluation. |
| **Decision Tree Classifier** | `DecisionTreeClassifier(random_state=42)` | Non-parametric hierarchical rule-based decision paths; handles non-linear relationships with high interpretability. |
| **K-Nearest Neighbors (KNN)** | `KNeighborsClassifier(n_neighbors=5, metric='minkowski')` | Instance-based local geometric classification; relies heavily on standardized Euclidean distance. |
| **Gaussian Naïve Bayes** | `GaussianNB()` | Probabilistic conditional independence framework; fast training with likelihood density estimation. |

---

## 📈 Evaluation Metrics

All models are systematically benchmarked across:
- **Accuracy:** Overall proportion of correct severity class predictions.
- **Precision (Macro & Weighted):** Exactness measure per severity tier.
- **Recall (Macro & Weighted):** Sensitivity to high-mortality catastrophic events.
- **F1-Score (Macro & Weighted):** Harmonic balance between Precision and Recall.
- **Log Loss (Cross-Entropy Loss):** Calibration quality of predicted class probability vectors.
- **Confusion Matrix:** True Positives, False Positives, and cross-tier misclassifications.

---

## 📂 Included Files

```
Activity_1_Disaster_Severity_Classification/
├── Classification_Assignment_Disaster_Severity.ipynb   # Complete executable Google Colab / Jupyter Notebook
├── Disaster_Severity_Classification_Report.pdf         # Comprehensive academic report document
├── global_classification_dataset.csv                   # Historical worldwide disaster dataset
└── README.md                                           # Documentation and execution guide
```

---

## 🚀 Execution Instructions

### Option 1: Run in Google Colab
1. Upload `Classification_Assignment_Disaster_Severity.ipynb` and `global_classification_dataset.csv` to your Google Drive.
2. Open the notebook in Google Colab.
3. Set the `DATA_PATH` variable to your CSV location:
   ```python
   DATA_PATH = "global_classification_dataset.csv"
   ```
4. Click **Runtime > Run all**.

### Option 2: Run Locally
```bash
# Clone the IA repository
git clone https://github.com/shruti-2506/IA.git
cd IA/Activity_1_Disaster_Severity_Classification

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# Launch notebook
jupyter notebook Classification_Assignment_Disaster_Severity.ipynb
```
