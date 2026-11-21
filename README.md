# IA - Artificial Intelligence & Machine Learning Coursework

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

> **Course Code:** 2304322T  
> **Course:** Artificial Intelligence & Machine Learning (In-Semester / Internal Assessment)  
> **Department:** Computer Engineering  
> **Institute:** MIT Academy of Engineering, Alandi (D), Pune - 412105  
> **Student:** Shruti Siddheshwar Thorat  
> **PRN:** 202401040053 | **Roll No:** 16 | **Batch:** A1  

---

## 📖 Repository Overview

This repository contains academic assignments, practical implementations, and advanced comparative machine learning projects completed as part of the **Artificial Intelligence & Machine Learning (IA)** curriculum:

1. **[Assignment 1: AI/ML Applications and PEAS Analysis](./Assignment_1_PEAS_Analysis/)**
   - Architectural study and formulation of rational intelligent agent systems.
   - Comprehensive **PEAS (Performance Measure, Environment, Actuators, Sensors)** framework applied across two domains:
     - *AI-Based Medical Diagnosis System* (Healthcare decision-support)
     - *AI-Based Smart Museum & Heritage Preservation System* (Cultural preservation & monitoring)
   - Detailed analysis of 5 intelligent agents, 5 distinct operational environments, 10 agent-environment combinations, societal benefits/risks, ethical concerns, legal ramifications, and comparative benchmarking.

2. **[Assignment 2: Feature Engineering & Data Visualization](./Assignment_2_Feature_Engineering/)**
   - End-to-end data pipeline implemented in Python & Google Colab.
   - Real-world web-scraped dataset from `books.toscrape.com`.
   - Data cleaning, currency sign stripping, rating normalization, and deduplication.
   - Advanced feature engineering: price interval binning, rating bracket categorization, ordinal label encoding, and one-hot encoding (`pd.get_dummies`).
   - Rich exploratory data analysis (EDA) using Matplotlib, Seaborn, and interactive Plotly visual analytics (Treemaps, Sunbursts, Sankey flow diagrams, Pairplots, and Heatmaps).

3. **[Activity 1: Disaster Severity Classification](./Activity_1_Disaster_Severity_Classification/)**
   - Multi-class classification system predicting disaster severity levels (`Minor`, `Moderate`, `Severe`, `Catastrophic`) on 14,644 historical disaster events from EM-DAT.
   - Implementation and evaluation of four classical machine learning algorithms:
     - **Support Vector Machine (Linear SVM)** with probability calibration.
     - **Decision Tree Classifier** for interpretable rule-based splitting.
     - **K-Nearest Neighbors (KNN)** with standardized distance metrics.
     - **Gaussian Naïve Bayes** for probabilistic classification.
   - Rigorous evaluation: Accuracy, Precision, Recall, F1-Score (Macro & Weighted), Log Loss, and Confusion Matrices.

4. **[Activity 2: ML Comparative Analysis & Streamlit Web Application](./Activity_2_ML_Comparative_Analysis/)**
   - Comprehensive comparative performance evaluation suite across two real-world tasks:
     - **Weather Temperature Regression:** 42,360 hourly Open-Meteo observations for Pune, India evaluated across 6 algorithms (Linear Regression, SVR, Decision Tree, Random Forest, KNN, MLP).
     - **Global Disaster Subgroup Classification:** 14,644 worldwide records evaluated across 6 algorithms (Logistic Regression, Linear SVM, Decision Tree, Random Forest, KNN, MLP).
   - Hyperparameter optimization using `GridSearchCV` with 5-fold cross-validation.
   - Full interactive **Streamlit Web Application** (`app.py`) for live prediction, model benchmarking, residual analysis, ROC curves, and dynamic data exploration.

---

## 📂 Repository Structure

```
IA/
├── Assignment_1_PEAS_Analysis/
│   ├── AI_ML_Applications_and_PEAS_Analysis.pdf    # Full academic report document
│   └── README.md                                   # Markdown summary of PEAS combinations & analysis
│
├── Assignment_2_Feature_Engineering/
│   ├── Assignment_2_Feature_Engineering.ipynb      # Executable Jupyter / Google Colab notebook
│   ├── feature_engineering.py                     # Standalone Python script
│   ├── Feature_Engineering_Report.pdf              # Execution analysis & visual output report
│   └── README.md                                   # Step-by-step pipeline documentation
│
├── Activity_1_Disaster_Severity_Classification/
│   ├── Classification_Assignment_Disaster_Severity.ipynb  # Executable Colab / Jupyter notebook
│   ├── Disaster_Severity_Classification_Report.pdf        # Academic assignment report
│   ├── global_classification_dataset.csv                  # 14,644 historical disaster records
│   └── README.md                                          # Detailed methodology & evaluation
│
├── Activity_2_ML_Comparative_Analysis/
│   ├── AIML_Activity_2_Comprehensive_Colab_Notebook.ipynb # Complete executable benchmark notebook
│   ├── AIML_Activity_2_Comprehensive_Report.pdf           # Comprehensive academic report document
│   ├── app.py                                             # Interactive multi-page Streamlit web app
│   ├── requirements.txt                                   # Application dependencies
│   ├── README.md                                          # Complete system guide & architecture
│   ├── classification/                                    # EDA, preprocessing, training & evaluation
│   ├── regression/                                        # EDA, cyclical encoding, training & evaluation
│   ├── comparison/                                        # Comparative tradeoff & error analysis
│   ├── datasets/                                          # Pune weather & global disaster datasets
│   ├── presentation/                                      # Interactive HTML & markdown slide decks
│   ├── reports/                                           # Academic DOCX/MD reports & quiz materials
│   └── visualizations/                                    # Pre-rendered evaluation charts & plots
│
├── .gitignore                                      # Exclusion rules for Python, caches & heavy models
├── requirements.txt                                # Python library dependencies
└── README.md                                       # Main repository documentation
```

---

## 🛠️ Technologies Used

| Category | Tools & Libraries |
|---|---|
| **Core Languages** | Python 3.8+, Markdown, HTML |
| **Data Manipulation** | Pandas, NumPy, SciPy |
| **Machine Learning** | Scikit-Learn, Joblib |
| **Interactive Web App** | Streamlit, Altair |
| **Data Visualization** | Matplotlib, Seaborn, Plotly Express & Graph Objects |
| **Environments** | Google Colab, Jupyter Notebook, Local CLI |
| **Version Control** | Git, GitHub |

---

## 💻 Installation & Execution Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/shruti-2506/IA.git
cd IA
```

### 2. Set Up a Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Running the Interactive Streamlit Web App (Activity 2)
```bash
cd Activity_2_ML_Comparative_Analysis
streamlit run app.py
```

### 5. Running Notebooks (Google Colab / Jupyter)
You can launch any of the notebooks directly in Jupyter:
```bash
jupyter notebook
```
Or upload them to [Google Colab](https://colab.research.google.com/):
- `Assignment_2_Feature_Engineering/Assignment_2_Feature_Engineering.ipynb`
- `Activity_1_Disaster_Severity_Classification/Classification_Assignment_Disaster_Severity.ipynb`
- `Activity_2_ML_Comparative_Analysis/AIML_Activity_2_Comprehensive_Colab_Notebook.ipynb`

---

## 👤 Author Information

- **Student Name:** Shruti Siddheshwar Thorat
- **PRN:** `202401040053`
- **Roll No:** `16` (Batch: `A1`)
- **Institution:** MIT Academy of Engineering, Pune
- **GitHub:** [@shruti-2506](https://github.com/shruti-2506)
