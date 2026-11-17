# IA - Artificial Intelligence & Machine Learning Coursework

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

> **Course:** Artificial Intelligence & Machine Learning (In-Semester / Internal Assessment)  
> **Student:** Shruti Siddheshwar Thorat  
> **PRN:** 202401040053  
> **Roll No:** 16  

---

## 📖 Repository Overview

This repository contains academic assignments and experimental implementations completed as part of the **Artificial Intelligence & Machine Learning (IA)** curriculum:

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

---

## 📂 Repository Structure

```
IA/
├── Assignment_1_PEAS_Analysis/
│   ├── AI_ML_Applications_and_PEAS_Analysis.pdf    # Full academic report document
│   └── README.md                                   # Markdown summary of PEAS combinations & analysis
├── Assignment_2_Feature_Engineering/
│   ├── Assignment_2_Feature_Engineering.ipynb      # Executable Jupyter / Google Colab notebook
│   ├── feature_engineering.py                     # Standalone Python script
│   ├── Feature_Engineering_Report.pdf              # Execution analysis & visual output report
│   └── README.md                                   # Step-by-step pipeline documentation
├── .gitignore                                      # Standard exclusion rules for Python & data science
├── requirements.txt                                # Python library dependencies
└── README.md                                       # Main repository documentation
```

---

## 🛠️ Technologies Used

| Category | Tools & Libraries |
|---|---|
| **Core Language** | Python 3.8+ |
| **Data Manipulation** | Pandas, NumPy |
| **Data Visualization** | Matplotlib, Seaborn, Plotly Express & Graph Objects |
| **Execution Environments** | Google Colab, Jupyter Notebook, Local CLI |
| **Version Control** | Git, GitHub |

---

## 🚀 Key Features

- **Theoretical Rigor:** Detailed architectural specification of 10 PEAS matrix combinations across two complex autonomous agent systems.
- **Data Engineering Pipeline:** Transformation of raw wide-format scraped data into clean, machine-learning-ready feature matrices.
- **Multi-Level Visual Analytics:** Interactive and static visualizations highlighting categorical distributions, cross-tabulations, and correlation matrices.
- **Clean Code & Reproducibility:** Modular Python scripts and documented Jupyter Notebook with pre-rendered results.

---

## 💻 Installation & Execution Instructions

### Prerequisites
- Python 3.8 or higher installed on your system.
- Git installed.

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/IA.git
cd IA
```

### 2. Set Up a Virtual Environment (Recommended)
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

### 4. Running Assignment 2
**Via Jupyter Notebook:**
```bash
jupyter notebook Assignment_2_Feature_Engineering/Assignment_2_Feature_Engineering.ipynb
```
*(Or upload the notebook directly into [Google Colab](https://colab.research.google.com/))*

**Via Python Script:**
```bash
cd Assignment_2_Feature_Engineering
python feature_engineering.py
```

---

## 📄 License
This project is prepared for academic and educational evaluation.
