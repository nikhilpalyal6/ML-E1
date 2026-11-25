# Digital Fraud Detection (ML-E1)

An end-to-end Machine Learning pipeline designed to detect fraudulent digital transactions using behavioral indicators, anomaly detection, and classification models.

---

## 📌 Overview

Digital transaction fraud is an increasingly complex challenge for financial institutions and payment service providers. This project implements data preprocessing, exploratory data analysis (EDA), specialized feature engineering, and model training/evaluation to identify fraudulent activities with high accuracy and recall.

## 🚀 Key Features

- **Comprehensive Feature Engineering**: Behavioral risk indicators including high-amount flags, night/off-hours activity, velocity/frequency analysis, and risk scoring.
- **Exploratory Data Analysis (EDA)**: Visualizations of transaction amounts, frequencies, time distributions, and correlations using Matplotlib and Seaborn.
- **Classification Models**: Implementations of Logistic Regression and Random Forest classifiers with cross-validation.
- **Performance Evaluation**: Comprehensive evaluation using Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices.

---

## 📂 Project Structure

```text
├── Digital fraud Detection/     # Historical project archives & backup copies
├── feature_documentation.md     # Detailed documentation of engineered features
├── feature_engineering.py       # Core feature engineering module
├── feature_engineering_cell.py  # Notebook integration helper
├── fix_notebook.py              # Notebook maintenance utility script
├── update_notebook.py           # Notebook update & formatting utility script
├── fraud_detection_dataset.csv  # Transaction dataset
├── Project.ipynb                # Main Jupyter Notebook with full analysis & models
├── Project_test.ipynb           # Model testing notebook
├── Project.py                   # Python script version of the end-to-end pipeline
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore patterns
└── README.md                    # Project documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone the repository:
```bash
git clone https://github.com/nikhilpalyal6/ML-E1.git
cd ML-E1
```

### 2. Set up a virtual environment (optional but recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install required packages:
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Running the Jupyter Notebook
Launch Jupyter Lab or Notebook to view and interact with the full analysis:
```bash
jupyter notebook Project.ipynb
```

### Running the Python Script
Execute the standalone pipeline directly from the command line:
```bash
python Project.py
```

---

## 📊 Models & Results

The pipeline evaluates multiple machine learning algorithms:
- **Logistic Regression**: Linear baseline model with standard scaling.
- **Random Forest Classifier**: Ensemble decision tree model optimized for handling complex, non-linear feature interactions and class imbalance.

Refer to `feature_documentation.md` for in-depth insights into the behavioral risk features that drive model predictions.
