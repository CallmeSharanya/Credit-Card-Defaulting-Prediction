# 💳 Credit Card Default Prediction

<p align="center">
  <img src="https://img.shields.io/github/languages/code-size/CallmeSharanya/Credit-Card-Defaulting-Prediction?color=informational" />
  <img src="https://img.shields.io/github/last-commit/CallmeSharanya/Credit-Card-Defaulting-Prediction?color=success" />
  <img src="https://img.shields.io/github/languages/top/CallmeSharanya/Credit-Card-Defaulting-Prediction" />
</p>

## 📌 Overview

This project focuses on predicting whether a customer will default on their credit card payment in the next month. Using a machine learning pipeline and exploratory data analysis (EDA), we train and evaluate various models to identify potential defaulters, helping financial institutions manage credit risk more effectively.

---

## 📊 Dataset

The dataset used is the **UCI Credit Card Default Dataset**, which contains data on 30,000 customers with features such as:

- **Demographics** (age, gender, education, marital status)
- **Payment history**
- **Bill statements**
- **Previous payments**

📁 Download: [UCI Credit Card Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

---

## 🛠️ Features of the Project

- ✅ Data Cleaning & Preprocessing
- 📈 Exploratory Data Analysis (EDA)
- ⚙️ Feature Engineering
- 🤖 Model Building (Logistic Regression, Random Forest, XGBoost, etc.)
- 🧪 Model Evaluation (Accuracy, Precision, Recall, F1-Score, AUC)
- 🔍 Hyperparameter Tuning with GridSearchCV
- 📊 Visualizations using Seaborn, Matplotlib

---

## 🔍 Exploratory Data Analysis

We explore and visualize data to understand:

- Distribution of categorical & numerical features
- Correlation heatmaps
- Default vs Non-default patterns
- Payment trends and credit limits

---

## 🤖 Models Implemented

| Model | Accuracy | AUC Score |
|-------|----------|-----------|
| Logistic Regression | ✅ Good baseline model |
| Random Forest | 🔥 High accuracy |
| XGBoost | 💯 Best performance |
| Decision Tree | 🌲 Easy to interpret |

> 📌 *Models are evaluated using cross-validation and ROC curves.*

---

## 📂 Project Structure

```
Credit-Card-Defaulting-Prediction/
│
├── Dataset/
│   └── UCI_Credit_Card.csv
├── EDA.ipynb
├── Model_Building.ipynb
├── utils/
│   └── helper_functions.py
├── requirements.txt
└── README.md
```

---

## 🧪 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/CallmeSharanya/Credit-Card-Defaulting-Prediction.git
   cd Credit-Card-Defaulting-Prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the notebooks:
   Open `EDA.ipynb` and `Model_Building.ipynb` using Jupyter Notebook or VS Code.

---

## 📌 Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

(Full list in `requirements.txt`)

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome! Feel free to fork the repo and submit a pull request.

---

## ✨ Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- scikit-learn, matplotlib, seaborn, XGBoost
