# 📊 Telco Customer Churn Prediction using Random Forest

This project aims to predict customer churn in a telecom company using a machine learning pipeline built with Random Forest. The solution includes full data preprocessing, EDA, class imbalance handling using SMOTE, and hyperparameter tuning via GridSearchCV.

---

## 📁 Dataset

- **Source:** IBM Sample Telco Churn Dataset
- **Rows:** 7,043
- **Target Variable:** `Churn` (Yes/No)
- **Features:** Customer demographics, service plans, usage, billing details, and more.

---

## 🎯 Project Goals

- Identify customers likely to churn
- Improve recall to reduce false negatives
- Build a robust and deployable ML pipeline

---

## 🛠️ Tools & Libraries

- Python (Pandas, NumPy)
- Scikit-learn
- Matplotlib, Seaborn
- SMOTE (from `imblearn`)
- `joblib` (for model saving)

---

## 🧪 Preprocessing & Feature Engineering

- Removed unnecessary columns (e.g., `customerID`)
- Converted binary and categorical variables
- Handled missing values
- SMOTE used to balance classes
- Split data into train-test sets with `stratify=y`

---

## 🔍 Model Training

### Model: `RandomForestClassifier`

### Hyperparameters tuned using `GridSearchCV`:
```python
param_grid = {
    'n_estimators': [100, 200],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
| Metric    | Value                  |
| --------- | ---------------------- |
| Accuracy  | \~80%                  |
| Precision | Good                   |
| Recall    | Improved (after SMOTE) |
| F1-Score  | Balanced               |

# Folder Structure

📁 Telco-Churn-Prediction
│
├── 📄 TChurn1.ipynb
├── 📄 telco_churn.pkl
├── 📄 WA_Fn-UseC_-Telco-Customer-Churn.csv
├── 📄 README.md
