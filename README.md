# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer will leave the company (churn), using the Telco Customer Churn dataset (7,043 customers).

## Steps

1. Data cleaning (dropped `customerID`, fixed blank `TotalCharges` values)
2. EDA with charts
3. Encoding categorical columns (Yes/No mapping and one-hot encoding)
4. Train/test split (80/20, stratified)
5. Scaling with MinMaxScaler (fit on train data only)
6. Handling class imbalance with SMOTE (train data only)
7. Training Logistic Regression, SVM and Random Forest, with and without SMOTE
8. 5-fold cross validation of the best model
9. Confusion matrix, feature importance and saving the model

## Results

| Model | Data | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | Normal | 0.799 | 0.642 | 0.551 | 0.593 |
| Logistic Regression | SMOTE | 0.739 | 0.505 | 0.789 | 0.616 |
| SVM | Normal | 0.789 | 0.628 | 0.497 | 0.555 |
| SVM | SMOTE | 0.759 | 0.535 | 0.690 | 0.603 |
| Random Forest | Normal | 0.803 | 0.659 | 0.532 | 0.589 |
| **Random Forest** | **SMOTE** | **0.773** | **0.555** | **0.725** | **0.629** |

Random Forest with SMOTE gave the best F1 score. It catches 271 out of 374 churn customers in the test set, and 5-fold cross validation gives a similar F1 (0.625).

Contract type, tenure and monthly charges are the most important features.

## Tech Used

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook churn_prediction.ipynb
```

## Dataset

[IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) dataset, stored in `data/Telco-Customer-Churn.csv`.
