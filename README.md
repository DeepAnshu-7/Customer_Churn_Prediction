# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer will leave the company (churn), using the Telco Customer Churn dataset (7,043 customers).

## Steps

1. Data cleaning (dropped `customerID`, fixed blank `TotalCharges` values)
2. EDA with charts
3. Encoding Yes/No columns as 1/0
4. Train/test split (80/20, stratified)
5. Pipeline with MinMaxScaler, SMOTENC (for class imbalance) and one-hot encoding, so every step learns only from training data
6. Comparing Logistic Regression, SVM and Random Forest with no balancing, SMOTE and class weight, using 5-fold cross validation
7. Hyperparameter tuning of the best model with GridSearchCV
8. Testing the final model on the test set
9. Checking the decision threshold with cross validation predictions on the training data (0.5 gave the best F1)
10. Feature importance and saving the model

## Results

5-fold cross validation on the training data:

| Model | Data | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| **Random Forest** | **SMOTE** | **0.779** | **0.565** | **0.728** | **0.636** | **0.840** |
| Random Forest | Class Weight | 0.767 | 0.544 | 0.760 | 0.634 | 0.844 |
| Logistic Regression | Class Weight | 0.749 | 0.518 | 0.798 | 0.628 | 0.845 |
| Logistic Regression | SMOTE | 0.758 | 0.531 | 0.767 | 0.627 | 0.840 |
| SVM | Class Weight | 0.748 | 0.517 | 0.783 | 0.623 | 0.826 |
| SVM | SMOTE | 0.764 | 0.542 | 0.714 | 0.616 | 0.818 |
| Logistic Regression | Normal | 0.801 | 0.651 | 0.540 | 0.590 | 0.846 |
| SVM | Normal | 0.802 | 0.665 | 0.512 | 0.578 | 0.807 |
| Random Forest | Normal | 0.800 | 0.664 | 0.505 | 0.573 | 0.842 |

Random Forest with SMOTE was tuned with GridSearchCV (best: `max_depth=10`, `min_samples_leaf=5`, `n_estimators=200`, cross validation F1 0.639).

Final model on the test set:

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0.764 | 0.542 | 0.719 | 0.618 | 0.838 |

It catches 269 out of 374 churn customers. Contract type, tenure and electronic check payment are the most important features.

## Tech Used

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn

## How to Run

Tested with Python 3.14.

```bash
pip install -r requirements.txt
jupyter notebook churn_prediction.ipynb
```

## Dataset

[IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) dataset, stored in `data/Telco-Customer-Churn.csv`.
