# Customer Churn Prediction

A machine learning project for predicting customer churn using the Telco customer dataset. The notebook covers data preprocessing, exploratory data analysis (EDA), categorical encoding, feature scaling, class-imbalance handling with SMOTE, and comparisons across multiple classification approaches.

## Project Workflow

1. Load and inspect the customer churn dataset.
2. Remove the customerID column.
3. Clean and convert TotalCharges.
4. Perform exploratory data analysis and visualize churn patterns.
5. Encode categorical variables using binary encoding and one-hot encoding.
6. Scale numerical features using MinMaxScaler / StandardScaler.
7. Split the data into training and test sets.
8. Handle class imbalance using SMOTE.
9. Train and evaluate:
  * Support Vector Machine (SVM)
  * Random Forest
  * Neural Network using TensorFlow/Keras
  * Logistic Regression
  * XGBoost
10. Evaluate models using accuracy, precision, recall, F1-score, classification reports, and confusion matrices.
11. Compare balanced and unbalanced approaches, including 5-fold and 10-fold cross-validation.


## Dataset

The notebook expects the dataset file to be named:

```text
Dataset.csv
```

Place Dataset.csv in the same directory as the notebook before running it.

The notebook works with 7,043 customer records and predicts the binary Churn target.

> **Note:** This repository does not include the dataset file unless you have permission to redistribute it. Add it locally as Dataset.csv to run the notebook.

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn (SMOTE)
* TensorFlow / Keras
* XGBoost
* Jupyter Notebook

## Models and Evaluation

The project experiments with both classical machine learning and neural-network approaches. It also investigates the effect of class-imbalance handling and cross-validation on model performance.

The notebook reports metrics such as:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### Example Results

For the Logistic Regression experiments:

* 80/20 unbalanced split: 80.41% accuracy
* 5-fold unbalanced cross-validation: 80.65% average accuracy
* 5-fold balanced cross-validation: 80.08% average recall
* 10-fold balanced cross-validation: 80.05% average recall

For the other experiments, the notebook contains detailed classification reports and evaluation results for SVM, Random Forest, Neural Network, and XGBoost.

## How to Run

1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd customer-churn-prediction
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Add the dataset

Place `Dataset.csv` in the project directory.

4. Open the notebook

```bash
jupyter notebook Customer_Churn_Prediction.ipynb
```

or open it using JupyterLab:

```bash
jupyter lab
```

## Project Structure

```text
customer-churn-prediction/
├── Customer_Churn_Prediction.ipynb
├── README.md
├── requirements.txt
└── Dataset.csv              # not included by default
```
## Future Improvements

* Hyperparameter tuning for the evaluated models.
* More systematic model comparison
* ROC-AUC and PR-AUC analysis
* Feature-importance and model-interpretability analysis
* Deployment as a web application/API