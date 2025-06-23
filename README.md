# Credit Risk Modeling

This project focuses on predicting the likelihood of loan default using machine learning models. It simulates a common credit scoring use case in financial institutions, where data analysts assess applicant risk to guide lending decisions. The goal is to build, evaluate, and compare models that help identify high-risk borrowers effectively.

## Project Overview

The workflow includes:

- Data preprocessing
- Feature engineering
- Training multiple classification models
- Comparing model performance using standard evaluation metrics
- Generating predictions for new loan applicants

The models used in this project are Logistic Regression, Random Forest, and XGBoost.

## Methodology

### Data Processing
- Handled missing values using mean/mode imputation
- Encoded categorical variables with one-hot encoding
- Scaled numerical features using standardization
- Performed stratified train-test split to maintain class balance

### Modeling Strategy
- Logistic Regression: Simple, interpretable baseline
- Random Forest: Robust ensemble model that handles nonlinearities and feature importance
- XGBoost: High-performance gradient boosting algorithm, often used in structured data problems

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

These metrics were selected to assess both overall model performance and sensitivity to false negatives, which are especially costly in the context of loan defaults.

## Results

| Model              | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.82     | 0.65      | 0.58   | 0.61     | 0.78    |
| Random Forest       | 0.86     | 0.72      | 0.66   | 0.69     | 0.85    |
| XGBoost             | 0.88     | 0.76      | 0.70   | 0.73     | 0.87    |

XGBoost provided the best balance of precision and recall, along with the highest ROC-AUC score. It is the preferred model for deployment in a setting where identifying defaulters is critical.

## How to Use

1. Clone this repository:
   git clone https://github.com/derrickkwillis/credit-risk-modeling.git
   cd credit-risk-modeling

2. Install dependencies:
   pip install -r requirements.txt

3. Train the models:
   python scripts/train_regression_model.py

4. Evaluate model performance:
   python scripts/model_train_comparison.py

5. Generate predictions for new applicants:
   python scripts/predict.py

## Notes

- The `.pkl` model files are not included in this repository due to GitHub file size limits. You can recreate them by running the training script.
- Test set CSVs (e.g., `X_test.csv`, `y_test.csv`) are not included to avoid excee