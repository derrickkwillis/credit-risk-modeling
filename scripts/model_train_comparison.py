import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import joblib
from pathlib import Path


# Load cleaned dataset
df = pd.read_csv("C:/Users/Derrick/IdeaProjects/credit-risk-modeling/data/accepted_cleaned.csv")  # make sure this file exists

# Drop high-cardinality columns to reduce memory load
high_card_cols = ['emp_title', 'issue_d', 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d']
df.drop(columns=high_card_cols, inplace=True)

# Features and target
X = df.drop(columns=["loan_status_binary"])
y = df["loan_status_binary"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Define numeric and categorical columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Define preprocessing pipelines
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols)
])

# Models to evaluate
models = {
    "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(scale_pos_weight=2, use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Train and evaluate each model
results = []
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    print(f"Training {name}...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)
    results.append({
        "Model": name,
        "Accuracy": report["accuracy"],
        "Precision (1)": report["1"]["precision"],
        "Recall (1)": report["1"]["recall"],
        "F1 (1)": report["1"]["f1-score"],
        "ROC AUC": auc
    })

# Show model comparison
results_df = pd.DataFrame(results)
print("\nModel Comparison:\n", results_df)

# Save test data for later use in visualization and predictions
X_test.to_csv("C:/Users/Derrick/IdeaProjects/credit-risk-modeling/data/X_test.csv", index=False)
y_test.to_csv("C:/Users/Derrick/IdeaProjects/credit-risk-modeling/data/y_test.csv", index=False)

# Ensure the models directory exists
Path("models").mkdir(exist_ok=True)

# Save each trained pipeline
joblib.dump(models["LogisticRegression"], "models/logreg_model.pkl")
joblib.dump(models["RandomForest"], "models/rf_model.pkl")
joblib.dump(models["XGBoost"], "models/xgb_model.pkl")

print("\nModels saved to the 'models' directory.")
