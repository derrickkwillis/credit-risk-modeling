import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from pathlib import Path

# --- Load cleaned dataset ---
# Data already cleaned during EDA; avoids redundancy here
df = pd.read_csv("data/accepted_cleaned.csv")

# --- Prepare features and target ---
X = df.drop(columns=["loan_status_binary"])
y = df["loan_status_binary"]

# --- Train/test split with stratification for class balance ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Identify feature types ---
# Numerical and categorical features are handled separately for optimal preprocessing
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# --- Preprocessing pipeline ---
# Scales numerical features, one-hot encodes categoricals, and handles unknowns safely
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])

# --- Full model pipeline ---
# A pipeline ensures reproducibility and avoids data leakage; ideal for deployment
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        class_weight="balanced",  # Adjusts for class imbalance automatically
        max_iter=1000,            # Allows convergence for large datasets
        random_state=42
    ))
])

# --- Model training ---
# Using pipeline.fit ensures preprocessing is integrated with model training
print("Training model...")
pipeline.fit(X_train, y_train)

# --- Model evaluation ---
# AUC and classification report provide insight into predictive performance
print("\nModel Performance:")
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# --- Save the trained pipeline ---
# Saves entire pipeline (including preprocessing); makes model portable and reproducible
Path("models").mkdir(exist_ok=True)
joblib.dump(pipeline, "models/logreg_model_with_all_features.pkl")
print("\nModel saved to models/logreg_model_with_all_features.pkl")