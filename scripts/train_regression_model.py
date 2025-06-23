import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path

# --- Load cleaned dataset ---
df = pd.read_csv("C:/Users/Derrick/IdeaProjects/credit-risk-modeling/data/accepted_cleaned.csv")  # make sure this file exists

# --- Prepare features and target ---
X = df.drop(columns=["loan_status_binary"])
y = df["loan_status_binary"]

# --- Train/test split with class stratification ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Identify column types ---
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols_all = X.select_dtypes(include=["object"]).columns.tolist()
cat_cols = [col for col in cat_cols_all if X[col].nunique() <= 100]
dropped_cols = [col for col in cat_cols_all if X[col].nunique() > 100]
print("Dropped high-cardinality columns:", dropped_cols)

# Drop those high-cardinality columns
X_train = X_train.drop(columns=dropped_cols)
X_test = X_test.drop(columns=dropped_cols)

# --- Define preprocessing steps ---
# Fill missing values + scale numerical
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Fill missing values + encode categoricals
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Combine into one preprocessor
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# --- Final pipeline ---
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

# --- Train model ---
print("Training model...")
pipeline.fit(X_train, y_train)

# --- Evaluate ---
print("\nModel Performance:")
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# --- Save model ---
Path("models").mkdir(exist_ok=True)
joblib.dump(pipeline, "models/logreg_model_with_all_features.pkl")
print("\nModel saved to models/logreg_model_with_all_features.pkl")
