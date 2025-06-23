import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# --- Load cleaned dataset ---
df = pd.read_csv("data/accepted_cleaned.csv")  # make sure this file exists

# --- Prepare features and labels ---
X = df.drop(columns=["loan_status_binary"])
y = df["loan_status_binary"]

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Define numeric and categorical columns ---
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# --- Build preprocessing pipeline ---
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])

# --- Build full modeling pipeline ---
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
])

# --- Train the model ---
print("Training model...")
pipeline.fit(X_train, y_train)

# --- Evaluate ---
print("\nModel Performance:")
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# --- Save the model ---
joblib.dump(pipeline, "models/logreg_model_with_all_features.pkl")
print("\nModel saved to models/logreg_model_with_all_features.pkl")
