import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from scipy.stats import randint as sp_randint

# =======================
# CONFIG
# =======================
DATA_PATH = "Car_Dataset.csv"
NUMERICAL_TARGET = "Insurance Premium"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# =======================
# LOAD DATA
# =======================
df = pd.read_csv(DATA_PATH)

if NUMERICAL_TARGET not in df.columns:
    raise ValueError(f"{NUMERICAL_TARGET} not found in dataset")

q1 = df[NUMERICAL_TARGET].quantile(0.33)
q2 = df[NUMERICAL_TARGET].quantile(0.66)


def premium_category(value):
    if value <= q1:
        return "Low"
    elif value <= q2:
        return "Medium"
    else:
        return "High"


df["insurance_premium_category"] = df[NUMERICAL_TARGET].apply(premium_category)

print("Category distribution:")
print(df["insurance_premium_category"].value_counts())

# =======================
# FEATURES & TARGET
# =======================
TARGET = "insurance_premium_category"
X = df.drop(columns=[TARGET, NUMERICAL_TARGET])
y = df[TARGET]

# Auto feature detection
num_features = X.select_dtypes(include=["number"]).columns.tolist()
cat_features = X.select_dtypes(exclude=["number"]).columns.tolist()

# =======================
# PREPROCESSING
# =======================
num_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

cat_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features),
    ]
)

# =======================
# MODEL + PIPELINE
# =======================
rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", rf),
    ]
)

# =======================
# HYPERPARAMETER SEARCH
# =======================
param_grid = {
    "model__n_estimators": sp_randint(150, 400),
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": sp_randint(2, 6),
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

# =======================
# TRAIN
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

search.fit(X_train, y_train)

best_model = search.best_estimator_

# =======================
# EVALUATION
# =======================
y_pred = best_model.predict(X_test)
print("\nBest Parameters:")
print(search.best_params_)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =======================
# SAVE MODEL
# =======================
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/car_insurance_model.pkl")

print("\n✅ Model saved as models/car_insurance_model.pkl")
