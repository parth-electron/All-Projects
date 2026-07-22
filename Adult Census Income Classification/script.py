
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --------------------------------------------------------------------------
# Task 1: Data Understanding
# --------------------------------------------------------------------------

DATA_PATH = "adult.csv"


def load_dataset(path=DATA_PATH, n_synthetic=3000):
    """Load the real Adult Census Income CSV if present, else build a
    synthetic stand-in with a similar schema and realistic relationships."""
    if os.path.exists(path):
        print(f"Loading real dataset from '{path}'")
        return pd.read_csv(path), False

    print(f"'{path}'")

    age = np.random.randint(17, 75, n_synthetic)
    workclass = np.random.choice(
        ["Private", "Self-emp-not-inc", "Local-gov", "State-gov",
         "Self-emp-inc", "Federal-gov"], n_synthetic,
        p=[0.70, 0.08, 0.07, 0.05, 0.05, 0.05]
    )
    education = np.random.choice(
        ["HS-grad", "Some-college", "Bachelors", "Masters", "Assoc-voc",
         "11th", "Doctorate"], n_synthetic,
        p=[0.32, 0.22, 0.17, 0.06, 0.06, 0.10, 0.07]
    )
    education_num = pd.Series(education).map({
        "11th": 7, "HS-grad": 9, "Assoc-voc": 11, "Some-college": 10,
        "Bachelors": 13, "Masters": 14, "Doctorate": 16
    }).values
    marital_status = np.random.choice(
        ["Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed"],
        n_synthetic, p=[0.46, 0.33, 0.14, 0.03, 0.04]
    )
    occupation = np.random.choice(
        ["Exec-managerial", "Prof-specialty", "Craft-repair", "Adm-clerical",
         "Sales", "Other-service", "Machine-op-inspct"], n_synthetic
    )
    relationship = np.random.choice(
        ["Husband", "Not-in-family", "Own-child", "Unmarried", "Wife"], n_synthetic
    )
    race = np.random.choice(
        ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
        n_synthetic, p=[0.85, 0.10, 0.03, 0.01, 0.01]
    )
    sex = np.random.choice(["Male", "Female"], n_synthetic, p=[0.67, 0.33])
    hours_per_week = np.random.randint(1, 100, n_synthetic).clip(1, 99)
    hours_per_week = np.round(np.random.normal(40, 12, n_synthetic)).clip(1, 99).astype(int)
    native_country = np.random.choice(
        ["United-States", "Mexico", "Philippines", "Germany", "India"],
        n_synthetic, p=[0.90, 0.03, 0.02, 0.02, 0.03]
    )
    capital_gain = np.where(np.random.rand(n_synthetic) < 0.08,
                             np.random.randint(1000, 20000, n_synthetic), 0)
    capital_loss = np.where(np.random.rand(n_synthetic) < 0.04,
                             np.random.randint(500, 3000, n_synthetic), 0)
    fnlwgt = np.random.randint(15000, 500000, n_synthetic)

    # Income class driven by known real-world patterns in this dataset
    income_score = (
        -6.0
        + 0.30 * education_num
        + 0.04 * age
        + 0.05 * (hours_per_week - 40)
        + 1.8 * (marital_status == "Married-civ-spouse")
        + 0.00006 * capital_gain
        + 0.9 * np.isin(occupation, ["Exec-managerial", "Prof-specialty"])
    )
    income_prob = 1 / (1 + np.exp(-income_score))
    income = np.where(np.random.rand(n_synthetic) < income_prob, ">50K", "<=50K")

    df = pd.DataFrame({
        "age": age, "workclass": workclass, "fnlwgt": fnlwgt,
        "education": education, "education.num": education_num,
        "marital.status": marital_status, "occupation": occupation,
        "relationship": relationship, "race": race, "sex": sex,
        "capital.gain": capital_gain, "capital.loss": capital_loss,
        "hours.per.week": hours_per_week, "native.country": native_country,
        "income": income,
    })
    return df, True


df, is_synthetic = load_dataset()

print("\n=== First five records ===")
print(df.head())

# --------------------------------------------------------------------------
# Task 1 (continued): identify feature types
# --------------------------------------------------------------------------

target_variable = "income"


df = df.replace("?", np.nan)

numerical_features = [c for c in df.select_dtypes(include=[np.number]).columns
                       if c != target_variable]
categorical_features = [c for c in df.columns
                         if c not in numerical_features + [target_variable]]

print("\n=== Feature types ===")
print("Numerical features   :", numerical_features)
print("Categorical features :", categorical_features)
print("Target variable      :", target_variable)

# --------------------------------------------------------------------------
# Task 2: Data Preprocessing
# --------------------------------------------------------------------------

print("\n=== Missing values per column (before handling) ===")
print(df.isnull().sum())

for col in numerical_features:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

for col in categorical_features:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

print("\n=== Missing values per column (after handling) ===")
print(df.isnull().sum())

df_encoded = df.copy()
label_encoders = {}
for col in categorical_features + [target_variable]:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

print("\n=== Target encoding map ===")
print(label_encoders[target_variable])

feature_columns = numerical_features + categorical_features
X = df_encoded[feature_columns]
y = df_encoded[target_variable]

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numerical_features] = scaler.fit_transform(X[numerical_features])

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTraining set size: {X_train.shape[0]} rows")
print(f"Testing set size : {X_test.shape[0]} rows")

# --------------------------------------------------------------------------
# Task 3: Model Development
# --------------------------------------------------------------------------

model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n=== Model coefficients ===")
for feature, coef in zip(feature_columns, model.coef_[0]):
    print(f"{feature:20s}: {coef:,.4f}")
print(f"{'intercept':20s}: {model.intercept_[0]:,.4f}")

# --------------------------------------------------------------------------
# Task 4: Model Evaluation
# --------------------------------------------------------------------------

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n=== Evaluation metrics ===")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-Score  : {f1:.4f}")
print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["<=50K", ">50K"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix — Adult Census Income Classification")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nSaved plot to confusion_matrix.png")

with open("metrics.txt", "w") as f:
    f.write(f"accuracy,{acc:.4f}\n")
    f.write(f"precision,{prec:.4f}\n")
    f.write(f"recall,{rec:.4f}\n")
    f.write(f"f1,{f1:.4f}\n")
    f.write(f"synthetic,{is_synthetic}\n")

print("\nDone.")
