import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ==============================================
# STEP 1: Load dataset
# ==============================================

DATA_PATH = "data/patients.csv"  # update if needed
df = pd.read_csv(DATA_PATH)

print("✅ Dataset loaded")
print("Total samples:", len(df))
print("Columns:", list(df.columns))

# ==============================================
# STEP 2: Define medical normal ranges
# ==============================================

NORMAL_RANGES = {
    "hemoglobin": (12.0, 17.5),
    "rbc_count": (4.5, 5.9),
    "wbc_count": (4500, 11000),
    "platelet_count": (150000, 400000),
    "crp": (0.0, 3.0),
    "esr": (0, 20),
    "glucose_fasting": (70, 100),
    "creatinine": (0.6, 1.3),
}

# ==============================================
# STEP 3: Compute engineered features IF missing
# ==============================================

def compute_engineered(row):
    low = high = severity = 0
    for col, (lo, hi) in NORMAL_RANGES.items():
        if col in row and pd.notna(row[col]):
            try:
                val = float(row[col])
                if val < lo:
                    low += 1
                    severity += 1
                elif val > hi:
                    high += 1
                    severity += 2
            except:
                pass
    return pd.Series([low, high, severity])

if not {"low_count", "high_count", "severity_score"}.issubset(df.columns):
    print("⚙️ Computing engineered features...")
    df[["low_count", "high_count", "severity_score"]] = df.apply(
        compute_engineered, axis=1
    )

# ==============================================
# STEP 4: Build feature list dynamically
# ==============================================

BASE_FEATURES = [
    "age",
    "hemoglobin",
    "rbc_count",
    "wbc_count",
    "platelet_count",
    "crp",
    "esr",
    "glucose_fasting",
    "creatinine",
]

ENGINEERED_FEATURES = [
    "low_count",
    "high_count",
    "severity_score",
]

FEATURE_COLS = [c for c in BASE_FEATURES if c in df.columns] + ENGINEERED_FEATURES

TARGET_COL = "risk_label"

if TARGET_COL not in df.columns:
    raise ValueError("❌ Target column 'risk_label' not found in dataset")

print("✅ Using features:", FEATURE_COLS)

X = df[FEATURE_COLS].copy()
y = df[TARGET_COL].copy()

# ==============================================
# STEP 5: Clean numeric data
# ==============================================

X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median(numeric_only=True))

# ==============================================
# STEP 6: Train / Test Split
# ==============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("📊 Training samples:", len(X_train))
print("📊 Testing samples:", len(X_test))

# ==============================================
# STEP 7: Train Random Forest
# ==============================================

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    min_samples_split=8,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==============================================
# STEP 8: Evaluate
# ==============================================

y_pred = model.predict(X_test)

print("\n🎯 Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\n📊 Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["LOW", "MEDIUM", "HIGH"]
    )
)

# ==============================================
# STEP 9: Save model
# ==============================================

os.makedirs("offline_model", exist_ok=True)
joblib.dump(model, "offline_model/risk_model_v2_clinical.pkl")
print("\n💾 Model saved as risk_model_v2_clinical.pkl in offline_model")
