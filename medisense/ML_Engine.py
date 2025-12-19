import re
import joblib

# ==================================================
# NORMAL RANGES (Clinical Safety Layer)
# ==================================================

NORMAL_RANGES = {
    "hemoglobin": (12.0, 17.5),
    "rbc_count": (4.5, 5.9),
    "wbc_count": (4.5, 11.0),
    "platelet_count": (150, 400),
    "crp": (0.0, 3.0),
    "esr": (0, 20),
    "glucose_fasting": (70, 100),
    "creatinine": (0.6, 1.3),
}

# ==================================================
# RANDOM FOREST MODEL LOADER
# ==================================================

class RiskModel:
    def __init__(self, model_path="offline_model/risk_model_v2_clinical.pkl"):
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load model: {e}")

    def predict(self, feature_vector):
        pred = self.model.predict([feature_vector])[0]
        prob = self.model.predict_proba([feature_vector])[0]
        return ["LOW", "MEDIUM", "HIGH"][pred], prob


# ==================================================
# INPUT HELPERS
# ==================================================

def extract_age(age_raw):
    match = re.search(r"\d+", str(age_raw))
    return int(match.group()) if match else 0


def extract_numeric(value):
    try:
        return float(re.findall(r"\d+\.?\d*", str(value))[0])
    except:
        return None


# ==================================================
# NORMALIZE STRUCTURED INPUT
# ==================================================

def normalize_structured_input(structured_data):
    patient_features = {
        "age": extract_age(
            structured_data.get("patient_metadata", {}).get("age")
        )
    }

    clinical_info = {
        "observations": [],
        "abnormal_findings": [],
    }

    test_results = structured_data.get("test_results", [])

    for test in test_results:
        name = test.get("test_name", "").upper()
        raw_value = test.get("value", "")
        unit = test.get("unit", "")

        numeric_value = extract_numeric(raw_value)
        key = name.lower().replace(" ", "_")

        status = "Normal"

        if numeric_value is not None and key in NORMAL_RANGES:
            lo, hi = NORMAL_RANGES[key]
            if numeric_value < lo:
                status = "LOW"
            elif numeric_value > hi:
                status = "HIGH"

        display_value = f"{raw_value} {unit}".strip()

        clinical_info["observations"].append({
            "marker": key,
            "numeric": numeric_value,
            "value": display_value,
            "status": status
        })

        if status in ["LOW", "HIGH"]:
            clinical_info["abnormal_findings"].append({
                "parameter": key,
                "value": display_value,
                "status": status
            })

    return patient_features, clinical_info


# ==================================================
# FEATURE ENGINEERING (MATCHES TRAINING EXACTLY)
# ==================================================

def build_feature_vector(patient_features, clinical_info):
    """
    Build feature vector that matches the training features exactly
    Order: age, hemoglobin, wbc_count, platelet_count, crp, esr, glucose_fasting, creatinine, low_count, high_count, severity_score
    """
    lab = {}

    for obs in clinical_info["observations"]:
        if obs["numeric"] is not None:
            lab[obs["marker"]] = obs["numeric"]

    low_count = 0
    high_count = 0
    severity_score = 0

    for a in clinical_info["abnormal_findings"]:
        if a["status"] == "LOW":
            low_count += 1
            severity_score += 1
        elif a["status"] == "HIGH":
            high_count += 1
            severity_score += 2

    # 🚨 EXACT ORDER — MUST MATCH TRAINING FEATURES
    return [
        patient_features.get("age", 0),
        lab.get("hemoglobin", 0),
        lab.get("wbc_count", 0),
        lab.get("platelet_count", 0),
        lab.get("crp", 0),
        lab.get("esr", 0),
        lab.get("glucose_fasting", 0),
        lab.get("creatinine", 0),
        low_count,
        high_count,
        severity_score
    ]


# ==================================================
# FINAL PIPELINE
# ==================================================

def run_pipeline(structured_input):
    patient_features, clinical_info = normalize_structured_input(structured_input)

    feature_vector = build_feature_vector(patient_features, clinical_info)

    model = RiskModel()
    ml_risk, confidence = model.predict(feature_vector)

    # 🔒 Clinical override
    if severity_score := feature_vector[-1] >= 4:
        final_risk = "HIGH"
        reason = "High cumulative severity score"
    elif feature_vector[-2] >= 2:
        final_risk = "HIGH"
        reason = "Multiple abnormal lab values"
    else:
        final_risk = ml_risk
        reason = "ML risk estimation"

    report = f"""
🩺 MEDICAL AUDIT SUMMARY
==================================================
FINAL RISK: {final_risk}

ML SUGGESTED RISK: {ml_risk}
CONFIDENCE SCORES: {confidence}

DECISION LOGIC:
- {reason}

PATHOLOGY OBSERVATIONS:
"""

    for obs in clinical_info["observations"]:
        report += f"- {obs['marker'].upper()}: {obs['value']} ({obs['status']})\n"

    return report, final_risk
