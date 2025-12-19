# test_pipeline.py
from ML_Engine import normalize_structured_input, build_feature_vector, run_pipeline as ml_run_pipeline

# ==================================================
# MEDICAL ABBREVIATION EXPLANATIONS
# ==================================================

ABBREVIATION_MAP = {
    "WBC": "White Blood Cell count – measures immune cells",
    "RBC": "Red Blood Cell count – measures oxygen-carrying cells",
    "HB": "Hemoglobin – protein that carries oxygen in blood",
    "PLATELET": "Platelet Count – helps blood clotting",
}

# ==================================================
# DEFAULT STRUCTURED INPUT
# ==================================================

DEFAULT_STRUCTURED_INPUT = {
    "patient_metadata": {
        "name": "Sample Report",
        "age": "35 YRS",
        "gender": "F",
        "patient_id": "N/A",
        "patient_address": "N/A"
    },
    "laboratory_info": {
        "lab_name": "Innovative Diagnostic Hong Kong",
        "lab_address": "Central, Hong Kong",
    },
    "test_results": [
        {"test_name": "WBC COUNT", "value": "5.5", "unit": "x10^9/L"},
        {"test_name": "EOSINOPHILS %", "value": "7.32", "unit": "%"},
        {"test_name": "HEMOGLOBIN", "value": "15.2", "unit": "g/dL"},
        {"test_name": "RBC COUNT", "value": "4.99", "unit": "million/µL"},
        {"test_name": "PLATELET COUNT", "value": "161", "unit": "x10^9/L"}
    ],
    "clinical_remarks": "Eosinophils percentage is slightly high."
}

# ==================================================
# HELPER FUNCTIONS
# ==================================================

def pretty_marker(key):
    return key.replace("_", " ").upper()

# ==================================================
# MAIN PIPELINE FUNCTION (for server.py integration)
# ==================================================

def run_pipeline(structured_input=None):
    """
    Main pipeline function that processes structured medical data
    and RETURNS a formatted analysis report as a string.
    """
    if structured_input is None:
        structured_input = DEFAULT_STRUCTURED_INPUT

    # Run ML processing
    patient, clinical_info = normalize_structured_input(structured_input)
    report, final_risk = ml_run_pipeline(structured_input)
    build_feature_vector(patient, clinical_info) 

    # Initialize list to hold output lines
    output_lines = []

    # 1. Pathology Observations
    output_lines.append("Pathology Observations")
    output_lines.append("-" * 40)

    used_markers = set()
    for obs in clinical_info["observations"]:
        label = pretty_marker(obs["marker"])
        used_markers.add(label.split()[0])
        output_lines.append(f"- {label} → {obs['value']} ({obs['status']})")

    # 2. Abnormal Findings
    output_lines.append("\nAbnormal Findings")
    output_lines.append("-" * 40)

    if not clinical_info["abnormal_findings"]:
        output_lines.append("No abnormal findings detected.")
    else:
        for a in clinical_info["abnormal_findings"]:
            label = pretty_marker(a["parameter"])
            used_markers.add(label.split()[0])
            output_lines.append(f"• {label} - {a['value']} ({a['status']})")

    # 3. Medical Term Explanations
    output_lines.append("\nMedical Term Explanations")
    output_lines.append("-" * 40)

    found = False
    for m in sorted(used_markers):
        if m in ABBREVIATION_MAP:
            output_lines.append(f"{m:<8} : {ABBREVIATION_MAP[m]}")
            found = True

    if not found:
        output_lines.append("No medical abbreviations to explain.")

    # 4. Final Health Assessment
    output_lines.append("\nFinal Health Assessment")
    output_lines.append("-" * 40)
    output_lines.append(str(final_risk))

    # Join list into a single string with newlines and return it
    return "\n".join(output_lines)

if __name__ == "__main__":
    # When running directly, print the returned string
    result = run_pipeline()
    print("Actual starts here")
    print(result)