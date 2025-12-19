from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import NLP_Engine  # Requires processor.py
import ML_Format as ML
import os
import ocr

app = Flask(__name__)
# Enable CORS so the HTML file (even if opened locally) can talk to this server
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Add a default route so you don't get a 404 if you visit the base URL
@app.route('/')
def home():
    # This looks for 'index.html' inside a 'templates' folder
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_report():
    # 1. Check if file is present
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    print(f"Received file: {file.filename}. Processing...")

    files_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(files_path)

    print(files_path)

    # 3. Return the specific JSON data structure you provided
    report = ocr.perform_structured_ocr(files_path)
    
    # Analyze the report using the processor
    response_data = NLP_Engine.analyse(report)

    # ADDED: Summary variable to be sent to frontend
    response_data['summary'] = ML.run_pipeline(response_data)

    # ENRICHMENT: Inject numeric ranges for the frontend gauges
    # (Since the raw JSON doesn't contain min/max values)
    for test in response_data['test_results']:
        enrich_with_ranges(test)

    return jsonify(response_data)

def enrich_with_ranges(test):
    """Adds visualization metadata (min, max, normalRange) based on test name."""
    try:
        test['value'] = float(test['value'])
    except ValueError:
        test['value'] = 0

    name = test['test_name'].lower()
    
    # Default Fallback
    test['min'] = 0
    test['max'] = 100
    test['normalRange'] = [20, 80]
    test['slightlyAbnormalRange'] = [10, 90] # Fallback

    # Custom Ranges for common Hematology tests
    if "wbc" in name:
        test.update({"min": 0, "max": 20, "normalRange": [4, 11], "slightlyAbnormalRange": [3, 13]})
    elif "neutrophils %" in name:
        test.update({"min": 0, "max": 100, "normalRange": [40, 75], "slightlyAbnormalRange": [35, 80]})
    elif "lymphocytes %" in name:
        test.update({"min": 0, "max": 100, "normalRange": [20, 45], "slightlyAbnormalRange": [15, 50]})
    elif "monocytes %" in name:
        test.update({"min": 0, "max": 20, "normalRange": [2, 10], "slightlyAbnormalRange": [1, 12]})
    elif "eosinophils %" in name:
        test.update({"min": 0, "max": 15, "normalRange": [1, 6], "slightlyAbnormalRange": [0.5, 8]})
    elif "haemoglobin" in name:
        test.update({"min": 5, "max": 20, "normalRange": [12, 16], "slightlyAbnormalRange": [10, 18]})
    elif "hematocrit" in name:
        test.update({"min": 20, "max": 60, "normalRange": [36, 46], "slightlyAbnormalRange": [33, 50]})
    elif "rbc" in name:
        test.update({"min": 2, "max": 8, "normalRange": [4, 5.5], "slightlyAbnormalRange": [3.5, 6]})
    elif "platelet" in name:
        test.update({"min": 0, "max": 600, "normalRange": [150, 450], "slightlyAbnormalRange": [130, 500]})
    elif "mcv" in name:
        test.update({"min": 50, "max": 120, "normalRange": [80, 100], "slightlyAbnormalRange": [75, 105]})

if __name__ == '__main__':
    # Run the server on port 5000
    app.run(debug=True, port=5000)