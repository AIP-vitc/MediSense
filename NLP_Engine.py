import re
import json
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

class NLPEngine:
    def __init__(self, model_path, tokenizer_path):
        # Load the ONNX model for offline inference
        self.session = ort.InferenceSession(model_path)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        # Headers used as anchors for report slicing
        self.headers = ["USER_INFO", "LAB_INFO", "TESTS_AND_VALUES", "REMARKS_AND_RESULTS", "DOCTOR_INFO"]

    def _clean(self, text):
        """Removes citation artifacts and extra whitespace from narrative."""
        if not text: return "N/A"
        # Removes citation markers like or [cite: 2]
        text = re.sub(r'\[(?:cite|source|source:):\s*\d+\]', '', text)
        text = text.replace("[", "").replace("]", "").strip()
        return " ".join(text.split())

    def _extract_labs(self, text):
        """Captures numeric values, IHC markers, and scientific units."""
        results = []
        # Support for blood values (15.2), biopsy markers (positive), and dimensions
        pattern = r"([\w\d\s%()\-]+):\s*([\d\.]+|positive|negative|patchy positivity|measuring [\d\.\sx]+cm)\s*([^\(\nHL]*)?\s*([HL])?"
        
        for line in text.split('\n'):
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                test_name = match.group(1).strip()
                # Skip medical theory lines (Shield Logic)
                if len(test_name.split()) > 10: continue 
                
                status_raw = match.group(4)
                status = "High" if status_raw == "H" else ("Low" if status_raw == "L" else "Normal")
                
                results.append({
                    "test_name": test_name,
                    "value": match.group(2).strip(),
                    "unit": self._clean(match.group(3)) if match.group(3) else "N/A",
                    "status": status
                })
        return results

    def process(self, raw_text):
        """Parses unstructured text into a strict JSON format."""
        sections = {}
        pattern = f"(\\[{'|'.join(self.headers)}\\])"
        parts = re.split(pattern, raw_text)
        for i in range(1, len(parts), 2):
            header = parts[i].strip("[]")
            sections[header] = parts[i+1].strip()

        user_seg = sections.get("USER_INFO", "")
        lab_seg = sections.get("LAB_INFO", "")
        doc_seg = sections.get("DOCTOR_INFO", "")

        return {
            "patient_metadata": {
                "name": re.search(r"Patient Name:\s*(.*)", user_seg).group(1).split('\n')[0].strip() if "Patient Name" in user_seg else "N/A",
                "age": re.search(r"Age:\s*([\w\s]+)", user_seg).group(1).split('\n')[0].strip() if "Age" in user_seg else "N/A",
                "gender": re.search(r"Gender:\s*(\w)", user_seg).group(1).strip() if "Gender" in user_seg else "N/A",
                "patient_id": re.search(r"ID:\s*(.*)", user_seg).group(1).strip() if "ID" in user_seg else "N/A",
                "patient_address": self._clean(re.search(r"Address:\s*(.*)", user_seg).group(1)) if "Address" in user_seg else "N/A"
            },
            "laboratory_info": {
                "lab_name": self._clean(re.search(r"(?:Clinic/Laboratory Name|Laboratory Name):\s*(.*)", lab_seg).group(1)) if "Laboratory Name" in lab_seg else "N/A",
                "lab_address": self._clean(re.search(r"Address:\s*(.*)", lab_seg).group(1)) if "Address" in lab_seg else "N/A",
                "phone": re.search(r"Tel:\s*([\+\d\s\-]+)", lab_seg).group(1).strip() if "Tel:" in lab_seg else "N/A",
                "website": re.search(r"Website:\s*([\w\.]+)", lab_seg).group(1).strip() if "Website" in lab_seg else "N/A"
            },
            "test_results": self._extract_labs(sections.get("TESTS_AND_VALUES", "")),
            "clinical_remarks": self._clean(sections.get("REMARKS_AND_RESULTS", "")),
            "authorized_personnel": {
                "primary_doctor": re.search(r"(?:Doctor's Name|Doctor's name):\s*(.*)", doc_seg).group(1).split('\n')[0].strip() if "Doctor's" in doc_seg else "N/A",
                "specialization": re.search(r"Specialization:\s*(.*)", doc_seg).group(1).strip() if "Specialization" in doc_seg else "N/A",
                "referred_by": re.search(r"Referred by:\s*(.*)", doc_seg).group(1).strip() if "Referred by" in doc_seg else "N/A"
            }
        }
    
def analyse(report):
    engine = NLPEngine("./offline_model/model.onnx", "./offline_model/tokenizer.json")
    return engine.process(report)

if __name__ == "__main__":
    # Point to the folder created by your exporter script

    report = """
        [USER_INFO]
        Patient Name: Mr. Saubhik Bhaumik
        Age: 28 YRS
        Gender: M
        ID: 1015

        [LAB_INFO]
        Clinic/Laboratory Name: Labsmart Software Sample Letterhead
        Address: N/A
        Contact: +91 12345 67890, yourlabname@gmail.com, https://www.yourlabname.in/
        Date of Report: 28/10/2024 06:04 PM

        [TESTS_AND_VALUES]
        HEMOGLOBIN: 14 g/dl (Reference: 13 - 17)

        [REMARKS_AND_RESULTS]
        Hemoglobin is the major protein of erythrocytes that transports oxygen from the lungs to peripheral tissues. It is measured by spectrophotometry on automated instruments after lysis of red cells and conversion of all hemoglobin to cyanmethemoglobin. The cyanmethemoglobin technique is the method of choice selected by the International Committee for Standardization in Hematology. The method measures all hemoglobin derivatives except sulfhemoglobin by hemolyzing the specimen and adding a reducing agent. As such, this method does not distinguish between intracellular versus extracellular hemoglobin (hemolysis). Hypertriglyceridemia and very high white blood cell counts can cause false elevations of Hb.

        [DOCTOR_INFO]
        Mr. Sachin Sharma, DMLT, Lab Incharge (Signature present)
        Dr. A. K. Asthana, MBBS, MD Pathologist (Signature present)
        Referred by: Dr. Sachin Patil (MBBS)
    """

    final_output = analyse(report)
    print(json.dumps(final_output, indent=4))