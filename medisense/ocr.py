from google import genai
import PIL.Image
import os

# 1. Configuration
API_KEY = "AIzaSyC3wEf_xHIE1mzyBRDDKIiR_HUVBz1Oyw8"
client = genai.Client(api_key=API_KEY)

# 2. Strict Prompt Construction
# This tells the model exactly how to behave every single time.
STRUCTURED_PROMPT = """
Perform OCR and extract information into the following exact blocks and order. 
Use the labels provided below. If a section is not found in the image, write "N/A" for that block.
DO NOT include any conversational text or markdown formatting (like bolding **).

[USER_INFO]
(Patient Name, Age, Gender, ID)

[LAB_INFO]
(Clinic/Laboratory Name, Address, Contact, Date of Report)

[TESTS_AND_VALUES]
(List all tests, results, units, and reference ranges)

[REMARKS_AND_RESULTS]
(Clinical interpretations, summary, or final diagnosis)

[DOCTOR_INFO]
(Doctor's name, specialization, and signature details)
"""

def perform_structured_ocr(image_path, output_filename="ocr_output.txt"):
    try:
        img = PIL.Image.open(image_path)
        print(f"Processing {image_path} into structured blocks...")

        # Use Gemini 3 Flash for the most reliable vision extraction in 2025
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=[STRUCTURED_PROMPT, img]
        )
        
        extracted_data = response.text.strip()
        return extracted_data

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    IMAGE_FILE = "img2.png" 
    
    if os.path.exists(IMAGE_FILE):
        result = perform_structured_ocr(IMAGE_FILE)
        if result:
            print("\n--- FORMATTED OUTPUT ---\n")
            print(result)
    else:
        print(f"File not found: {IMAGE_FILE}")