# **MediSense: Smart Disease Prediction and Drug Recommendation System**

MediSense is a health analysis platform designed to bridge the gap between complex medical terminology and patient understanding. By leveraging advanced Natural Language Processing (NLP) and Optical Character Recognition (OCR), the system transforms technical medical reports and diagnostic data into clear, straightforward summaries tailored for individuals without a healthcare background.

## **Project Overview**

The primary mission of MediSense is to empower patients by making their health data accessible. Medical reports often contain dense jargon that can be confusing or alarming for patients. This system utilizes AI models to extract key information from uploaded documents and rephrase it into simple language, ensuring that users can easily understand their health status and diagnostic results.

## **Key Features**

**Patient-Centric Summarization**: Automatically simplifies technical medical reports into clear, easy-to-understand summaries.

**Document Processing (OCR)**: Integrates Google's OCR to extract text from scanned medical images and PDF documents.

**AI-Driven Insights**: Utilizes ML Models to analyze medical data and provide personalized health explanations.

**User-Friendly Interface:** A clean, responsive web platform designed for easy document uploads and result visualization.

**Local Deployment**: Supports local deployment for privacy and agile testing.

## **Technical Stack**

**Languages**: Python

**OCR**: Google OCR, Qwen3-VL

**Machine Learning**: Scikit-learn, Random Forest

**Deep Learning**: PyTorch

**NLP**: NLTK, VADER

**Backend**: Flask / FastAPI

**Frontend**: HTML5, CSS3, JavaScript, Tailwind CSS

## **System Architecture**

**The workflow consists of three main stages:**

**Data Acquisition**: Users upload medical reports in image or PDF format.

**Information Extraction**: The system uses OCR to digitize the text and preprocesses it for the AI model.

**Simplification Engine**: The AI processes the technical text and generates a summary that focuses on clarity and accessibility for the patient.

## **Installation and Setup**

**Clone the repository**

```
git clone [https://github.com/AIP-vitc/MediSense.git](https://github.com/AIP-vitc/MediSense.git)
cd MediSense
```

**Create a virtual environment**

```
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

**Install dependencies**

`pip install -r requirements.txt`


**Run the application**

`python app.py`

## **Authors**

Kumar Shaurya,
Rajarshi Saha,
Vedant Jadhav,
Tirth Mangukiya
