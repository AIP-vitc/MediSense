# Heading 1**MediSense: Smart Disease Prediction and Drug Recommendation System**

MediSense is an integrated healthcare solution that leverages Machine Learning and Deep Learning to predict diseases based on symptoms and provide data-driven drug recommendations. By utilizing Natural Language Processing (NLP), the system also analyzes patient reviews to ensure recommendations are based on real-world efficacy and user satisfaction.

## Heading 2**Project Overview**

The primary goal of MediSense is to assist healthcare professionals and patients by transforming symptom data into actionable medical insights. The system employs high-accuracy models like RoBERTa for disease classification and uses weighted average methodologies for drug suggestions.

## Heading 2**Key Features**

Disease Prediction: Utilizes multiple Machine Learning and Deep Learning prototypes to predict potential health conditions from user-input symptoms.

Drug Recommendation: Suggests medications based on disease conditions, user ratings, and usefulness metrics.

Sentiment Analysis: Employs Natural Language Processing tools to analyze patient reviews for better recommendation accuracy.

User Interface: A web-based platform for easy interaction and result visualization.

Data-Driven Insights: Processes extensive medical datasets containing thousands of unique conditions and drug names.

## Heading 2**Technical Stack**

**Languages**: Python

**Machine Learning**: Scikit-learn, XGBoost, Random Forest

**Deep Learning**: PyTorch, TensorFlow, RoBERTa

**NLP**: NLTK, VADER, Sentiment Analysis

**Backend**: Flask / FastAPI

**Frontend**: HTML5, CSS3, JavaScript, Tailwind CSS

## Heading 2**System Architecture**

The workflow consists of three main stages:

Data Preprocessing: Cleaning medical datasets, handling missing values, and tokenizing text data.

Model Training: Implementing and comparing various algorithms to achieve high diagnostic accuracy.

Recommendation Engine: Filtering drugs based on condition-specific ratings and peer reviews to provide reliable suggestions.

## Heading 2**Installation and Setup**

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


Run the application

`python app.py`

## Heading 2**Authors**

1.Kumar Shaurya
2.Rajarshi Saha
3.Vedant Jadhav
4.Tirth Mangukiya
