# 🧠 Multi-Model Text Classification App
This is a Streamlit-based application that uses three machine learning models — Naive Bayes, Random Forest, and LSTM — to classify input text. The app is deployed on Hugging Face Spaces and can be used to test and compare model predictions.

## 🚀 Try it now
👉 [Click here to use the app on Hugging Face](https://khansaaqureshi-real-fake-news-detection.hf.space)

## 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- TensorFlow / Keras
- Joblib
- Hugging Face Spaces

## 📁 Files in this Repository

- `app.py`: Main Streamlit app
- `requirements.txt`: Python dependencies
- `naive_bayes_model.pkl`: Pre-trained Naive Bayes model
- `random_forest_model.pkl`: Pre-trained Random Forest model
- `lstm_model.h5`: Pre-trained LSTM model (Keras)
- `vectorizer.pkl`: Text vectorizer (CountVectorizer or TF-IDF)
- `tokenizer.pkl`: Tokenizer for LSTM model
- `README.md`: Project description
- `Real_Fake_News_Detection.docx`: Project Description document

## 🧪 How It Works

1. Input your text
2. The app vectorizes or tokenizes your input
3. Three models give their classification output
4. Compare results on screen

---

## 📦 Installation

To run locally:

```bash
git clone https://github.comKhansa05/Multi-Model-Text-Classification.git
cd Multi-Model-Text-Classification
pip install -r requirements.txt
streamlit run app.py

