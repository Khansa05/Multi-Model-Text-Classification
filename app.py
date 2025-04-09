import streamlit as st
import joblib
import numpy as np
import re
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Load models and tools
nb_model = joblib.load('naive_bayes_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')
lstm_model = load_model('lstm_model.h5')
vectorizer = joblib.load('vectorizer.pkl')
tokenizer = joblib.load('tokenizer.pkl')

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text.strip()

# Prediction function
def predict_news(text):
    clean_text = preprocess_text(text)

    # Vectorizer-based models
    vec_input = vectorizer.transform([clean_text]).toarray()
    nb_pred = nb_model.predict(vec_input)[0]
    rf_pred = rf_model.predict(vec_input)[0]

    # LSTM prediction
    seq_input = tokenizer.texts_to_sequences([clean_text])
    padded_input = pad_sequences(seq_input, maxlen=100)
    lstm_pred = lstm_model.predict(padded_input)[0][0]

    return {
        "Naïve Bayes": "Real" if nb_pred else "Fake",
        "Random Forest": "Real" if rf_pred else "Fake",
        "LSTM": "Real" if lstm_pred > 0.5 else "Fake"
    }

# Streamlit UI
st.title("📰 Fake News Detection App")
st.write("Check whether a news article is **Fake** or **Real** using ML and Deep Learning models.")

user_input = st.text_area("Paste a news article below:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        predictions = predict_news(user_input)
        st.success("Prediction Results:")
        st.write(f"**Naïve Bayes:** {predictions['Naïve Bayes']}")
        st.write(f"**Random Forest:** {predictions['Random Forest']}")
        st.write(f"**LSTM:** {predictions['LSTM']}")
