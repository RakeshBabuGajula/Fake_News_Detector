import streamlit as st
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = tf.keras.models.load_model("fake_news_model.keras", compile=False)

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_length = 150

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text

st.title("📰 Fake News Detector")
st.write("Paste a news article below and click Predict.")

user_input = st.text_area("Enter News Text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some news text!")
    else:
        cleaned = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        pad = pad_sequences(seq, maxlen=max_length)
        pred = model.predict(pad)[0][0]
        if pred > 0.5:
            st.success("✅ Prediction: REAL News")
        else:
            st.error("❌ Prediction: FAKE News")