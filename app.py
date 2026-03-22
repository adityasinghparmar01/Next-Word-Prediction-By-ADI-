import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model("lstm_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load max length
with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

# Function to predict next word
def predict_next_word(text):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')

    predicted = model.predict(sequence, verbose=0)
    predicted_word_index = np.argmax(predicted)

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return ""

# UI
st.set_page_config(page_title="Next Word Predictor", page_icon="😎")

st.title("ADI😎 Next Word Prediction using LSTM")
st.write("Enter a sentence for getting the next word from me #ADI")

user_input = st.text_input("Enter your text:")

if st.button("Predict Next Word"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        next_word = predict_next_word(user_input)
        st.success(f"👉 Predicted Next Word: **{next_word}**")

# Generate multiple words
st.subheader("🔁 Generate Sentence")

num_words = st.slider("Number of words to generate:", 1, 10, 3)

if st.button("Generate Sequence"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        text = user_input
        for _ in range(num_words):
            next_word = predict_next_word(text)
            text += " " + next_word
        st.success(f"✨ Generated Text: {text}")