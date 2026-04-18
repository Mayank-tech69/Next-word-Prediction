import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="✨",
    layout="centered"
)

# ------------------ LOAD FILES ------------------
@st.cache_resource
def load_assets():
    model = load_model("lstm_model.h5")
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    maxlen = pickle.load(open("max_len.pkl", "rb"))
    return model, tokenizer, maxlen

model, tokenizer, maxlen = load_assets()

# ------------------ STYLING ------------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1f1c2c, #928dab);
        color: white;
    }
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
        color: #dcdcdc;
    }
    .stTextInput>div>div>input {
        border-radius: 12px;
        padding: 10px;
    }
    .stButton>button {
        background-color: #ff4b2b;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown('<div class="title">✨ Next Word Prediction AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Type a sentence and let AI complete it</div>', unsafe_allow_html=True)

# ------------------ INPUT ------------------
text_input = st.text_input("Enter your text:")
num_words = st.slider("Number of words to generate", 1, 10, 3)

# ------------------ PREDICTION FUNCTION ------------------
def predict_next_words(text, n):
    for _ in range(n):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = token_list[-maxlen:]
        token_list = np.pad(token_list, (maxlen - len(token_list), 0), mode='constant')

        predicted = model.predict(np.array([token_list]), verbose=0)
        predicted_word_index = np.argmax(predicted)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                output_word = word
                break

        text += " " + output_word
    return text

# ------------------ BUTTON ------------------
if st.button("🚀 Generate"):
    if text_input.strip() == "":
        st.warning("Please enter some text")
    else:
        result = predict_next_words(text_input, num_words)
        st.success(result)

# ------------------ FOOTER ------------------
st.markdown("""
---
Made with ❤️ using Streamlit
""")
