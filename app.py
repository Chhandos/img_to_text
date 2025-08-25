import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import gdown
import os
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image

st.set_page_config(page_title="Image Captioning", layout="centered")

# -------------------
# Google Drive Model Download
# -------------------
MODEL_URL = "https://drive.google.com/uc?id=1G7OOgMFkGf1-No3h-w8CUeh3FyD11Ay1"
MODEL_PATH = "CaptionModel.keras"

if not os.path.exists(MODEL_PATH):
    st.write("⬇️ Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.success("✅ Model downloaded successfully!")

# -------------------
# Helper functions
# -------------------
def extract_features(image_bytes, cnn_model):
    # open image from BytesIO
    image = Image.open(image_bytes).convert("RGB")
    image = image.resize((299, 299))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = cnn_model.predict(image, verbose=0)
    return feature

def generate_caption(model, photo, word_index, index_word, max_length=37):
    in_text = "startseq"
    for _ in range(max_length):
        sequence = [word_index[w] for w in in_text.split() if w in word_index]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = index_word.get(yhat)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    final_caption = in_text.split()[1:-1]
    return " ".join(final_caption)

# -------------------
# Streamlit UI
# -------------------
st.title("Generate Caption")
st.write("Upload an image and click **Generate Caption** to see results!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Caption"):
        st.write(" ⏳ Loading model... please wait ")

        # Load model & tokenizer
        model = tf.keras.models.load_model(MODEL_PATH)
        with open("word_index.pkl", "rb") as f:
            word_index = pickle.load(f)
        index_word = {v: k for k, v in word_index.items()}

        # Load CNN encoder
        cnn_model = Xception(include_top=False, pooling="avg")

        # Extract features
        features = extract_features(uploaded_file, cnn_model)

        # Generate caption
        caption = generate_caption(model, features, word_index, index_word, max_length=37)

        st.subheader("Generated Caption:")
        st.success(caption)
