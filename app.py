import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load Model
model = tf.keras.models.load_model("CNN_model.keras")

st.title("Face Mask Detection")

file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if file is not None:

    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=250)

    # Preprocess Image
    img = img.resize((224, 224))   # Change if your model uses another size
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    pred = model.predict(img)

    if pred[0][0] > 0.5:
        st.error("Without Mask")
    else:
        st.success("With Mask")