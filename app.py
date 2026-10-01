import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

@st.cache_data

def load_class_names():
    with open("data/class_names.json", "r") as f:
        return json.load(f)
    

model = load_model()
class_names = load_class_names()

st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image to detect its disease.")

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )

    image = image.resize((224, 224))

    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index]

    predicted_class = class_names[predicted_index]

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence * 100:.2f}%")

    st.subheader("Top 3 Predictions")

    top_indices = np.argsort(predictions[0])[-3:][::-1]

    for index in top_indices:
        probability = predictions[0][index]

        st.write(
            f"{class_names[index]}: "
            f"{probability * 100:.2f}%"
        )

        st.progress(float(probability))
