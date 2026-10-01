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
    with open("class_names.json", "r") as f:
        return json.load(f)

model = load_model()
class_names = load_class_names()

st.title("🌿 Plant Disease Detection")
st.write("Select a plant and upload its leaf image.")

plants = [
    "Apple",
    "Blueberry",
    "Cherry",
    "Corn",
    "Grape",
    "Orange",
    "Peach",
    "Pepper",
    "Potato",
    "Raspberry",
    "Soybean",
    "Squash",
    "Strawberry",
    "Tomato"
]

selected_plant = st.selectbox(
    "Select Plant",
    plants
)

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
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

    predictions = model.predict(img_array, verbose=0)[0]

    plant_indices = [
        i for i, name in enumerate(class_names)
        if name.startswith(selected_plant + "___")
    ]

    plant_predictions = predictions[plant_indices]

    predicted_position = np.argmax(plant_predictions)
    predicted_index = plant_indices[predicted_position]

    confidence = predictions[predicted_index]
    predicted_class = class_names[predicted_index]

    disease = predicted_class.split("___")[1]

    st.success(f"Plant: {selected_plant}")
    st.success(f"Disease: {disease}")
    st.info(f"Confidence: {confidence * 100:.2f}%")

    st.subheader("Predictions")

    top_positions = np.argsort(plant_predictions)[-3:][::-1]

    for position in top_positions:
        index = plant_indices[position]
        probability = predictions[index]

        disease_name = class_names[index].split("___")[1]

        st.write(
            f"{disease_name}: "
            f"{probability * 100:.2f}%"
        )

        st.progress(float(probability))
