import streamlit as st
import pickle

# Load model
with open("spam_detection_model.pkl", "rb") as file:
    model = pickle.load(file)

# Page configuration
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📱",
    layout="centered"
)

# Title
st.title("📱 SMS Spam Detector")
st.write("Enter an SMS message and let the machine learning model determine whether it is spam.")

# Input
message = st.text_area(
    "Enter your message:",
    placeholder="Example: Congratulations! You have won a free prize..."
)

# Prediction
if st.button("Check Message", type="primary"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:
        prediction = model.predict([message])[0]

        if prediction == 1:
            st.error("🚨 SPAM MESSAGE")
            st.write("This message has been classified as spam.")

        else:
            st.success("✅ HAM — NOT SPAM")
            st.write("This message appears to be legitimate.")
    
