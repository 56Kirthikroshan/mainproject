from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import streamlit as st
from PIL import Image
import gdown
import os

# Path to save model
MODEL_PATH = "dnb.keras"

# Check if the model file already exists locally, if not, download it
if not os.path.exists(MODEL_PATH):
    # Google Drive shareable link
    url = "https://drive.google.com/uc?export=download&id=1JafYOKCVMi8g9e2rti4Utp_aVjpnPDxJ"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load the model
model = load_model(MODEL_PATH)

# Class labels
class_labels = {0: "Abnormal", 1: "Normal"}

# Streamlit UI
st.title("Hydronephrosis Detector")
st.write("Upload an image to check if it's normal or abnormal.")

# Image upload widget
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Preprocess the image
    image = Image.open(uploaded_file)
    image = image.convert("RGB")  # Ensure the image is in RGB format
    image = image.resize((224, 224))  # Resize to 224x224 as expected by the model
    image = img_to_array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image)  # Returns a numpy array

    # Debug: Display prediction probabilities
    st.write(f"Prediction probabilities: {prediction}")

    # Determine class based on threshold
    predicted_class = 1 if prediction[0][0] > 0.5 else 0  # Threshold: 0.5
    result = class_labels[predicted_class]

    st.success(f"Result: {result}")
