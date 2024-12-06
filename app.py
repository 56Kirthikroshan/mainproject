from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import streamlit as st
import gdown
import os

# Model path
MODEL_PATH = "dnb.keras"

# Function to download the model if it doesn't exist
def download_model():
    if not os.path.exists(MODEL_PATH):
        # Google Drive link to the model file
        url = "https://drive.google.com/uc?export=download&id=1JafYOKCVMi8g9e2rti4Utp_aVjpnPDxJ"
        st.write("Downloading model file...")
        gdown.download(url, MODEL_PATH, quiet=False)

# Load the model with error handling
def load_trained_model():
    try:
        model = load_model(MODEL_PATH)
        st.write("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocess uploaded image for prediction
def preprocess_image(image_file):
    image = Image.open(image_file)
    image = image.convert("RGB")  # Ensure it's in RGB format
    image = image.resize((224, 224))  # Resize to model's expected input shape
    image = img_to_array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Initialize app UI
st.title("Hydronephrosis Detector")
st.write("Upload an image to check if it's Normal or Abnormal.")

# Download and load the model
download_model()
model = load_trained_model()

# Class labels
class_labels = {0: "Abnormal", 1: "Normal"}

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Processing the image...")

    # Preprocess image
    image = preprocess_image(uploaded_file)

    # Make predictions
    if model:
        prediction = model.predict(image)  # Returns probabilities
        st.write(f"Prediction probabilities: {prediction}")

        # Determine the class
        predicted_class = 1 if prediction[0][0] > 0.5 else 0  # Binary classification threshold
        result = class_labels[predicted_class]

        # Display result
        st.success(f"Prediction: {result}")
    else:
        st.error("Model could not be loaded. Please try again later.")
