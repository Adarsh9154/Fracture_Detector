import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Set up the Streamlit app
st.set_page_config(page_title="Image Classifier", layout="centered")
st.title("🖼️ Image Classifier using Deep Learning")
st.write("Upload an image and get a prediction from your trained model.")

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")  # Replace with your file name
    return model

model = load_model()

# Upload image from user
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Define class labels (modify based on your model)
class_names = ['Cat', 'Dog', 'Other']  # <-- Replace with actual class names

# Preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess and predict
    st.write("Classifying...")
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]

    # Display result
    st.success(f"✅ Prediction: **{predicted_class}**")
