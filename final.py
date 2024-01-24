import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Function to load the pre-trained food classification model
def load_model():
    # Replace 'path_to_your_model' with the actual path to your saved model
    model = tf.keras.models.load_model('path_to_your_model')
    return model

# Function to preprocess the input image before feeding it to the model
def preprocess_image(image):
    # Resize the image to match the model's expected sizing
    image = image.resize((224, 224))
    # Convert the PIL image to a numpy array
    img_array = np.array(image)
    # Normalize the image
    img_array = img_array / 255.0
    # Expand dimensions to create a batch-size of 1
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Load the pre-trained model
model = load_model()

# Streamlit App
st.title("Food Classification App")

# User input for image upload
uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make predictions
    predictions = model.predict(processed_image)

    # Display the top prediction
    class_names = ["Burger", "Pizza", "Sushi", "Salad", "Ice Cream", "Other"]
    predicted_class = class_names[np.argmax(predictions)]
    confidence = predictions[0][np.argmax(predictions)]

    st.subheader("Prediction:")
    st.write(f"This looks like a {predicted_class} with confidence: {confidence:.2%}")
