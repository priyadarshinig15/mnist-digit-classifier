import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# Load model
model = load_model("Digit_recogniser_Model.h5")

st.title("Handwritten Digit Recognizer")

st.write("Draw a digit below:")

# Create canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Prediction button
if st.button("Predict"):

    if canvas_result.image_data is not None:

        img = canvas_result.image_data[:, :, 0]

        img = Image.fromarray((img).astype('uint8'))

        img = img.resize((28, 28))

        img = np.array(img)

        img = img / 255.0

        img = img.reshape(1, 28, 28, 1)

        prediction = model.predict(img)

        predicted_digit = np.argmax(prediction)

        st.success(f"Predicted Digit: {predicted_digit}")