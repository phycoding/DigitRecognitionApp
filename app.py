import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

# Load the trained model
model = 2
model = tf.keras.models.load_model("model\digit_recognition_model.keras")

# Sidebar elements for customization
st.sidebar.header("Canvas Options")
stroke_width = st.sidebar.slider("Stroke width: ", 25, 35, 25)
stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#FFFFFF")
bg_color = st.sidebar.color_picker("Background color hex: ", "#000000")
bg_image = st.sidebar.file_uploader("Background image (optional):", type=["png", "jpg"])
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform", "polygon")
)
realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Canvas for drawing
st.markdown("### Draw a digit on the canvas")
canvas_result = st_canvas(
    # fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=280,
    width=280,
    drawing_mode=drawing_mode,
    display_toolbar=True,
    key="full_app",
)

# Processing the canvas image and making predictions
if canvas_result.image_data is not None:
    # Convert the image to the format expected by the model
    image = canvas_result.image_data.astype('uint8')
    
    # Convert the image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a mask to capture the drawn area
    _, thresh = cv2.threshold(image_gray, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours to get the bounding box of the drawn area
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour which should be the drawn area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop the drawn area
        cropped_image = image_gray[y:y+h, x:x+w]
        
        # Resize the cropped image to 28x28 pixels
        image_resized = cv2.resize(cropped_image, (28, 28))
        
        # Normalize the image
        image_normalized = image_resized.astype('float32') / 255
        
        # Reshape the image to fit the model input
        image_reshaped = image_normalized.reshape(1, 28, 28, 1)
        
        # Display the processed image
        st.markdown("### Processed Image")
        st.image(image_resized, width=250)
        
        # Make prediction
        prediction = model.predict(image_reshaped)
        predicted_digit = np.argmax(prediction, axis=1)
        
        # Display the prediction
        st.markdown(f"### Predicted Digit: **{predicted_digit[0]}**")
    else:
        st.markdown("### No drawing detected. Please draw a digit on the canvas.")
