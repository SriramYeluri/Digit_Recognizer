import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
import cv2

st.title('Digit Recognizer')
canvas_result = st_canvas(
    fill_color="#000000",  # Fixed fill color with some opacity
    stroke_width=10,
    stroke_color='#ffffff',
    background_color="#000000",
    height=150, width=150,
    drawing_mode='freedraw',
    key="canvas",
)

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)
img = canvas_result.image_data
img = img.astype('uint8')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (28, 28))

model = keras.models.load_model('model.hdf5')

if st.button('Predict'):
    img = np.expand_dims(img, axis=0)
    pred_output = model.predict(img)
    out = np.argmax(pred_output[0])
    st.success(out)
    st.title('Bar Graph')
    st.bar_chart(pred_output[0])

