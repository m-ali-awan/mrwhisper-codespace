import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import io
import base64
import requests

# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
)

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

realtime_update = st.sidebar.checkbox("Update in realtime", True)

    

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=150,
    drawing_mode=drawing_mode,
    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas",
)

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)
    # Convert the image data to bytes
    img = Image.fromarray((canvas_result.image_data * 255).astype('uint8'))
    img = img.convert("RGB")  # Convert image to RGB mode
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")

    # Upload the image to imgbb
    url = "https://api.imgbb.com/1/upload"
    payload = {
        "key": "dc68aff6262d4c421e917827f72fb5de",  # Replace with your API key
        "image": base64.b64encode(buffered.getvalue()).decode("utf-8"),
    }
    response = requests.post(url, payload)
    if response.status_code == 200:
        image_url = response.json()["data"]["url"]
        st.write(f"Image URL: {image_url}")
    else:
        st.write("Failed to upload the image.")