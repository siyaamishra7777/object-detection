import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 AI Object Detection System")

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

option = st.radio(
    "Select Detection Mode",
    ["📷 Live Camera", "🖼️ Image Upload"],
    horizontal=True
)

# IMAGE MODE
if option == "🖼️ Image Upload":

    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")

        results = model.predict(
            np.array(image),
            conf=0.25,
            verbose=False
        )

        annotated = results[0].plot()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Detected Objects")
            st.image(
                annotated,
                channels="BGR",
                use_container_width=True
            )

# CAMERA MODE
else:

    st.subheader("📷 Live Camera Detection")

    camera = st.camera_input("Take a picture")

    if camera:
        image = Image.open(camera).convert("RGB")

        results = model.predict(
            np.array(image),
            conf=0.25,
            verbose=False
        )

        annotated = results[0].plot()

        st.image(
            annotated,
            channels="BGR",
            caption="Detected Objects",
            use_container_width=True
        )