import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("YOLOv8 Image Classification (ImageNet)")
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # YOLOv8 nano classification pretrained on ImageNet

model = load_model()

source = st.radio("Choose input source:", ["Upload Image", "Use Camera"])

if source == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        img_array = np.array(image)
        results = model(img_array)

        top1_class_id = results[0].probs.top1
        class_name = results[0].names[top1_class_id]
        class_prob = results[0].probs.top1conf.item() * 100
        st.write(f"Prediction: **{class_name}** ({class_prob:.2f}% confidence)")
if source == "Use Camera":
    picture = st.camera_input("Take a picture")
    if picture is not None:
        img = Image.open(picture)
        st.image(img, caption="Captured Image", use_column_width=True)
        img_array = np.array(img)
        results = model(img_array)
        top1_class_id = results[0].probs.top1
        class_name = results[0].names[top1_class_id]
        class_prob = results[0].probs.top1conf.item() * 100
        st.write(f"Prediction: **{class_name}** ({class_prob:.2f}% confidence)")
