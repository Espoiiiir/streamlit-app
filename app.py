import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import requests

import streamlit as st
from PIL import Image

st.title("Image Component Analysis")

# Image upload widget
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("Image uploaded successfully!")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Please upload an image file.")

# Load a pre-trained model with the updated weights parameter
try:
    model = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
except requests.exceptions.RequestException as e:
    st.error("Failed to load the model weights. Check your internet connection.")
    raise e

model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

def analyze_image(image):
    # Transform the image and make predictions
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)

    # Extract the labels of detected objects
    labels = outputs[0]['labels']
    return labels

# Map label indices to names (COCO dataset labels)
COCO_INSTANCE_CATEGORY_NAMES = [
    '_background_', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
    'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A',
    'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

if st.button("Analyse Image"):
    if uploaded_file is not None:
        st.write("Analyzing image...")
        labels = analyze_image(image)
        label_names = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in labels]
        st.write("Detected components:")
        for name in label_names:
            st.write(name)
    else:
        st.write("Please upload an image first.")
