from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from torchvision import transforms
from train import SiameseNetwork
import numpy as np
import base64
import uvicorn  # For running FastAPI in Streamlit
import threading  # For running FastAPI alongside Streamlit
import streamlit as st
import os

# Initialize FastAPI app
fastapi_app = FastAPI()

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load("C:/Users/vijay/Desktop/Code/Signature_Verification v2/best_model.pth", weights_only=False, map_location=device))
model.eval()

# Define preprocessing for input images
def preprocess_image(image):
    image = image.convert('L')
    # Resize and transform into tensor
    transform = transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Function to create a combined side-by-side image
def create_combined_image(image1, image2, label1="Signature to Verify", label2="Original Signature"):
    # Resize both images to the same size (side by side comparison)
    width, height = image1.size
    combined = Image.new("RGB", (width * 2, height), "white")
    combined.paste(image1, (0, 0))
    combined.paste(image2, (width, 0))

    # Draw labels below each image
    draw = ImageDraw.Draw(combined)
    font = ImageFont.load_default()

    draw.text((10, height - 20), label1, fill="black", font=font)
    draw.text((width + 10, height - 20), label2, fill="black", font=font)

    return combined

@fastapi_app.post("/verify_signature")
async def verify_signature(
    signature_to_verify: UploadFile = File(...),
    original_signature: UploadFile = File(...)
):
    # Load and preprocess the images
    image1 = Image.open(BytesIO(await signature_to_verify.read()))
    image2 = Image.open(BytesIO(await original_signature.read()))

    tensor1 = preprocess_image(image1)
    tensor2 = preprocess_image(image2)

    # Move the tensors to the same device as the model (GPU or CPU)
    tensor1 = tensor1.to(device)
    tensor2 = tensor2.to(device)

    # Perform inference
    with torch.no_grad():
        output1, output2 = model(tensor1, tensor2)
        # Use Cosine similarity (dot product) instead of Euclidean distance
        cosine_similarity = F.cosine_similarity(output1, output2).item()

    # Determine prediction based on a threshold (based on your results with PR curve or ROC-AUC)
    threshold = 0.7  # Set threshold; update this based on the results from your testing
    prediction = "Original" if cosine_similarity > threshold else "Forged"

    # Create a combined image for visualization
    combined_image = create_combined_image(image1.resize((300, 300)), image2.resize((300, 300)))

    # Convert the combined image to base64 for JSON response
    buffered = BytesIO()
    combined_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Return JSON response with all details
    return JSONResponse({
        "prediction": prediction,
        "score": round(cosine_similarity, 4),
        "combined_image": img_str
    })

# ======== STREAMLIT FRONTEND =========
# Streamlit wrapper to launch FastAPI
def run_fastapi():
    port = int(os.getenv("PORT", 8000))  # Use the PORT environment variable
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)

# Run FastAPI in a separate thread
thread = threading.Thread(target=run_fastapi)
thread.daemon = True
thread.start()

# Streamlit UI
st.title("Signature Verification API")
st.write("This app exposes a FastAPI backend for verifying signatures.")

st.write("API Endpoint: `/verify_signature`")
st.code("""
POST /verify_signature
Form Data:
- signature_to_verify: UploadFile
- original_signature: UploadFile
Response:
- prediction: "Original" or "Forged"
- score: Cosine similarity score
- combined_image: Base64-encoded image
""")
st.success("FastAPI is running in the background!")
