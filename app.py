from flask import Flask, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import requests
import io
from torchvision.models import ResNet18_Weights
from flask_cors import CORS
import os
import json  # Ajout de l'import pour JSON


# Load the model with the updated weights parameter
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval()

app = Flask(__name__)

CORS(app)  # Permet les requêtes depuis n'importe quelle origine

# Image preprocessing
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Charger le fichier de classes localement lors de l'initialisation
with open('imagenet_class_index.json') as f:
    class_idx = json.load(f)
    
# Function to get category name
def get_category_name(idx):
    category_name = class_idx[str(idx)][1]
    category_mapping = {
        "jersey": "t-shirt",
        "gown": "robe",
        "overskirt": "longue robe",
        "suit": "costume",
        "fur_coat": "manteau en fourrure",
        "trench_coat": "manteau",
        "sombrero": "chapeau",
        "bulletproof_vest": "veste",
        "sweatshirt": "pull / chemise",
        "cowboy_boot": "bottes",
        "Loafer": "chaussures",
        "sock": "chaussettes",
        "knee_pad": "chaussettes",
        "cloak": "Echarpe",
        "backpack": "casquette",
        "cowboy_hat": "chapeau",
        "ski_mask": "chapeau",
    }
    return category_mapping.get(category_name, category_name)


# Define the home route to prevent 404 errors
@app.route('/')
def home():
    return jsonify({"message": "API is running. Use /predict for predictions."})

# Define the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # Read the image file from the request
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    input_tensor = process_image(img)

    # Perform prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)
        category = get_category_name(predicted_idx.item())
    
    return jsonify({"category": category})

if __name__ == '__main__':
    app.run()

