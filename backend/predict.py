import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import json

# Load trained model
model = load_model("model/trained_model.h5")  # Ensure correct path to the model

# Load class indices from the saved JSON file
with open("model/class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse the class_indices for easy lookup (index -> class_name)
class_names = {v: k for k, v in class_indices.items()}

# Italian to English class name translation (if needed)
translate = {
    "cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly",
    "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep",
    "scoiattolo": "squirrel", "ragno": "spider"
}

def predict_animal(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(64, 64))  # Match input size from training
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match input shape

    # Get model predictions
    prediction = model.predict(img_array)[0]  # Get the prediction for the single image
    sorted_indices = prediction.argsort()[::-1]  # Sort the prediction by confidence (highest first)

    # Create result list with translated English labels and confidence scores
    results = []
    for i in sorted_indices:
        class_id = class_names[i]  # Get class name by index
        english_name = translate.get(class_id, class_id)  # Translate to English if available
        confidence = float(prediction[i]) * 100  # Convert prediction to percentage
        results.append((english_name, confidence))

    return results
