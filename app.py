from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# 🔥 Load model (same folder)
model = tf.keras.models.load_model("covid_model_v2.keras")

# Class labels (same order as training)
class_names = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

# Image preprocessing
def preprocess_image(image):
    image = image.resize((160, 160))  # must match training
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Test route
@app.route('/')
def home():
    return "✅ Model API is running"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(file).convert('RGB')

    img = preprocess_image(image)

    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction) * 100)

    return jsonify({
        "disease": predicted_class,
        "confidence": round(confidence, 2)
    })

# Render requires this
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)