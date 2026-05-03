from flask import Flask, request, jsonify
from flask_cors import CORS   # 🔥 ADD THIS
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# 🔥 ENABLE CORS (IMPORTANT FIX)
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://127.0.0.1:5501",
            "http://localhost:5501",
            "https://your-vercel-app.vercel.app"  # update later
        ]
    }
})

# 🔥 Load model
model = tf.keras.models.load_model("covid_model_v2.keras")

# Class labels
class_names = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

# Preprocess
def preprocess_image(image):
    image = image.resize((160, 160))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Health check
@app.route('/')
def home():
    return jsonify({
        "status": "success",
        "message": "X-ray Model API is running"
    })

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        image = Image.open(file).convert('RGB')

        img = preprocess_image(image)

        prediction = model.predict(img)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction) * 100)

        return jsonify({
            "status": "success",
            "disease": predicted_class,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Run
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
