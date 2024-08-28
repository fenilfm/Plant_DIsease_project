from flask import Flask, request, jsonify
import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__, static_folder='static')

# Load model and class indices
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, 'trained_model', 'plant_disease_model.h5')
class_indices_path = os.path.join(working_dir, 'class_indices.json')

model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(class_indices_path))

def load_and_preprocess_image(image, target_size=(224, 224)):
    try:
        img = Image.open(image)
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0
        return img_array
    except Exception as e:
        raise ValueError(f"Image preprocessing error: {str(e)}")

def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown")
    return predicted_class_name

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        prediction = predict_image_class(model, image, class_indices)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
