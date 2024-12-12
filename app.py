from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import pickle

app = Flask(__name__)

# Load the model
def load_model(filename="\\DigitRecognizer\\digit_recognizer_model.pkl"):
    """Load the trained model from a pickle file."""
    with open(filename, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
    return model["W1"], model["b1"], model["W2"], model["b2"]

# Preprocess image for prediction
def preprocess_image(file):
    """
    Preprocess the image to feed into the model:
    - Convert to grayscale
    - Resize to 28x28
    - Normalize pixel values
    - Flatten into a (784, 1) array
    """
    img = Image.open(file).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img) / 255.0  # Normalize pixel values
    return img_array.reshape(784, 1)  # Flatten to a column vector (784, 1)

# Forward propagation for prediction
def forward_prop(W1, b1, W2, b2, X):
    Z1 = np.dot(W1, X) + b1
    A1 = np.maximum(Z1, 0)  # ReLU activation
    Z2 = np.dot(W2, A1) + b2
    A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)  # Softmax activation
    return A2

# Prediction logic
def predict(file):
    """
    Predict the digit from an image file.
    """
    W1, b1, W2, b2 = load_model()
    X = preprocess_image(file)
    A2 = forward_prop(W1, b1, W2, b2, X)
    predicted_digit = np.argmax(A2, axis=0)[0]
    return int(predicted_digit)

# Flask API endpoint
@app.route('/predict', methods=['POST'])
def predict_digit():
    """
    API endpoint for digit prediction.
    Accepts an image file in a POST request and returns the predicted digit.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    image_file = request.files['image']
    try:
        # Predict the digit using the uploaded image file
        predicted_digit = predict(image_file)

        # Return the result
        return jsonify({'predicted_digit': predicted_digit}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
