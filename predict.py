import tensorflow as tf
from PIL import Image
import numpy as np
import sys
import os

# Check if image path is provided
if len(sys.argv) != 2:
    print("Usage: python predict.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

# Check if image exists
if not os.path.exists(image_path):
    print(f"Error: Image file '{image_path}' not found.")
    sys.exit(1)

# Load the trained model
try:
    model = tf.keras.models.load_model('model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Define image size (must match training)
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Load and preprocess the image
try:
    img = Image.open(image_path)
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Resize to target size
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    print("Image processed successfully.")
except Exception as e:
    print(f"Error processing image: {e}")
    sys.exit(1)

# Make prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]

# Class labels (must match training)
class_labels = ['fresh', 'rotten']
predicted_label = class_labels[predicted_class]

# Get confidence
confidence = predictions[0][predicted_class] * 100

# Output result
print(f"Prediction: The fruit/vegetable is {predicted_label.upper()} with {confidence:.2f}% confidence.")