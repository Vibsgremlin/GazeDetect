import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model("gaze_model.h5")

# Define class labels (Modify based on your dataset categories)
class_labels = [
    "BottomCenter", "BottomLeft", "BottomRight",
    "MiddleLeft", "MiddleCenter", "MiddleRight",
    "TopLeft", "TopCenter", "TopRight"
]

# Load a test image (Change this path to an actual image from your dataset)
image_path = r"C:\Users\91965\datasets\TestSet\BottomCenter\557.jpg"  # Modify this!
if not os.path.exists(image_path):
    print(f"❌ Error: Image not found at {image_path}")
    exit()

# Read and preprocess image (Convert to grayscale)
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
image_resized = cv2.resize(image, (64, 64))  # Resize to model input size
image_normalized = image_resized / 255.0  # Normalize pixel values
image_input = np.expand_dims(image_normalized, axis=-1)  # Add channel dimension (64, 64, 1)
image_input = np.expand_dims(image_input, axis=0)  # Expand for batch (1, 64, 64, 1)

# Predict gaze direction
prediction = model.predict(image_input)
predicted_index = np.argmax(prediction)  # Get class index
predicted_label = class_labels[predicted_index]  # Convert to class name

# Display image with prediction
plt.figure(figsize=(6, 6))
plt.imshow(image, cmap="gray")  # Display grayscale image
plt.title(f"Predicted Gaze Direction: {predicted_label}", fontsize=14)
plt.axis("off")
plt.show()

print(f"✅ Prediction: {predicted_label}")
