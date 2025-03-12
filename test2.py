import tensorflow as tf
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model("gaze_model.h5")

# Load test image
image_path = r"C:\Users\91965\datasets\TestSet\MiddleRight\136.jpg"  # Change this
image = cv2.imread(image_path)
image = cv2.resize(image, (64, 64))  # Resize to model input size
image = image / 255.0  # Normalize
image = np.expand_dims(image, axis=0)  # Expand dims for batch processing

# Predict
prediction = model.predict(image)
predicted_label = np.argmax(prediction)  # Get predicted class index

# Print result
print(f"Predicted Gaze Direction: {predicted_label}")
