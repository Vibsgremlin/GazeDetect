import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Define dataset paths
BASE_PATH = r"C:\Users\91965\datasets"
TRAIN_PATH = os.path.join(BASE_PATH, "TrainingSet")
TEST_PATH = os.path.join(BASE_PATH, "TestSet")
IMPROVEMENT_PATH = os.path.join(BASE_PATH, "ImprovementSet", "Improv Set")

# Define image categories (from ImprovementSet)
CATEGORIES = ["BottomCenter", "BottomLeft", "BottomRight", "MiddleLeft", "MiddleRight", "TopCenter", "TopLeft", "TopRight"]

# Allowed image formats
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg")

X, y = [], []

def load_images_from_folder(folder_path, label):
    """Loads images from a folder and assigns a given label."""
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(VALID_EXTENSIONS):
            continue  # Skip non-image files

        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

        if img is None:
            print(f"Skipping unreadable image: {filename}")
            continue

        img = cv2.resize(img, (64, 64))  # Resize to 64x64
        img = img / 255.0  # Normalize

        X.append(img)
        y.append(label)

# Load training images
print("Loading TrainingSet images...")
load_images_from_folder(TRAIN_PATH, label=1)

# Load test images
print("Loading TestSet images...")
load_images_from_folder(TEST_PATH, label=0)

# Load ImprovementSet images (loop through categories)
print("Loading ImprovementSet images...")
for category in CATEGORIES:
    category_path = os.path.join(IMPROVEMENT_PATH, category)
    if os.path.exists(category_path):
        load_images_from_folder(category_path, label=2)  # Label 2 for ImprovementSet
    else:
        print(f"Skipping missing category: {category}")

# Ensure data is valid
if len(X) == 0:
    print("No valid images found. Please check your dataset.")
    exit()

# Convert to NumPy arrays
X = np.array(X).reshape(-1, 64, 64, 1)
y = np.array(y)

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save processed data
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print(f" Processed {len(X)} images. Training: {len(X_train)}, Testing: {len(X_test)}")
