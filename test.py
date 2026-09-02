import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "model/face_mask_detector.keras"
IMG_SIZE = 160

# ==========================================
# CHECK MODEL
# ==========================================

if not os.path.exists(MODEL_PATH):
    print("ERROR: Model file not found!")
    print("Expected:", MODEL_PATH)
    exit()

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# ==========================================
# GET IMAGE PATH
# ==========================================

image_path = input("\nEnter image path: ").strip()

# Remove quotes if pasted with quotes
image_path = image_path.strip('"').strip("'")

# ==========================================
# CHECK IMAGE
# ==========================================

if not os.path.exists(image_path):
    print("\nERROR: Image not found!")
    print("Check the image path.")
    exit()

# ==========================================
# LOAD IMAGE
# ==========================================

img = Image.open(image_path).convert("RGB")

print("Image loaded successfully.")

# ==========================================
# RESIZE
# ==========================================

img = img.resize((IMG_SIZE, IMG_SIZE))

# ==========================================
# CONVERT TO NUMPY
# ==========================================

img_array = np.array(img, dtype=np.float32)

# ==========================================
# ADD BATCH DIMENSION
# ==========================================

img_array = np.expand_dims(img_array, axis=0)

# ==========================================
# PREDICTION
# ==========================================

prediction = model.predict(
    img_array,
    verbose=0
)[0][0]

prediction = float(prediction)

# ==========================================
# RESULT
# ==========================================

print("\n======================================")
print("Raw prediction:", prediction)

if prediction < 0.5:

    confidence = (1 - prediction) * 100

    print("Prediction: WITH MASK")
    print(f"Confidence: {confidence:.2f}%")

else:

    confidence = prediction * 100

    print("Prediction: WITHOUT MASK")
    print(f"Confidence: {confidence:.2f}%")

print("======================================")