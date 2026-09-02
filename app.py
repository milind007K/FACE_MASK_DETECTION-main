import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered"
)

# ==========================================
# TITLE
# ==========================================

st.title("😷 Face Mask Detection")
st.write("Upload a face image and the AI model will predict whether the person is wearing a mask.")

st.divider()

# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "model/face_mask_detector.keras"
IMG_SIZE = 160

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()

# ==========================================
# CHECK MODEL
# ==========================================

if model is None:

    st.error(
        "Model not found. Please make sure "
        "model/face_mask_detector.keras exists."
    )

    st.stop()

# ==========================================
# FILE UPLOADER
# ==========================================

uploaded_file = st.file_uploader(
    "Upload a Face Image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================
# PREDICTION
# ==========================================

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.divider()

    # ======================================
    # PREPROCESS IMAGE
    # ======================================

    resized_image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    image_array = np.array(
        resized_image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # ======================================
    # MODEL PREDICTION
    # ======================================

    prediction = model.predict(
        image_array,
        verbose=0
    )[0][0]

    prediction = float(prediction)

    # ======================================
    # DISPLAY RESULT
    # ======================================

    st.subheader("### Prediction")

    if prediction < 0.5:

        confidence = (1 - prediction) * 100

        st.success("😷 WITH MASK")

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    else:

        confidence = prediction * 100

        st.error("🙂 WITHOUT MASK")

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    # ======================================
    # DEBUG INFORMATION
    # ======================================

    with st.expander("Prediction Details"):

        st.write(
            f"Raw prediction: {prediction:.6f}"
        )

        st.write(
            f"Image size used by model: "
            f"{IMG_SIZE} × {IMG_SIZE}"
        )