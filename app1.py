import streamlit as st
import numpy as np
from PIL import Image



st.title("🧠 Digit Recognition App")


@st.cache_resource
def get_model():
    """Load the trained model once per app session."""
    from tensorflow.keras.models import load_model

    return load_model("digit_model.h5")


model = None
model_error = None
try:
    model = get_model()
except Exception as exc:
    model_error = exc
    st.error("TensorFlow could not be loaded in this environment.")
    st.info(
        "Install/repair Microsoft Visual C++ Redistributable (x64) and "
        "use a TensorFlow-compatible Python environment, then restart the app."
    )
    st.code(str(exc))

uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None and model is not None:
    # Show image
    image = Image.open(uploaded_file).convert('L')  # grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize to 28x28
    image = image.resize((28, 28))

    # Convert to array
    img_array = np.array(image)

    # Normalize
    img_array = img_array / 255.0

    # Reshape for model
    img_array = img_array.reshape(1, 28, 28)

    # Predict
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)

    st.success(f"Predicted Digit: {digit}")

elif uploaded_file is not None and model is None and model_error is not None:
    st.warning("Prediction is unavailable until TensorFlow loads successfully.")