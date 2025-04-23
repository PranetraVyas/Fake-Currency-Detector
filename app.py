import streamlit as st
import cv2
import numpy as np
import pickle
from skimage.feature import local_binary_pattern

# --- 1. Load the Trained Model ---
try:
    with open('currency_model_lbp.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    st.error("Error: The trained model file ('currency_model_lbp.pkl') was not found. Please train the model first.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# --- 2. Feature Extraction Function (must be the same as in training) ---
def extract_lbp_features(gray_image, radius=3, n_points=8 * 3):
    """Extracts Local Binary Pattern (LBP) features from a grayscale image."""
    lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins))
    return hist

def extract_features(image):
    """Loads an image, extracts LBP features, and returns the feature vector."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = extract_lbp_features(gray)
    return features

# --- 3. Streamlit UI ---
st.title("Fake Currency Detection")

uploaded_file = st.file_uploader("Upload a currency note image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if image is not None:
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Extract features
        features = extract_features(image)

        if features is not None:
            # Make prediction (ensure features are in the correct shape)
            prediction_proba = model.predict_proba(features.reshape(1, -1))[0]
            prediction_class = np.argmax(prediction_proba)
            confidence = prediction_proba[prediction_class] * 100

            st.subheader("Prediction:")
            if prediction_class == 1:
                st.success(f"This is likely a REAL currency note (Confidence: {confidence:.2f}%).")
            else:
                st.error(f"This is likely a FAKE currency note (Confidence: {confidence:.2f}%).")

            st.subheader("Prediction Probabilities:")
            st.write(f"Probability of Fake: {prediction_proba[0] * 100:.2f}%")
            st.write(f"Probability of Real: {prediction_proba[1] * 100:.2f}%")

        else:
            st.warning("Could not extract features from the uploaded image.")
    else:
        st.error("Error: Could not decode the uploaded image.")