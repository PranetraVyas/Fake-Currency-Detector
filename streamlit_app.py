import streamlit as st
import cv2
import numpy as np
import pickle
from skimage.feature import local_binary_pattern
from PIL import Image  # For working with images

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
st.title("Fake Currency Detection Analysis")

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
            # Make prediction
            prediction_proba = model.predict_proba(features.reshape(1, -1))[0]
            prediction_class = np.argmax(prediction_proba)
            confidence = prediction_proba[prediction_class] * 100

            st.subheader("Prediction Result:")
            if prediction_class == 1:
                st.success(f"This note is likely REAL (Confidence: {confidence:.2f}%).")
            else:
                st.error(f"This note is likely FAKE (Confidence: {confidence:.2f}%).")

            st.subheader("Detailed Analysis:")
            st.write(f"**Overall Confidence:** {confidence:.2f}%")
            st.write(f"**Probability of Real:** {prediction_proba[1] * 100:.2f}%")
            st.write(f"**Probability of Fake:** {prediction_proba[0] * 100:.2f}%")

            st.info("Based on the features extracted (primarily texture analysis using Local Binary Patterns), the model has determined the likelihood of the note being real or fake.")
            st.warning("This analysis is based on the features the model was trained on. A comprehensive real-world analysis would involve examining multiple security features that are not currently implemented in this basic model.")

            st.subheader("Considerations for Real vs. Fake:")
            st.markdown(
                """
                **To determine if a note is genuinely real or fake, one would typically examine several key security features:**

                * **Watermark:** Check for a clear and detailed watermark when held against the light.
                * **Security Thread:** Look for an embedded thread that appears continuous when held to light and may have text or fluorescence.
                * **Intaglio (Raised Print):** Feel for raised printing on areas like the portrait and seals.
                * **See-Through Register:** Observe if designs on the front and back align perfectly against the light.
                * **Latent Image:** Tilt the note to see if a hidden image becomes visible.
                * **Micro-lettering:** Use magnification to check for small, clear text in specific areas.
                * **Fluorescence (UV Light):** Examine the note under UV light for specific glowing patterns.
                * **Optically Variable Ink:** Check if certain numerals change color when tilted.
                * **Paper Quality:** Assess the feel and crispness of the paper.
                * **Serial Numbers:** Verify the uniqueness, font, and alignment of the serial numbers.
                * **Formatting and Alignment:** Ensure all elements are printed precisely with consistent spacing.

                **This AI model currently focuses on texture patterns captured by Local Binary Patterns. It does not directly analyze these other crucial security features.** For a more accurate assessment, a system would need to incorporate algorithms to specifically detect and analyze each of these individual features.
                """
            )

        else:
            st.warning("Could not extract features from the uploaded image.")
    else:
        st.error("Error: Could not decode the uploaded image.")