import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import matplotlib.pyplot as plt
import tempfile

st.title("Fake ₹500 Currency Note Detector - ORB & SSIM")

st.markdown("Upload a ₹500 note image and compare it with a genuine reference using feature matching and SSIM.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Load reference image path (ensure this exists)
reference_path = "Dataset/500_dataset/500_s1.jpg"
ref_img = cv2.imread(reference_path)
ref_img = cv2.resize(ref_img, (1167, 519))

# Function to compute ORB feature matches
def compute_orb(template_img, query_img):
    orb = cv2.ORB_create(nfeatures=700, scaleFactor=1.2, nlevels=8, edgeThreshold=15)
    kpts1, descs1 = orb.detectAndCompute(template_img, None)
    kpts2, descs2 = orb.detectAndCompute(query_img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descs1, descs2)
    matches = sorted(matches, key=lambda x: x.distance)
    matched_img = cv2.drawMatches(template_img, kpts1, query_img, kpts2, matches[:30], None, flags=2)
    return matched_img, len(matches)

if uploaded_file is not None:
    user_img = Image.open(uploaded_file).convert("RGB")
    user_img = np.array(user_img)
    user_img = cv2.resize(user_img, (1167, 519))

    # SSIM calculation
    gray_user = cv2.cvtColor(user_img, cv2.COLOR_RGB2GRAY)
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    ssim_score, _ = ssim(gray_user, gray_ref, full=True)

    # ORB Matching
    matched_img, num_matches = compute_orb(ref_img, cv2.cvtColor(user_img, cv2.COLOR_RGB2BGR))

    st.image(user_img, caption="Uploaded Note", width=350)
    st.image(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB), caption="Reference Note", width=350)
    st.image(matched_img, caption=f"ORB Feature Matches (Top 30 shown)", use_column_width=True)

    st.markdown(f"**SSIM Score:** {ssim_score:.2f}")
    st.markdown(f"**Number of ORB Matches:** {num_matches}")

    # Simple threshold-based decision
    if ssim_score > 0.85 and num_matches > 100:
        st.success("This note appears to be *Genuine*.")
    else:
        st.error("This note appears to be *Fake* or tampered.")
else:
    st.info("Upload a ₹500 note image to begin analysis.")
