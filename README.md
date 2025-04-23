# Fake-Currency-Detector

This project is a system for detecting fake currency notes using image processing and machine learning techniques. It includes feature extraction, model training, and a user interface for testing currency notes.

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Setup Instructions](#setup-instructions)
4. [Usage](#usage)
5. [Code Overview](#code-overview)
6. [Model Training](#model-training)
7. [Testing Currency Notes](#testing-currency-notes)
8. [Streamlit App](#streamlit-app)
9. [Dataset](#dataset)
10. [Dependencies](#dependencies)
11. [License](#license)

---

## Overview

The Fake-Currency-Detector uses image processing techniques such as Local Binary Patterns (LBP) and Structural Similarity Index (SSIM) to extract features from currency note images. A machine learning model (Random Forest Classifier) is trained on these features to classify notes as real or fake.

---

## Features

- **Feature Extraction**: Extracts texture-based features using LBP.
- **Model Training**: Trains a Random Forest Classifier on extracted features.
- **Currency Note Testing**: Provides a GUI and Streamlit-based interface for testing currency notes.
- **Result Analysis**: Displays detailed results for each feature tested.

---

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Fake-Currency-Detector
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the dataset is placed in the `Dataset/` directory.

4. Train the model using the `main_test.ipynb` notebook.

---

## Usage

### Running the GUI Application
Run the `Testing.ipynb` notebook to launch the GUI for testing currency notes.

### Running the Streamlit App
Run the `app.py` file to launch the Streamlit-based web application:
```bash
streamlit run app.py
```

---

## Code Overview

### Feature Extraction
The feature extraction uses Local Binary Patterns (LBP) to analyze texture patterns in grayscale images.

```python
def extract_lbp_features(gray_image, radius=3, n_points=8 * 3):
    lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins))
    return hist
```

### Model Training
The model is trained using a Random Forest Classifier in the `main_test.ipynb` notebook.

```python
clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)
```

### Testing Currency Notes
The `Testing.ipynb` notebook provides a GUI for testing currency notes. It uses OpenCV for image processing and tkinter for the GUI.

```python
def testFeature_1_2_7():
    # Feature detection and analysis logic
    pass
```

### Streamlit App
The `app.py` file provides a web-based interface for testing currency notes.

```python
uploaded_file = st.file_uploader("Upload a currency note image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    features = extract_features(image)
    prediction = model.predict(features.reshape(1, -1))
```

---

## Model Training

1. Open the `main_test.ipynb` notebook.
2. Run all cells to extract features, train the model, and save it as `currency_model_lbp.pkl`.

---

## Testing Currency Notes

### GUI Application
1. Open the `Testing.ipynb` notebook.
2. Run the notebook to launch the GUI.
3. Select an image of a currency note and analyze the results.

### Streamlit App
1. Run the `app.py` file:
   ```bash
   streamlit run app.py
   ```
2. Upload a currency note image and view the results.

---

## Streamlit App

The Streamlit app provides a user-friendly interface for testing currency notes. It displays the uploaded image, prediction results, and confidence scores.

---

## Dataset

Place the dataset in the `Dataset/` directory. The dataset should include images of real and fake currency notes organized into separate folders.

---

## Dependencies

- Python 3.8+
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Scikit-image
- Tkinter

---

## License

This project is licensed under the MIT License.