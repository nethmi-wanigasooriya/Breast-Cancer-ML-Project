import streamlit as st
import joblib
import numpy as np

model = joblib.load('breast_cancer_model.pkl')

st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🩺", layout="wide")

st.title("🩺 Breast Cancer Diagnostic Predictor")
st.write("Enter the key cell features to predict whether the tumor is **Malignant** or **Benign**.")

st.sidebar.header("Feature Inputs")

# Select/Input main features (or key features used in your model)
# Scikit-Learn Breast Cancer Dataset typically uses 30 features or key features like below:
mean_radius = st.sidebar.number_input("Mean Radius", min_value=0.0, max_value=40.0, value=14.0)
mean_texture = st.sidebar.number_input("Mean Texture", min_value=0.0, max_value=40.0, value=19.0)
mean_perimeter = st.sidebar.number_input("Mean Perimeter", min_value=0.0, max_value=250.0, value=90.0)
mean_area = st.sidebar.number_input("Mean Area", min_value=0.0, max_value=2500.0, value=650.0)
mean_smoothness = st.sidebar.number_input("Mean Smoothness", min_value=0.0, max_value=1.0, value=0.1)

# Note: If your model requires 30 features, make sure all 30 features are passed into np.array
# For standard 30-feature Scikit-Learn dataset (filling remaining features with average values if simplified):

if st.button("Predict Diagnosis"):
    # Build input feature vector (Example with 30 features padded with defaults if needed)
    # Adjust according to how many features your model was trained on
    input_data = [mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]
    
    # If trained on full 30 features of Breast Cancer dataset:
    if len(input_data) < 30:
        # Pad remaining features with reasonable defaults for simple interface
        input_data += [0.0] * (30 - len(input_data))

    final_input = np.array([input_data])
    prediction = model.predict(final_input)[0]
    
    # Target 0: Malignant, 1: Benign (or vice versa based on sklearn default)
    st.markdown("---")
    if prediction == 0:
        st.error("⚠️ **Prediction:** Malignant (High Risk - Please Consult a Doctor)")
    else:
        st.success("✅ **Prediction:** Benign (Low Risk)")
