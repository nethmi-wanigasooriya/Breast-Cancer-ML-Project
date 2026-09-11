# Breast Cancer Diagnostic Classification & Web Predictor

An end-to-end Machine Learning project designed to predict whether a breast mass is **Malignant** (Cancerous) or **Benign** (Non-cancerous) using clinical cell features. The trained model is deployed as an interactive Web Application using **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.0%2B-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Live Web Application

Try the interactive diagnostic predictor live in your browser:  
**[Launch Breast Cancer Predictor Web App](https://breast-cancer-ml-project-n4h3zv4lpj7nrezqzag3jq.streamlit.app/)**

---

## Project Overview
Breast cancer is one of the most common cancers among women worldwide. Early and accurate diagnosis significantly increases the chances of successful treatment. 

This project implements a binary classification workflow using the **Wisconsin Breast Cancer Dataset**. It covers the complete machine learning lifecycle:
- Exploratory Data Analysis (EDA)
- Data Preprocessing & Feature Scaling
- Model Training & Evaluation
- Model Serialization (`.pkl`)
- Web Deployment via Streamlit Cloud

---

## Dataset Information
The dataset used is the standard **Breast Cancer Wisconsin (Diagnostic) Dataset** from `scikit-learn`:
- **Total Samples:** 569
- **Features:** 30 numerical computed features (e.g., radius, texture, perimeter, area, smoothness, compactness, concavity, etc.)
- **Target Classes:** 
  - `0`: Malignant (High Risk)
  - `1`: Benign (Low Risk)

---

## Workflow & Architecture

1. **Exploratory Data Analysis (EDA):** Evaluated feature distributions and correlation matrices to understand feature significance.
2. **Data Preprocessing:** Handled features matrix ($X$) and target vector ($y$), split into **80% Training** and **20% Testing** sets.
3. **Model Training & Comparison:** Evaluated Supervised ML models including **Logistic Regression** to achieve high accuracy and low false-negative rate.
4. **Model Serialization:** Exported the trained model using `joblib` into `breast_cancer_model.pkl` for fast inference.
5. **Deployment:** Built a user-friendly diagnostic web application using **Streamlit**.

---

## Tech Stack

- **Language:** Python 3.x
- **Libraries:**
  - `pandas` - Data manipulation
  - `numpy` - Array processing
  - `matplotlib` & `seaborn` - Visualizations
  - `scikit-learn` - Machine Learning algorithms & evaluation metrics
  - `joblib` - Model saving/loading
  - `streamlit` - Interactive Web UI
- **Hosting Platform:** Streamlit Community Cloud

---

## How to Run Locally

1. **Clone this repository:**
   ```bash
   git clone [https://github.com/nethmi-wanigasooriya/breast-cancer-ml-project.git](https://github.com/nethmi-wanigasooriya/breast-cancer-ml-project.git)
   cd breast-cancer-ml-project
